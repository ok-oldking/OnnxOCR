import logging
import hashlib
import threading
from pathlib import Path

import numpy as np
import cv2


def _model_cache_path(model_dir, backend, suffix):
    """Return a collision-safe cache path rooted in the working directory."""
    model_path = Path(model_dir).resolve()
    path_hash = hashlib.sha256(str(model_path).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path.cwd() / "cache" / backend
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_path.parent.name}-{model_path.stem}-{path_hash}{suffix}"


def _cache_is_current(model_path, cache_path):
    """A changed source model must not reuse an older optimized model."""
    return (
        cache_path.is_file()
        and cache_path.stat().st_mtime_ns >= model_path.stat().st_mtime_ns
    )


class PredictBase(object):
    def __init__(
        self,
        model_dir,
        use_openvino=True,
        use_npu=True,
        logger=None,
        force_static_shape=False,
        openvino_num_requests=1,
    ):
        self.is_openvino = use_openvino
        if logger is None:
            logger = logging.getLogger('onnxocr')
            if not logger.handlers:
                logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        self.logger = logger
        if self.is_openvino:
            import openvino as ov
            core = ov.Core()
            cache_dir = Path.cwd() / "cache" / "openvino"
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
            logger.info(f'openvino cache_dir: {cache_dir}')
            
            # Do not call core.available_devices here.  That API enumerates all
            # registered plugins and can initialize the GPU plugin even though
            # OCR is compiled for CPU/NPU only.  Probe NPU directly when it was
            # explicitly requested, then fall back to CPU.
            preferred_devices = []
            if use_npu:
                try:
                    npu_devices = core.get_property("NPU", "AVAILABLE_DEVICES")
                except Exception as exc:
                    npu_devices = []
                    logger.info(f'openvino npu unavailable: {exc}')

                if npu_devices:
                    from .check_npu import check_npu_driver_valid
                    npu_driver_valid = check_npu_driver_valid(logger)

                    if npu_driver_valid:
                        preferred_devices.append("NPU")
                        logger.info('openvino use npu')

            preferred_devices.append("CPU")
            logger.info(
                f'openvino device candidates {preferred_devices}; '
                'gpu discovery disabled'
            )
            
            self.session = None
            self.is_npu = False
            self.force_shape = None # Track if we forced a specific shape
            
            for device in preferred_devices:
                try:
                    model = core.read_model(model=model_dir)
                    input_layer = model.inputs[0]
                    
                    if device == "NPU" or force_static_shape:
                        try:
                            # NPU often requires static shapes. 
                            # Different shapes for detection vs recognition
                            if "det" in model_dir.lower():
                                self.force_shape = (1, 3, 960, 960)
                            elif "rec" in model_dir.lower():
                                self.force_shape = (1, 3, 48, 960)
                            elif "cls" in model_dir.lower():
                                self.force_shape = (1, 3, 48, 192)

                            if self.force_shape:
                                self.logger.info(f'openvino force shape {model_dir} for {device} from {input_layer.get_partial_shape()} to {self.force_shape}')
                                model.reshape({input_layer.any_name: self.force_shape})                                
                        except Exception as e:
                            self.logger.warning(f"Failed to reshape model for {device}: {e}")
                            self.force_shape = None 
                    else:
                        # For CPU, ensure dynamic shapes are allowed
                        try:
                            model.reshape({input_layer.any_name: [-1, 3, -1, -1]})
                        except Exception:
                            pass
                    
                    self.session = core.compile_model(model=model, device_name=device)
                    self.model_dir = model_dir # Store for identifying model type in run()
                    if device == "NPU":
                        self.is_npu = True
                    break
                except Exception as e:
                    self.force_shape = None
                    if device == preferred_devices[-1]: raise e
                    self.logger.warning(f"Failed to compile model on {device}: {e}. Trying next...")

            # CompiledModel.__call__ reuses a synchronous InferRequest.  Use an
            # AsyncInferQueue instead so the same OCR model can safely serve
            # concurrent callers with separate requests.
            self._async_queue = ov.AsyncInferQueue(
                self.session, jobs=openvino_num_requests
            )
            self._async_submit_lock = threading.Lock()
            self._async_queue.set_callback(self._on_openvino_complete)
            logger.info(
                f'openvino async infer requests: {len(self._async_queue)}'
            )
        else:
            import onnxruntime
            providers = ['CPUExecutionProvider']
            model_path = Path(model_dir).resolve()
            cache_path = _model_cache_path(
                model_path, "onnxruntime", ".optimized.onnx"
            )

            if _cache_is_current(model_path, cache_path):
                try:
                    self.session = onnxruntime.InferenceSession(
                        str(cache_path), providers=providers
                    )
                    logger.info(f'onnxruntime loaded cached model: {cache_path}')
                except Exception as exc:
                    logger.warning(
                        f'Failed to load cached ONNX model {cache_path}: {exc}. '
                        'Rebuilding cache.'
                    )
                    cache_path.unlink(missing_ok=True)
                    self.session = self._create_onnxruntime_session(
                        onnxruntime, model_path, cache_path, providers
                    )
            else:
                self.session = self._create_onnxruntime_session(
                    onnxruntime, model_path, cache_path, providers
                )

    def _create_onnxruntime_session(
        self, onnxruntime, model_path, cache_path, providers
    ):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.optimized_model_filepath = str(cache_path)
        session = onnxruntime.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        self.logger.info(f'onnxruntime cached optimized model: {cache_path}')
        return session

    @staticmethod
    def _on_openvino_complete(infer_request, job):
        try:
            # InferRequests are reused as soon as the callback returns, so each
            # caller must own a copy of its outputs.
            job["outputs"] = [
                np.array(tensor.data, copy=True)
                for tensor in infer_request.output_tensors
            ]
        except BaseException as exc:
            job["error"] = exc
        finally:
            job["done"].set()

    def _run_openvino_async(self, input_feed):
        job = {"done": threading.Event(), "outputs": None, "error": None}

        # AsyncInferQueue handles request reuse. Serialize only submission so
        # concurrent Python callers cannot race its flow-control operation.
        with self._async_submit_lock:
            self._async_queue.start_async(
                input_feed, userdata=job, share_inputs=False
            )

        job["done"].wait()
        if job["error"] is not None:
            raise RuntimeError(
                "OpenVINO asynchronous inference failed"
            ) from job["error"]
        return job["outputs"]

    def run(self, output_name, input_feed):
        if self.is_openvino:
            # Handle NPU/Static shape force
            force_shape = getattr(self, "force_shape", None)
            if (getattr(self, "is_npu", False) or force_shape) and force_shape:
                target_b, target_c, target_h, target_w = force_shape
                input_batch_size = next(iter(input_feed.values())).shape[0]
                
                all_outputs = [[] for _ in self.session.outputs]
                
                for i in range(0, input_batch_size, target_b):
                    current_feed = {}
                    # Meta info to resize detection heatmaps back
                    # This must be synchronized across all inputs in the batch
                    batch_resize_info = [] # (orig_h, orig_w, new_h, new_w)
                    
                    for name in input_feed:
                        data = input_feed[name]
                        chunk = data[i:i+target_b]
                        
                        # Pad batch if needed
                        if chunk.shape[0] < target_b:
                            pad_width = [(0, target_b - chunk.shape[0])] + [(0, 0)] * (len(chunk.shape) - 1)
                            chunk = np.pad(chunk, pad_width, mode='constant')
                        
                        # Adapt spatial shape
                        if chunk.shape[2:] != (target_h, target_w):
                            processed_chunk = np.zeros(force_shape, dtype=chunk.dtype)
                            for b in range(target_b):
                                img = chunk[b].transpose(1, 2, 0) # HWC
                                oh, ow = img.shape[:2]
                                
                                # Letterbox: keep aspect ratio
                                scale = min(target_w / ow, target_h / oh)
                                nw, nh = int(ow * scale), int(oh * scale)
                                
                                # Use safe resize
                                if nw > 0 and nh > 0:
                                    if nw == ow and nh == oh:
                                        img_resized = img
                                    else:
                                        img_resized = cv2.resize(img, (nw, nh))
                                    
                                    # Identify model type for appropriate padding
                                    model_dir_lower = getattr(self, "model_dir", "").lower()
                                    is_rec_or_cls = "rec" in model_dir_lower or "cls" in model_dir_lower
                                    
                                    if is_rec_or_cls:
                                        # For recognition/classification, constant padding is better to avoid smearing text
                                        # value=0 matches the padding used in predict_rec.py
                                        img_padded = cv2.copyMakeBorder(img_resized, 0, target_h - nh, 0, target_w - nw, cv2.BORDER_CONSTANT, value=0)
                                    else:
                                        # Use BORDER_REPLICATE for detection to avoid sharp edges at the bottom/right padding
                                        img_padded = cv2.copyMakeBorder(img_resized, 0, target_h - nh, 0, target_w - nw, cv2.BORDER_REPLICATE)
                                        
                                    processed_chunk[b] = img_padded.transpose(2, 0, 1)
                                
                                if name == next(iter(input_feed)):
                                    batch_resize_info.append((oh, ow, nh, nw))
                            current_feed[name] = processed_chunk
                        else:
                            current_feed[name] = chunk
                            if name == next(iter(input_feed)):
                                for _ in range(target_b):
                                    batch_resize_info.append((target_h, target_w, target_h, target_w))
                    
                    chunk_results = self._run_openvino_async(current_feed)
                    actual_count = min(target_b, input_batch_size - i)
                    
                    for j, res in enumerate(chunk_results):
                        
                        # Robust layout check: OpenVINO sometimes returns NHWC on certain hardware/drivers
                        # but our post-processing expects NCHW.
                        if len(res.shape) == 4 and res.shape[1] > res.shape[3] and (res.shape[3] == 1 or res.shape[3] == 3):
                            # Likely NHWC (N, H, W, C), transpose to NCHW (N, C, H, W)
                            res = res.transpose(0, 3, 1, 2)
                        
                        # Recognition models expect 3D output (N, L, C)
                        # If OpenVINO returns 4D (N, L, 1, C) or (N, 1, L, C), squeeze it
                        if "rec" in getattr(self, "model_dir", "").lower() and len(res.shape) == 4:
                            # Squeeze any dimension of size 1
                            if res.shape[1] == 1:
                                res = np.squeeze(res, axis=1)
                            elif res.shape[2] == 1:
                                res = np.squeeze(res, axis=2)

                        # If output is a heatmap (4D), resize it back to original input size
                        if len(res.shape) == 4:
                            # Handle potential stride (e.g. output is H/stride, W/stride)
                            out_h, out_w = res.shape[2:]
                            ratio_h = out_h / target_h
                            ratio_w = out_w / target_w
                            
                            restored_list = []
                            for b in range(actual_count):
                                oh, ow, nh, nw = batch_resize_info[b]
                                s_nh, s_nw = int(nh * ratio_h), int(nw * ratio_w)
                                
                                # Extract only the relevant part of the heatmap
                                heatmap = res[b, :, :s_nh, :s_nw].transpose(1, 2, 0) # (s_nh, s_nw, C)
                                if (s_nh == oh and s_nw == ow):
                                    heatmap_restored = heatmap
                                else:
                                    heatmap_restored = cv2.resize(heatmap, (ow, oh))
                                if len(heatmap_restored.shape) == 2:
                                    heatmap_restored = heatmap_restored[..., None]
                                restored_list.append(heatmap_restored.transpose(2, 0, 1))
                            all_outputs[j].append(np.array(restored_list))
                        else:
                            all_outputs[j].append(res[:actual_count])
                
                return [np.concatenate(outs, axis=0) for outs in all_outputs]

            outputs = self._run_openvino_async(input_feed)
            
            # Robust layout check: OpenVINO sometimes returns NHWC on certain hardware/drivers
            # but our post-processing expects NCHW.
            processed_outputs = []
            for res in outputs:
                if len(res.shape) == 4 and res.shape[1] > res.shape[3] and (res.shape[3] == 1 or res.shape[3] == 3):
                    # Likely NHWC (N, H, W, C), transpose to NCHW (N, C, H, W)
                    processed_outputs.append(res.transpose(0, 3, 1, 2))
                elif "rec" in getattr(self, "model_dir", "").lower() and len(res.shape) == 4:
                    # Squeeze recognition output if it's 4D
                    if res.shape[1] == 1:
                        processed_outputs.append(np.squeeze(res, axis=1))
                    elif res.shape[2] == 1:
                        processed_outputs.append(np.squeeze(res, axis=2))
                    else:
                        processed_outputs.append(res)
                else:
                    processed_outputs.append(res)
            return processed_outputs
        else:
            return self.session.run(output_name, input_feed)

    def get_output_name(self):
        output_name = []
        if self.is_openvino:
            for node in self.session.outputs:
                output_name.append(node.get_any_name())
        else:
            for node in self.session.get_outputs():
                output_name.append(node.name)
        return output_name

    def get_input_name(self):
        input_name = []
        if self.is_openvino:
            for node in self.session.inputs:
                input_name.append(node.get_any_name())
        else:
            for node in self.session.get_inputs():
                input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
