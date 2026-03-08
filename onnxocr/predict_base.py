import logging
import numpy as np
import cv2

class PredictBase(object):
    def __init__(self, model_dir, use_openvino=True, use_npu=True, logger=None):
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
            
            # Identify hardware
            devices = core.available_devices
            logger.info(f'openvino core.available_devices {devices}')
            preferred_devices = []
            if use_npu and "NPU" in devices:
                preferred_devices.append("NPU")
                logger.info('openvino use npu')
            # if "GPU" in devices: preferred_devices.append("GPU") GPU too slow for dynamic shapes
            preferred_devices.append("CPU")
            
            self.session = None
            self.is_npu = False
            self.force_shape = None # Track if we forced a specific shape
            
            for device in preferred_devices:
                try:
                    model = core.read_model(model=model_dir)
                    input_layer = model.inputs[0]
                    
                    if device == "NPU":
                        try:
                            # NPU often requires static shapes. 
                            # Different shapes for detection vs recognition
                            if "det" in model_dir.lower():
                                self.force_shape = (1, 3, 640, 640)
                            elif "rec" in model_dir.lower():
                                self.force_shape = (1, 3, 48, 320)
                            elif "cls" in model_dir.lower():
                                self.force_shape = (1, 3, 48, 192)

                            if self.force_shape:
                                model.reshape({input_layer.any_name: self.force_shape})
                        except Exception:
                            self.force_shape = None 
                    else:
                        # For CPU, ensure dynamic shapes are allowed
                        try:
                            model.reshape({input_layer.any_name: [-1, 3, -1, -1]})
                        except Exception:
                            pass
                    
                    self.session = core.compile_model(model=model, device_name=device)
                    if device == "NPU":
                        self.is_npu = True
                    break
                except Exception as e:
                    self.force_shape = None
                    if device == preferred_devices[-1]: raise e
                    print(f"[Warning] Failed to compile model on {device}: {e}. Trying {preferred_devices[preferred_devices.index(device)+1]}...")
        else:
            import onnxruntime
            providers = ['CPUExecutionProvider']

            with open(model_dir, 'rb') as f:
                self.session = onnxruntime.InferenceSession(f.read(), None, providers=providers)

    def run(self, output_name, input_feed):
        if self.is_openvino:
            # Handle NPU/Static shape force
            if getattr(self, "is_npu", False) and getattr(self, "force_shape", None):
                target_b, target_c, target_h, target_w = self.force_shape
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
                            processed_chunk = np.zeros(self.force_shape, dtype=chunk.dtype)
                            for b in range(target_b):
                                img = chunk[b].transpose(1, 2, 0) # HWC
                                oh, ow = img.shape[:2]
                                
                                # Letterbox: keep aspect ratio
                                scale = min(target_w / ow, target_h / oh)
                                nw, nh = int(ow * scale), int(oh * scale)
                                
                                # Use safe resize
                                if nw > 0 and nh > 0:
                                    img_resized = cv2.resize(img, (nw, nh))
                                    processed_chunk[b, :, :nh, :nw] = img_resized.transpose(2, 0, 1)
                                
                                if name == next(iter(input_feed)):
                                    batch_resize_info.append((oh, ow, nh, nw))
                            current_feed[name] = processed_chunk
                        else:
                            current_feed[name] = chunk
                            if name == next(iter(input_feed)):
                                for _ in range(target_b):
                                    batch_resize_info.append((target_h, target_w, target_h, target_w))
                    
                    chunk_results = self.session(inputs=current_feed)
                    actual_count = min(target_b, input_batch_size - i)
                    
                    for j, out_node in enumerate(self.session.outputs):
                        res = chunk_results[out_node]
                        
                        # If output is a heatmap (4D), resize it back to original input size
                        if len(res.shape) == 4 and res.shape[2:] == (target_h, target_w):
                            restored_list = []
                            for b in range(actual_count):
                                oh, ow, nh, nw = batch_resize_info[b]
                                heatmap = res[b, :, :nh, :nw].transpose(1, 2, 0) # (nh, nw, C)
                                heatmap_restored = cv2.resize(heatmap, (ow, oh))
                                if len(heatmap_restored.shape) == 2:
                                    heatmap_restored = heatmap_restored[..., None]
                                restored_list.append(heatmap_restored.transpose(2, 0, 1))
                            all_outputs[j].append(np.array(restored_list))
                        else:
                            all_outputs[j].append(res[:actual_count])
                
                return [np.concatenate(outs, axis=0) for outs in all_outputs]

            result_dict = self.session(inputs=input_feed)
            return [result_dict[out] for out in self.session.outputs]
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