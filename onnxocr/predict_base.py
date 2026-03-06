class PredictBase(object):
    def __init__(self, model_dir, use_gpu=False, use_dml=False, use_openvino=False):
        self.is_openvino = use_openvino
        if self.is_openvino:
            import openvino as ov
            core = ov.Core()
            
            # Identify hardware
            devices = core.available_devices
            preferred_devices = []
            if "NPU" in devices: preferred_devices.append("NPU")
            # if "GPU" in devices: preferred_devices.append("GPU") GPU too slow for dynamic shapes
            preferred_devices.append("CPU")
            
            self.session = None
            self.is_npu = False
            self.force_shape = None # Track if we forced a specific shape
            
            for device in preferred_devices:
                try:
                    model = core.read_model(model=model_dir)
                    if device == "NPU":
                        try:
                            input_layer = model.inputs[0]
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
            if use_gpu:
                providers = [('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"}), 'CPUExecutionProvider']
            elif use_dml:
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            with open(model_dir, 'rb') as f:
                self.session = onnxruntime.InferenceSession(f.read(), None, providers=providers)

    def run(self, output_name, input_feed):
        if self.is_openvino:
            # NPU handle: if we are on NPU and forced a static shape
            if getattr(self, "is_npu", False) and getattr(self, "force_shape", None):
                import cv2
                import numpy as np
                
                target_b, target_c, target_h, target_w = self.force_shape
                
                # Dynamic batching for NPU (handles the case where input_feed has more than target_b)
                input_batch_size = next(iter(input_feed.values())).shape[0]
                
                if input_batch_size > target_b or any(input_feed[name].shape != self.force_shape for name in input_feed):
                    all_outputs = [[] for _ in self.session.outputs]
                    
                    for i in range(0, input_batch_size, target_b):
                        current_feed = {}
                        for name in input_feed:
                            data = input_feed[name]
                            chunk = data[i:i+target_b]
                            
                            # Handle padding if the last chunk is smaller than target_b
                            if chunk.shape[0] < target_b:
                                pad_width = [(0, target_b - chunk.shape[0])] + [(0, 0)] * (len(chunk.shape) - 1)
                                chunk = np.pad(chunk, pad_width, mode='constant')
                            
                            # Handle resizing if the chunk shape doesn't match force_shape
                            if chunk.shape != self.force_shape:
                                resized_chunk = np.zeros(self.force_shape, dtype=chunk.dtype)
                                for b in range(target_b):
                                    # chunk[b] is (C, H, W)
                                    img = chunk[b].transpose(1, 2, 0)
                                    img_resized = cv2.resize(img, (target_w, target_h))
                                    resized_chunk[b] = img_resized.transpose(2, 0, 1)
                                current_feed[name] = resized_chunk
                            else:
                                current_feed[name] = chunk
                        
                        chunk_results = self.session(inputs=current_feed)
                        for j, out in enumerate(self.session.outputs):
                            res = chunk_results[out]
                            # Take only what we need (in case we padded)
                            actual_count = min(target_b, input_batch_size - i)
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