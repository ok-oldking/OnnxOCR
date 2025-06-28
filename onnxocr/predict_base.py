class PredictBase(object):
    def __init__(self, model_dir, use_gpu=False, use_dml=False, use_openvino=False):
        self.is_openvino = use_openvino
        if self.is_openvino:
            import openvino as ov
            core = ov.Core()
            model = core.read_model(model=model_dir)
            self.session = core.compile_model(model=model, device_name="CPU")
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