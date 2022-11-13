import torch.onnx
import onnx
from onnx import shape_inference, numpy_helper
from utils.load_model import load_feature_extractor, load_anomaly_detector, load_models
from video_demo import ad_prediction
from AD_live_prediction import features_extraction
from network.anomaly_detector_model import AnomalyDetector

# For 3D recognizer(e.g. I3D), the input should be $batch $clip $channel $time $height $width(e.g. 1 1 3 32 224 224)

'''
Function to convert model to ONNX
ONNX: 1. feature_extractor / 2. anomaly_detector / 3. check_model
'''

def Convert_C3D_ONNX(model):
    model.eval()
    batch_size = 1
    # feature_extraction.onnx [batch_size, channels, depth, height, width]
    dummy_input = torch.randn(batch_size, 3, 16, 112, 112)


    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/KHUCV_AnomalyDetection/RUN64/I3D_feature_extractor.onnx",
                      # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('C3D Feature extractor model has been converted to ONNX')

def Convert_I3D_RGB_ONNX(model):
    model.eval()
    # (batch, channel, t, h, w)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 16, 240, 320)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "C:/Users/davi/Desktop\Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos/KHUCV_AnomalyDetection/RUN64/i3d(RGB)_feature_extractor.onnx",
                      # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('I3D(RGB) Feature extractor model has been converted to ONNX')

def Convert_I3D_flow_ONNX(model):
    model.eval()
    # (batch, channel, t, h, w)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 2, 16, 224, 224)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "C:/Users/davi/Desktop\Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos/KHUCV_AnomalyDetection/RUN64/i3d(flow)_feature_extractor.onnx",
                      # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('I3D(flow) Feature extractor model has been converted to ONNX')

def Convert_AnomalyDetector_ONNX_1(model):
    model.eval()
    # anomaly_detector.onnx
    dummy_input = torch.randn(1, 32)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "C:/Users/davi/Desktop/Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos/KHUCV_AnomalyDetection/RUN64/anomaly_detector.onnx",
                      # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Anomaly detector model has been converted to ONNX')

def Convert_AnomalyDetector_ONNX_2(model):
    model.eval()
    # anomaly_detector.onnx
    dummy_input = torch.randn(32, 4096)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "C:/Users/davi/Desktop/Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos/KHUCV_AnomalyDetection/RUN64/anomaly_detector_new.onnx",
                      # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Anomaly detector model has been converted to ONNX')


if __name__ == "__main__":
    menu_num = int(input(
        'Choose the number of model to export ONNX: 1. C3D / 2. I3D(RGB) / 3. I3D(flow) / 4. Anomaly detector'))

    if menu_num == 1:
        # feature_extractor.onnx
        model = load_feature_extractor('c3d',
                                       'C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/pretrained/c3d.pickle',
                                       'cpu')
        Convert_C3D_ONNX(model)

    if menu_num == 2:
        # feature_extractor.onnx
        model = load_feature_extractor('i3d_RGB',
                                       'C:/Users/davi/Desktop/Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos\AnomalyDetectionCVPR2018-Pytorch-master/pretrained/models/rgb_imagenet.pt',
                                       'cpu')

        Convert_I3D_RGB_ONNX(model)

    if menu_num == 3:
        # feature_extractor.onnx
        model = load_feature_extractor('i3d_flow',
                                       'C:/Users/davi/Desktop/Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos\AnomalyDetectionCVPR2018-Pytorch-master/pretrained/models/flow_charades.pt',
                                       'cpu')

        Convert_I3D_flow_ONNX(model)

    if menu_num == 4:
        # anomaly_detector.onnx
        model = load_anomaly_detector(
            'C:/Users/davi/Desktop/Davi/[CVPR 2018] Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/exps/c3d_models_155/epoch_155.pt',
            'cpu')
        Convert_AnomalyDetector_ONNX_1(model)

    # else:
    #     # check onnx
    #     onnx_model = onnx.load(
    #         "C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/onnx/anomaly_detector.onnx")
    #     onnx.save(onnx.shape_inference.infer_shapes(onnx.load(
    #         "C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/onnx/anomaly_detector.onnx")),
    #               "C:/Users/davi/Desktop/Davi/Real-world-Anomaly-Detection-in-Surveillance-Videos/AnomalyDetectionCVPR2018-Pytorch-master/onnx/anomaly_detector.onnx")
    #     onnx.checker.check_model(onnx_model)
    #     graph = onnx_model.graph
    #     initializers = dict()
    #     for init in graph.initializer:
    #         initializers[init.name] = numpy_helper.to_array(init)
    #     print(initializers.keys())

        # ort_session = onnxruntime.InferenceSession("feathernets.onnx")
        #
        #
        # def to_numpy(tensor):
        #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        #     # ONNX 런타임에서 계산된 결과값 ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")