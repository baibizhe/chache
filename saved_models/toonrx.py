import io
import numpy as np
import timm

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


def get_model(device):
    # model = monai.networks.nets.Densenet(spatial_dims=2, in_channels=3, out_channels=3).to(device)
    model_name = "mobilenetv3_small_075"
    # model_name = "tf_mobilenetv3_small_100"
    model = timm.create_model(model_name,num_classes=3,in_chans=3,pretrained=False)
    return model

def main():
    fold  =3
    for i in range(fold):
        model = get_model("cuda")
        model.load_state_dict(torch.load("fold_{}classification_model.pth".format(i),map_location="cpu"),strict=True)
        model.eval()
        dummy=torch.zeros((1,3,1296,2304))
        torch.onnx.export(model,  # model being run
                          dummy,  # model input (or a tuple for multiple inputs)
                          "class_model{}.onnx".format(i),  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        print(i)

if __name__ == '__main__':
    main()

