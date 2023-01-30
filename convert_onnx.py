import torch
import torchvision
from unet import UNet

model_dir = "checkpoints_/checkpoint_epoch10.pth"
out_dir = "vanilla_unet.onnx"
net = UNet(n_channels=1, n_classes=2, bilinear=False)
net.load_state_dict(torch.load(model_dir))
net.eval()

input_var = torch.rand(1,1,120,320)
torch.onnx.export(net, input_var, out_dir, verbose=True, export_params=True,opset_version=11)
