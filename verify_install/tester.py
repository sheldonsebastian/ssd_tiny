from mmdet.apis import init_detector, inference_detector
import mmrotate
import torch
import warnings

warnings.filterwarnings("ignore")
print(f"MMRotate version {mmrotate.__version__}")

# download model using: `mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .`
config_file = "oriented_rcnn_r50_fpn_1x_dota_le90.py"
checkpoint_file = "oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")

model = init_detector(config_file, checkpoint_file, device=device)
response = inference_detector(model, "demo.jpg")

print(response)
