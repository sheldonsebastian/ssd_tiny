import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector

# Choose to use a config and initialize the detector
config = "ssd_tiny_src/oriented_rcnn_r50_fpn_1x_dota_le90.py"
# Setup a checkpoint file to load
checkpoint = "latest.pth"

# Set the device to be used for evaluation
device = "cpu"

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None
config.model.roi_head.bbox_head.num_classes = 1

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = ("ship",)

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

img = mmcv.imread("ssd_tiny_src/ssdd_tiny/images/001129.png")
result = inference_detector(model, img)

show_result_pyplot(model, img, result, score_thr=0.3, out_file="output.jpg")
