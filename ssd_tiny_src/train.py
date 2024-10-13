from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmcv import Config
from mmdet.apis import set_random_seed
import os.path as osp
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import mmcv
import os
from azure.storage.blob import BlobServiceClient
import uuid
import json


@ROTATED_DATASETS.register_module()
class TinyDataset(DOTADataset):
    """SAR ship dataset for detection."""

    CLASSES = ("ship",)


cfg = Config.fromfile("oriented_rcnn_r50_fpn_1x_dota_le90.py")

# Modify dataset type and path
cfg.dataset_type = "TinyDataset"
cfg.data_root = "ssdd_tiny/"

cfg.data.test.type = "TinyDataset"
cfg.data.test.data_root = "ssdd_tiny/"
cfg.data.test.ann_file = "val"
cfg.data.test.img_prefix = "images"

cfg.data.train.type = "TinyDataset"
cfg.data.train.data_root = "ssdd_tiny/"
cfg.data.train.ann_file = "train"
cfg.data.train.img_prefix = "images"

cfg.data.val.type = "TinyDataset"
cfg.data.val.data_root = "ssdd_tiny/"
cfg.data.val.ann_file = "val"
cfg.data.val.img_prefix = "images"

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = "oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth"

# Set up working dir to save files and logs.
unique_id = uuid.uuid4()
cfg.work_dir = f"./outputs_{unique_id}"

cfg.optimizer.lr = 0.001
cfg.lr_config.warmup = None
cfg.runner.max_epochs = 3
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = "mAP"
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 3

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(0)
cfg.device = "cuda"

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f"Config:\n{cfg.pretty_text}")


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

print("Finished Training and now uploading to Azure Blob Storage")


def upload_to_azure_blob_storage(local_path, container_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            blob_path = os.path.relpath(file_path, local_path)
            blob_client = container_client.get_blob_client(blob_path)

            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {file_path} to {blob_path}")


configs_json_path = "blob_configs.json"
with open(configs_json_path) as f:
    blob_cfg = json.load(f)

# Define your Azure Blob Storage connection string and container name
connection_string = blob_cfg["CONNECTION_STR"]
container_name = blob_cfg["CONTAINER_NAME"]
local_output_path = cfg.work_dir

# Upload all files in the output directory to Azure Blob Storage
upload_to_azure_blob_storage(local_output_path, container_name, connection_string)
