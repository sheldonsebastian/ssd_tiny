# SSD-Tiny and MMRotate on Azure ML

Welcome to the repository for training the mmrotate model on the SSD-Tiny dataset using Azure ML cloud. This setup leverages serverless GPU for efficient and scalable model training.

## Key Features

- **Serverless GPU**: Utilize Azure ML's serverless GPU capabilities for cost-effective and scalable training.
- **SSD-Tiny Dataset**: Specifically tailored for the SSD-Tiny dataset to ensure optimal performance.
- **Seamless Integration**: Easy setup and integration with Azure ML for streamlined workflows.

## Repository Hierarchy

|Name|Description|
|----|------|
|0_verify_install.ipynb| Testing notebook for checking installation|
|1_submit_ssd_job.ipynb|To submit job on serverless GPU for Azure ML|
|ssd_tiny_src/train|Training script|
|verify_install| Folder containing artifacts for verifying installation|
|2_ssd_inf.py|Inference Script for trained model|
|Dockerfile|Custom Azure ML environment for MMrotate model|
|environment.yml|Conda environment file|

## Steps to run

1. Create conda env using environment.yml file
2. Create configs.env and blob_configs.json using the respective sample files
3. Run the 1_submit_ssd_job.ipynb file to submit job.

## References

- <https://github.com/open-mmlab/mmrotate/blob/main/demo/MMRotate_Tutorial.ipynb>
