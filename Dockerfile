FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:latest

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Set environment variables
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0"

# Install openmim and mmcv-full
RUN pip3 install -U openmim
RUN mim install mmcv-full
RUN mim install mmdet

# Install MMRotate
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmrotate.git mmrotate
WORKDIR mmrotate
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN pip install --force-reinstall -v "numpy==1.25.2"
RUN pip install future tensorboard
RUN pip install azure-storage-blob

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ssd_tiny_src ssd_tiny_src