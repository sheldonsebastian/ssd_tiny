FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:latest

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Set environment variables
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0"

# Install openmim and mmcv-full
RUN pip3 install -U openmim
RUN mim install mmcv-full
RUN mim install mmdet

# Install MMRotate
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmrotate.git /mmrotate
WORKDIR /mmrotate
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
