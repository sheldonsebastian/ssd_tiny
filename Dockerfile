FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:18

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
