FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Basic build deps + OpenCV runtime libs + Eigen + BLAS/LAPACK (needed for CLIPPER/SCS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget curl ripgrep \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libeigen3-dev \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*
 # added L14 until L23. From 18 onwards is for display 

# Make sure any git@github.com:... URL becomes HTTPS (no username prompt)
RUN git config --global url."https://github.com/".insteadOf git@github.com: \
 && git config --global url."https://github.com/".insteadOf ssh://git@github.com/ \
 && git config --global url."https://github.com/".insteadOf git://github.com/

# CUDA env for building MSDeformAttn
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Base image already has torch==2.4.0 CUDA 11.8
# Install a matching torchvision + OpenCV
RUN pip install --no-cache-dir \
    "torchvision==0.19.0" \
    "opencv-python"

#############################
# Detectron2
#############################

RUN git clone https://github.com/facebookresearch/detectron2.git \
 && cd detectron2 \
 && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir \
      "git+https://github.com/cocodataset/panopticapi.git" \
      "git+https://github.com/mcordts/cityscapesScripts.git"

#############################
# MaskDINO + MSDeformAttn CUDA kernel
#############################
WORKDIR /workspace

#RUN git clone https://github.com/IDEA-Research/MaskDINO.git \
# && cd MaskDINO \

COPY MaskDINO /workspace/MaskDINO

RUN cd MaskDINO \
 && pip install --no-cache-dir -r requirements.txt \
 && cd maskdino/modeling/pixel_decoder/ops \
 && TORCH_CUDA_ARCH_LIST="6.1" FORCE_CUDA=1 python setup.py build install

#############################
# Install ROMAN
#############################

WORKDIR /workspace

# Build & install GTSAM (dependency for Kimera-RPGO)
RUN git clone https://github.com/borglab/gtsam.git \
 && cd gtsam \
 && mkdir build && cd build \
 && cmake .. \
      -DGTSAM_POSE3_EXPMAP=ON \
      -DGTSAM_ROT3_EXPMAP=ON \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
      -DGTSAM_BUILD_TESTS=OFF \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig


WORKDIR /workspace

#RUN git clone https://github.com/mit-acl/roman.git

COPY roman /workspace/roman



# Add demo data for ROMAN
# ADD demo_data/ demo_data/

#############################
# MaskDINO data / config
#############################

#WORKDIR /workspace/MaskDINO

#ADD office.jpeg office.jpeg
#ADD MaskDINO-ADE20K.pth maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth
#RUN cp configs/ade20k/semantic-segmentation/* .

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]


ENV DISPLAY=:0

CMD ["/bin/bash"]

