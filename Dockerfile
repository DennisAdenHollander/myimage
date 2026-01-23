FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

#RUN apt-get update && apt-get install -y --no-install-recommends \
#    ca-certificates gnupg \ 
# && rm -rf /var/lib/apt/lists/*



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
 # From line 18 onwards is for display 

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
# Clone repo
#############################
ARG REPO_URL="https://github.com/DennisAdenHollander/myimage.git"
ARG REPO_REF="main"

RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /tmp/repo

#############################
# Place components in /workspace
#############################
RUN mv /tmp/repo/MaskDINO /workspace/MaskDINO \
 && mv /tmp/repo/roman /workspace/roman \
 && mv /tmp/repo/entrypoint.sh /workspace/entrypoint.sh \
 && chmod +x /workspace/entrypoint.sh \
 && rm -rf /tmp/repo

RUN chmod +x /workspace/entrypoint.sh \
 && chmod +x /workspace/roman/roman_mount/runs/all_runs.sh || true

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
RUN cd /workspace/MaskDINO \
 && pip install --no-cache-dir -r requirements.txt \
 && cd maskdino/modeling/pixel_decoder/ops \
 && TORCH_CUDA_ARCH_LIST="6.1" FORCE_CUDA=1 python setup.py build install

#############################
# GTSAM (dependency for Kimera-RPGO / ROMAN)
#############################
RUN git clone --depth 1 https://github.com/borglab/gtsam.git \
 && cd gtsam \
 && mkdir build && cd build \
 && cmake .. \
      -DGTSAM_POSE3_EXPMAP=ON \
      -DGTSAM_ROT3_EXPMAP=ON \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
      -DGTSAM_BUILD_TESTS=OFF \
 && make -j2 \
 && make install \
 && ldconfig

WORKDIR /workspace
RUN mkdir -p /workspace/roman/roman_mount/Odom_results/{FastSAM,MaskDINO,MaskDINO+Plane,MaskDINO+Plane+Class,MaskDINO+Plane+Class+Conf}

WORKDIR /workspace
RUN mkdir -p /workspace/roman/roman_mount/VIO_results/{FastSAM,MaskDINO,MaskDINO+Plane,MaskDINO+Plane+Class,MaskDINO+Plane+Class+Conf}

WORKDIR /workspace

#############################
# Entrypoint
#############################
ENV DISPLAY=:0

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]

