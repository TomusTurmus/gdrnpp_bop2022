# Build args:
#  BASE_IMAGE - a CUDA-enabled base image (default targets CUDA 11.6 + Ubuntu 20.04)
#  PYTHON_VERSION - Python version to install into conda env (default 3.8)
#  CONDA_ENV - name of the conda environment
ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ARG PYTHON_VERSION=3.8
ARG CONDA_ENV=gdrnpp
FROM ${BASE_IMAGE}

# Re-declare build args after FROM so they are available during build
ARG PYTHON_VERSION
ARG CONDA_ENV
# Export as environment variables so RUN steps can reference them
ENV CONDA_ENV=${CONDA_ENV}
ENV PYTHON_VERSION=${PYTHON_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates git curl \
    gcc-10 g++-10 cmake openexr libopenexr-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 libgtk2.0-dev qt5-default ffmpeg procps \
    pkg-config libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libglvnd0 libglx0 libegl1 libglvnd-dev \
    libgoogle-glog-dev libeigen3-dev libsuitesparse-dev libatlas-base-dev \
    python3-distutils python3-apt sudo unzip && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -ya

ENV PATH=${CONDA_DIR}/bin:${PATH}

# Accept Anaconda channel terms up front so conda can resolve packages non-interactively.
RUN ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda env with requested python
RUN ${CONDA_DIR}/bin/conda create -y -n ${CONDA_ENV} python=${PYTHON_VERSION} && \
    ${CONDA_DIR}/bin/conda clean -ya

SHELL ["/bin/bash", "-lc"]

# Install into the target environment without relying on shell activation.
# PIN the torch family: an unpinned install resolves to the latest PyTorch
# (2.4.x), which has NO CUDA-11.6 build and silently falls back to CPU-only.
# torch 1.13.1 / torchvision 0.14.1 are the CUDA-11.6-era pair this repo expects.
RUN ${CONDA_DIR}/bin/conda install -y -n ${CONDA_ENV} -c pytorch -c nvidia \
    pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 && \
    ${CONDA_DIR}/bin/conda clean -ya

# Install common Python build deps and pip. numpy pinned <2 to match torch 1.13
# (numpy 2.x triggers the "compiled against API version 0xf" ABI warning).
RUN ${CONDA_DIR}/bin/conda install -y -n ${CONDA_ENV} pip cython "numpy=1.23" pyyaml scipy matplotlib scikit-image cffi && \
    ${CONDA_DIR}/bin/conda clean -ya

# pip >=24.1 strictly enforces PEP 440 and REJECTS packages with invalid metadata.
# pytorch-lightning 1.x declares `torch (>=1.9.*)` (illegal: .* only with ==/!=),
# so pip 24.1+ makes it uninstallable -> ResolutionImpossible -> the whole
# requirements install aborts. Pin pip <24.1 so legacy metadata is tolerated.
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} python -m pip install "pip<24.1"

# Create workspace and copy repository (assumes build context includes repo root)
WORKDIR /workspace
COPY . /workspace

# Apply the torch lock to EVERY subsequent pip install (build + runtime container).
# Any package demanding a different torch now fails loudly instead of silently
# pulling torch 2.4.1+cu121 and breaking torchvision/detectron2.
ENV PIP_CONSTRAINT=/workspace/requirements/constraints.txt

# Run repository dependency installer (apt + pip -r requirements.txt).
RUN chmod +x ./scripts/install_deps.sh ./scripts/compile_all.sh
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./scripts/install_deps.sh

# Guard: install_deps.sh has no `set -e`, so a partial requirements install would
# pass silently. Fail the build here if any critical dep is missing or torch moved.
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} python -c "\
import torch, loguru, setproctitle, pytorch_lightning, timm, transforms3d, albumentations, open3d, cv2; \
assert torch.__version__.startswith('1.13.1'), 'torch moved: '+torch.__version__; \
from pytorch_lightning.lite.wrappers import _LiteModule; \
print('requirements OK, torch', torch.__version__, 'pl', pytorch_lightning.__version__)"

# CUDA-matched extras, kept out of requirements.txt so pip cannot resolve them
# against a newer torch. mmcv-full is a prebuilt wheel for torch 1.13.1 / CUDA
# 11.6 (no source build, no CPU fallback). detectron2 has no prebuilt wheel for
# this combo, so it is built from source (v0.6) against the installed torch.
# pytorch3d is intentionally NOT installed: it is listed in requirements.txt but
# never imported anywhere in core/lib/ref, and its wheel install is fragile.
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} pip install --no-cache-dir \
    "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/detectron2.git@v0.6"

# Native extensions (build against the now-correct CUDA torch).
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./core/csrc/uncertainty_pnp/build_ceres.sh
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./scripts/compile_all.sh

# pyassimp (Python) dlopen()s the native assimp library at import time. The repo's
# install_deps.sh tries to apt-install it but runs after the package lists were
# cleared, so it silently fails -> "assimp library not found". Install it here.
RUN apt-get update && apt-get install -y --no-install-recommends libassimp-dev && \
    rm -rf /var/lib/apt/lists/*

# Pin Pillow < 10 as the LAST step so nothing can override it. detectron2 0.6 (and
# parts of this repo) use the `Image.LINEAR` alias, which was removed in Pillow 10
# -> "module 'PIL.Image' has no attribute 'LINEAR'". 9.5.0 keeps the alias.
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} pip install --no-cache-dir \
    --force-reinstall "Pillow==9.5.0"

ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${PATH}"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
