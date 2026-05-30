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
RUN ${CONDA_DIR}/bin/conda install -y -n ${CONDA_ENV} -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.6 && \
    ${CONDA_DIR}/bin/conda clean -ya

# Install common Python build deps and pip
RUN ${CONDA_DIR}/bin/conda install -y -n ${CONDA_ENV} pip cython numpy pyyaml scipy matplotlib scikit-image cffi && \
    ${CONDA_DIR}/bin/conda clean -ya

# Create workspace and copy repository (assumes build context includes repo root)
WORKDIR /workspace
COPY . /workspace

# Run repository dependency installer and build native extensions.
RUN chmod +x ./scripts/install_deps.sh ./scripts/compile_all.sh
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./scripts/install_deps.sh
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./core/csrc/uncertainty_pnp/build_ceres.sh
RUN ${CONDA_DIR}/bin/conda run -n ${CONDA_ENV} bash ./scripts/compile_all.sh

ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${PATH}"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
