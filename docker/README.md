# gdrnpp Docker image

This folder contains a Dockerfile to build an image suitable for running the `gdrnpp` codebase with CUDA, PyTorch and detector compilation.

Usage

Build example (default: CUDA 11.6 base image, Ubuntu 20.04):

```bash
docker build -t gdrnpp:cuda11.6 -f docker_support/gdrnpp/Dockerfile .
```

Override the base image to target a different CUDA / Ubuntu combination:

```bash
docker build --build-arg BASE_IMAGE=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 -t gdrnpp:cuda10.2 -f Dockerfile .
docker build --build-arg BASE_IMAGE=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 -t gdrnpp:cuda10.2 -f docker_support/gdrnpp/Dockerfile .
```

Notes
- The Dockerfile installs Miniconda and creates a conda environment named `gdrnpp`.
- It attempts to install `pytorch` and `torchvision` via `conda` (channels `pytorch` and `conda-forge`). If you need a specific PyTorch+CUDA build use a matching `BASE_IMAGE` or override installation steps.
- The Dockerfile will run `./scripts/install_deps.sh` and `./scripts/compile_all.sh` if present in the build context. Ensure these scripts exist and are executable in your repository root.
- `detectron2` is installed from the official GitHub repo (source) — adjust or pin versions if needed.

Recommended build workflow

1. Build from the repository root so the Dockerfile can copy the project sources and the entrypoint script.
2. Build the image using one of the commands above.
3. Run an interactive container:

```bash
docker run --gpus all -it --rm gdrnpp:cuda11.6
```

The container now activates the `gdrnpp` conda environment automatically on start, so you do not need to run `conda init` manually inside a throwaway container.

If you want, I can refine the Dockerfile to pin exact PyTorch versions, support automated mapping from CUDA versions to conda cudatoolkit, or split into a multi-stage build. Want that? 
