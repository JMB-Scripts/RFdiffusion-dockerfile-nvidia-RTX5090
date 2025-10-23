#-----------------------------------------------------------------------
# STAGE 1: THE "BUILDER"
#
# This stage installs all development tools (CUDA toolkit, compilers, etc.)
# and builds the complete Python environment including compiling DGL from source.
# It will be large, but we only copy the necessary parts later.
#-----------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

# Prevent installers (like apt) from asking interactive questions
ENV DEBIAN_FRONTEND=noninteractive

#-----------------------------------------------------------------------
# 1. System Dependencies
#-----------------------------------------------------------------------
# Install basic tools: git, wget, build tools, and a specific C++ compiler
# (g++-11) needed for PyTorch/CUDA compatibility.
RUN apt-get -q update && \
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates gnupg \
        build-essential ninja-build g++-11 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------------------------
# 2. Modern CMake
#-----------------------------------------------------------------------
# Ubuntu's default CMake (3.22) is too old for PyTorch/DGL build.
# Install a recent version manually.
ENV CMAKE_VERSION=3.29.3
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

#-----------------------------------------------------------------------
# 3. CUDA Toolkit 12.8 (Full)
#-----------------------------------------------------------------------
# Install the full CUDA SDK from NVIDIA's network repository.
# This is required to compile DGL with CUDA support.
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get -q update && \
    apt-get -y install cuda-toolkit-12-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Set environment variables for CUDA (runtime and build time)
ENV PATH="/usr/local/cuda-12.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64"
ENV CUDA_HOME="/usr/local/cuda-12.8"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.8"

#-----------------------------------------------------------------------
# 4. Mambaforge (Conda Environment Manager)
#-----------------------------------------------------------------------
# Mamba is a faster alternative to Conda.
ENV CONDA_DIR=/opt/conda
RUN wget --quiet "https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh" -O ~/mambaforge.sh && \
    /bin/bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

#-----------------------------------------------------------------------
# 5. Create Conda Environment
#-----------------------------------------------------------------------
RUN mamba create -n rfdiffusion python=3.11 -y

# Set SHELL to automatically run subsequent commands within the conda env.
SHELL ["conda", "run", "-n", "rfdiffusion", "/bin/bash", "-c"]

#-----------------------------------------------------------------------
# 6. Install PyTorch Nightly (FIRST!)
#-----------------------------------------------------------------------
# Install the PyTorch nightly build for CUDA 12.8.
# This is required for newer GPUs (like RTX 40/50 series, sm_90+).
# Install this first to establish the base PyTorch version.
RUN pip install -U --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

#-----------------------------------------------------------------------
# 7. Compile DGL from Source
#-----------------------------------------------------------------------
# Clone and compile DGL from source to ensure compatibility with
# PyTorch nightly and CUDA 12.8, including the graphbolt component.
WORKDIR /tmp
# --recursive is crucial for submodules like graphbolt
RUN git clone --recursive https://github.com/dmlc/dgl.git && \
    cd dgl && \
    mkdir build && \
    cd build && \
    # Configure the build, enabling CUDA
    cmake -D USE_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 .. && \
    # Compile using a limited number of cores (e.g., 4) to conserve RAM.
    # Increase if you have >32GB RAM allocated to Docker.
    make -j20 && \
    # Install the compiled Python package
    cd ../python && \
    pip install .

#-----------------------------------------------------------------------
# 8. Install Other Python Dependencies
#-----------------------------------------------------------------------
# Install remaining packages. Pip will choose versions of torchdata and e3nn
# compatible with the installed PyTorch nightly build.
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 pyrsistent>=0.19.3 pandas pydantic>=2.0 \
    wandb pynvml torchdata e3nn decorator gitpython
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git

#-----------------------------------------------------------------------
# 9. Build RFdiffusion and Dependencies (SE3Transformer)
#-----------------------------------------------------------------------
# Copy the RFdiffusion source code into the image.
# NOTE: Assumes the RFdiffusion source is in the build context.
WORKDIR /app/RFdiffusion
COPY . .
# Build SE3Transformer first. Its requirements.txt might install older
# dependencies (like e3nn==0.3.3), potentially overwriting newer ones.
RUN cd env/SE3Transformer && \
    pip install -r requirements.txt && \
    python setup.py install && \
    # Build RFdiffusion itself
    cd /app/RFdiffusion && \
    pip install -e .

#-----------------------------------------------------------------------
# 10. Apply e3nn Hotfix (AFTER SE3Transformer install)
#-----------------------------------------------------------------------
# The installation of SE3Transformer (using e3nn==0.3.3) overwrites the
# newer e3nn, so we apply the patch *after* it's been potentially downgraded.
# This fixes the 'weights_only=False' error with newer PyTorch versions.
RUN sed -i "s/torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))/torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'), weights_only=False)/" /opt/conda/envs/rfdiffusion/lib/python3.11/site-packages/e3nn/o3/_wigner.py

# Optional: Verify the fix was applied (uncomment during debugging if needed)
# RUN grep "weights_only=False" /opt/conda/envs/rfdiffusion/lib/python3.11/site-packages/e3nn/o3/_wigner.py || (echo "e3nn patch failed!" && exit 1)

#-----------------------------------------------------------------------
# 11. NumPy Compatibility Fix
#-----------------------------------------------------------------------
# Ensure NumPy version is < 2.0 to prevent ABI conflicts with older compiled packages.
RUN pip install --no-cache-dir "numpy<2"

#-----------------------------------------------------------------------
# STAGE 2: THE FINAL IMAGE
#
# This is the lean image we will actually use. It only contains
# runtime libraries and our application code.
#-----------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

#-----------------------------------------------------------------------
# 1. CUDA Runtime Compatibility
#-----------------------------------------------------------------------
# Install ONLY the CUDA compatibility package. The PyTorch nightly build
# installed via pip is self-contained and brings its own CUDA libs.
# This package acts as a bridge to the host's NVIDIA driver.
RUN apt-get -q update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get -q update && \
    apt-get -y install cuda-compat-12-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

#-----------------------------------------------------------------------
# 2. Copy Environment and Application
#-----------------------------------------------------------------------
# Copy the complete Conda environment (including PyTorch, DGL, etc.)
# from the 'builder' stage.
ENV CONDA_DIR=/opt/conda
COPY --from=builder /opt/conda /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the built RFdiffusion application code from the 'builder' stage.
WORKDIR /app/RFdiffusion
COPY --from=builder /app/RFdiffusion .

#-----------------------------------------------------------------------
# 3. Entrypoint Configuration
#-----------------------------------------------------------------------
# Set final environment variables.
# DO NOT set LD_LIBRARY_PATH, to avoid conflicts with PyTorch's libs.
ENV DGLBACKEND="pytorch"
ENV PYTHONUNBUFFERED=1

# Use the hybrid ENTRYPOINT/CMD approach for flexibility:
# - ENTRYPOINT sets the execution context (our conda environment).
# - CMD sets the default command (the script), which can be easily
#   overridden (e.g., with 'bash' or 'pip install').
ENTRYPOINT ["conda", "run", "-n", "rfdiffusion"]
CMD ["python", "scripts/run_inference.py"]
