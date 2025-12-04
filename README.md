# RFdiffusion Docker for CUDA 12.8 / RTX 5090 GPUs

## 

It was quite a journey getting this Dockerfile working correctly,(I learned a lot in the process).
If this Dockerfile helps you, I'd be grateful for feedback through issues or pull requests so I know it's useful to the community.

##

Updates 2025-11-18 the docker is avalible here

https://hub.docker.com/r/jmbscripts/rfdiffusion-nvidia-rtx5090

## Purpose

This repository provides a `Dockerfile` to build a working environment for [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) specifically tailored for:

* **CUDA 12.8**
* **Modern NVIDIA GPUs** (e.g., RTX 5090) requiring compute capability `sm_1200` or higher.
* Resolving complex dependency conflicts between **PyTorch nightly builds**, **DGL**, **e3nn**, and **torchdata**.

The standard RFdiffusion installation often fails on newer hardware due to these rapidly evolving dependencies. This Dockerfile incorporates several fixes and workarounds discovered through extensive debugging.

## Key Features & Fixes

* Uses **Ubuntu 22.04** as the base image.
* Installs **CUDA Toolkit 12.8** from NVIDIA's official network repository in the builder stage.
* Installs the **PyTorch nightly build** compiled for CUDA 12.8, ensuring compatibility with the latest GPU architectures.
* **Compiles DGL from source** to guarantee compatibility with the PyTorch nightly build and CUDA 12.8, including necessary components like GraphBolt.
* Installs compatible versions of other dependencies like `e3nn` and `torchdata`.
* Includes a **hotfix for `e3nn`** to prevent `torch.load` errors related to the `weights_only` parameter change in newer PyTorch versions.
* Forces **NumPy < 2.0** to avoid ABI compatibility issues.
* Uses a **multi-stage build** to keep the final image relatively lean, excluding build tools.
* Configures the final image to avoid CUDA library conflicts (`LD_LIBRARY_PATH` is unset) and installs only the minimal `cuda-compat` package.
* Uses a flexible `ENTRYPOINT`/`CMD` setup, allowing easy execution of the default script or other commands (like `bash` or `pip`).

## Prerequisites (Host Machine)

1.  **Docker:** Docker Engine installed (not Docker Desktop as it never recognize my GPU).
2.  **NVIDIA GPU:** A CUDA-enabled NVIDIA GPU.
3.  **NVIDIA Drivers:** Recent drivers supporting CUDA 12.8 or higher.
4.  **NVIDIA Container Toolkit:** Properly installed and configured to allow Docker access to the GPU.
    * **Test:** Before building the image, verify that your Docker installation can access the GPU by running the following command. It should successfully run `nvidia-smi` inside a container and display your GPU information. If this fails, the RFdiffusion image won't work either.
        ```bash
        docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
        ```

## Build Instructions

1.  **Place RFdiffusion Source Code:**
    * Clone or download the official RFdiffusion source code.
    * Place the entire `RFdiffusion` source code directory inside *this* directory (the build context), alongside the `Dockerfile`. Your directory structure should look like:
        ```
        your-repo-directory/
        └── RFdiffusion/  <-- Official RFdiffusion source code here
            ├── env/
            ├── docker/ <--- put the dockerfile here
            ├── scripts/
            └── ...
        ```

2.  **Build the Docker Image:**
    ```bash
    docker build -f docker/RTX-5090.dockerfile -t myrfd .
    ```
    * **Note:** This build process will take a **long time** (potentially over an hour), especially the DGL compilation step.
    * **Memory:** Ensure Docker has sufficient memory allocated (at least 16GB recommended, 32GB+ is safer for DGL compilation). The `make -j20` command in the Dockerfile limits parallelism to conserve memory; you can increase the number (e.g., `make -j40`) if you have more RAM allocated to Docker.

## Run Instructions

1.  **Prepare Directories:** Create directories on your host machine for inputs, outputs, and RFdiffusion models.
    ```bash
    mkdir -p inputs outputs models
    ```
    * Place your input PDB file(s) (e.g., `target.pdb`) in the `inputs` directory.
    * Place the pre-trained RFdiffusion model checkpoint files (`.pt`) in the `models` directory. The `outputs` directory will store the results.

2.  **Run RFdiffusion Inference:**
    Use the `docker run` command, mounting your directories as volumes. Adjust the script arguments as needed.
    ```bash
    # Example for binder design
    docker run --gpus all -v $(pwd)/inputs:/inputs -v $(pwd)/outputs:/outputs -v $(pwd)/models:/models myrfd \
    python scripts/run_inference.py \
    inference.output_prefix=/outputs/test \
    inference.model_directory_path=/models \
    inference.input_pdb=/inputs/5TPN.pdb \
    inference.num_designs=3 \
    'contigmap.contigs=[10-40/A163-181/10-40]'50
    ```
    * `--gpus all`: Makes all host GPUs visible (RFdiffusion script typically uses only one).
    * `-v ...`: Maps your local directories to the expected paths inside the container.
    * `myrfd`: The name you gave the image during the build.
    * `python scripts/run_inference.py ...`: The command to run inside the container, followed by RFdiffusion arguments.
    * `contigmap.contigs`: **Crucial argument.** Define your design goal here. Use single quotes around the entire argument.

3.  **Accessing a Shell:** To debug or run other commands inside the container:
   useful to see if there is something wrong in the final docker

    ```bash
    docker run --gpus all --rm -it \
        -v "$(pwd)/inputs":/inputs \
        -v "$(pwd)/outputs":/outputs \
        -v "$(pwd)/models":/models \
        myrfd bash
    ```
    You will get a bash prompt already inside the `rfdiffusion` conda environment.
    ```bash
   source /opt/conda/bin/activate    
   conda activate rfdiffusion
    ```

## Thanks:
 
  to Gemini (I knew nothing about docker worlds, cuda)
  
  to DTZhou1996 for all the works around.

## License

Consider adding a `LICENSE` file (e.g., MIT, Apache 2.0) to clarify how others can use your work.
