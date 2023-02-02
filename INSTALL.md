# Installing Kaolin Wisp

NVIDIA Kaolin Wisp can be installed either manually or using Docker.

## Manual Installation

### Quick Start
Full installation with interactive visualizer, for torch 1.12.1, cuda 11.3 and kaolin 0.13.0:
```
conda create -n wisp python=3.9
git clone git@github.com:NVIDIAGameWorks/kaolin-wisp.git
cd kaolin-wisp
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
pip install -r requirements.txt
pip install -r requirements_app.txt
python setup.py develop
```
Test the NeRF app with lego data (download separately):
``` 
python app/nerf/main_nerf.py --dataset-path data/lego --config app/nerf/configs/nerf_hash.yaml
```

### Installation Steps in Detail

#### 1. Create an anaconda environment

The easiest way to get started is to create a virtual Python 3.8 Anaconda environment:
```
conda create -n wisp python=3.8
conda activate wisp
pip install --upgrade pip
```

#### 2. (Optional) Install OpenEXR

Some features of our library, like support for the [RTMV dataset](http://www.cs.umd.edu/~mmeshry/projects/rtmv/) 
and logging of multi-layer EXR files (which you can visualize with
awesome tools like [tev](https://github.com/Tom94/tev)) will only work if you install OpenEXR.
These steps are optional, and these features will only be enabled if you follow these steps.

Install OpenEXR on Ubuntu:

```
sudo apt-get update
sudo apt-get install libopenexr-dev 
```

Install OpenEXR on Windows:

```
pip install pipwin
pipwin install openexr
```

#### 3. Install PyTorch

You should first install PyTorch by following the [official instructions](https://pytorch.org/). The code has been tested with `1.9.1` to `1.12.0` on Ubuntu 20.04. 

#### 4. Install Kaolin

kaolin can be installed with pip (use the correct torch + cuda version):
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VER}_cu${CUDA_VER}.html
```

For example, for torch 1.12.1 + cuda 11.3:
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
```

See the [Kaolin Installation Doc](https://kaolin.readthedocs.io/en/latest/notes/installation.html) for additional methods.

_The minimum required version of Kaolin is `0.12.0`._  

#### 5. Installing Wisp

Install the rest of the dependencies from [requirements](requirements.txt).

If you wish to use the interactive renderer and training visualizer, you will need [additional dependencies](requirements_app.txt).
Note that you need to have OpenGL available on your system.

```
pip install -r requirements.txt
pip install -r requirements_app.txt
```

To install wisp, simply execute:
```
python setup.py develop
```
in the main wisp directory. You should now be able to run some examples!

## Using Docker

### 1. Using our Dockerfile (Linux Only)

An easy way to use Wisp is to use our Dockerfile.

You first need to have a base image with [Kaolin Core](https://github.com/NVIDIAGameWorks/kaolin) installed,
we suggested using the [Dockerfile](https://github.com/NVIDIAGameWorks/kaolin/blob/master/tools/linux/Dockerfile.install) of Kaolin Core to build it,
this Dockerfile also takes a Base Image with PyTorch preinstalled, you can either build it with this [Dockerfile](https://github.com/NVIDIAGameWorks/kaolin/blob/master/tools/linux/Dockerfile.base)
or use one available on [dockerhub](https://hub.docker.com/r/pytorch/pytorch) or [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

```
# Clone Kaolin Core
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin/ path/to/kaolin
cd path/to/kaolin

# (Optional) Build the Core pytorch docker image
docker build -f tools/linux/Dockerfile.base -t kaolinbase --network=host \
    --build-arg CUDA_VERSION=11.3.1 \
    --build-arg CUDNN_VERSION=8 \
    --build-arg PYTHON_VERSION=3.9 \
    --build-arg PYTORCH_VERSION=1.11.0 \
    .

# Build the Kaolin Core docker image
# (replace kaolinbase by any image with pytorch preinstalled)
docker build -f tools/linux/Dockerfile.install -t kaolin --network=host \
    --build-arg BASE_IMAGE=kaolinbase \
    .

# Build the Wisp docker image
cd path/to/wisp
docker build -f tools/linux/Dockerfile -t wisp --network=host \
    --build-arg BASE_IMAGE=kaolin \
    --build-arg INSTALL_RENDERER \
    .
```

### 2. Running the Docker container
In order to run the interactive renderer you need to forward the ``DISPLAY`` environment variable
and expose the X server on the host.

```
# expose the X server on the host.
sudo xhost +local:root

# Run the container
docker run --rm -it --gpus=all --net=host --ipc=host -e DISPLAY=$DISPLAY wisp
```

## Testing Your Installation

1. Download some [sample data](https://drive.google.com/file/d/18hY0DpX2bK-q9iY_cog5Q0ZI7YEjephE/view?usp=sharing).
2. Extracted the data somewhere
3. You can train a NeRF using [NGLOD](https://nv-tlabs.github.io/nglod/) with:
```
WISP_HEADLESS=1 python3 app/main_nerf.py -config /app/nerf/configs/nerf_hash.yaml --dataset-path /path/to/lego --dataset-num-workers 4

```

4. To run training with the interactive renderering engine, run:
```
python3 app/main_nerf.py -config /app/nerf/configs/nerf_hash.yaml --dataset-path /path/to/lego --dataset-num-workers 4
```

See `README.md` for an elaborate explanation about running wisp.
