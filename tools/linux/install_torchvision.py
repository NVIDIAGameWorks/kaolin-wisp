import os
import torch

torch_to_torchvision = {
    "1.11.0": "0.12.0",
    "1.12.0": "0.13.0",
    "1.12.1": "0.13.1",
    "1.13.0": "0.14.0",
    "1.13.1": "0.14.1",
    "2.0.0" : "0.15.0",
}

supported_cuda_versions = ["11.3", "11.4", "11.5", "11.6", "11.7", "11.8"]

torch_version = torch.__version__.split("+")[0]

if not torch_version in torch_to_torchvision:
    raise Exception(f"torch version {torch_version} not supported.")
else:
    if torch.version.cuda not in supported_cuda_versions:
        raise Exception(f"CUDA version {torch.version.cuda} not supported")
    major, minor = torch.version.cuda.split(".")
    cuda_string = "cu" + major + minor
    torchvision_version = torch_to_torchvision[torch_version]
    os.system(f"pip install torchvision=={torchvision_version}+{cuda_string} --index-url https://download.pytorch.org/whl/{cuda_string}")
