# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

PACKAGE_NAME = 'wisp'
DESCRIPTION = 'Kaolin-Wisp: A PyTorch library for performing research on neural fields'
URL = 'https://github.com/NVIDIAGameWorks/kaolin-wisp'
AUTHOR = 'Towaki Takikawa'
LICENSE = 'NVIDIA Source Code License'
version = '0.1.0'

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

if not torch.cuda.is_available():
    if os.getenv('FORCE_CUDA', '0') == '1':
        # From: https://github.com/NVIDIA/apex/blob/c4e85f7bf144cb0e368da96d339a6cbd9882cea5/setup.py
        # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
        # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "If your intention is to cross-compile, this is not an error.\n"
            "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
            "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
            "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
            "If you wish to cross-compile for a single specific architecture,\n"
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        )
        if os.getenv("TORCH_CUDA_ARCH_LIST", None) is None:
            _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
            if int(bare_metal_major) == 11:
                if int(bare_metal_minor) == 0:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
    else:
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "This script will install only with CPU support and will have very limited features.\n"
            'If your wish to cross-compile for GPU `export FORCE_CUDA=1` before running setup.py\n'
            "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
            "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
            "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
            "If you wish to cross-compile for a single specific architecture,\n"
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        )

def get_extensions():
    extra_compile_args = {'cxx': ['-O3']} 
    define_macros = []
    include_dirs = []
    extensions = []
    sources = glob.glob('wisp/csrc/**/*.cpp', recursive=True)
 
    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None


    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('wisp/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': ['-O3']})
        #include_dirs = get_include_dirs()
    else:
        assert(False, "CUDA is not available. Set FORCE_CUDA=1 for Docker builds")

    extensions.append(
        extension(
            name='wisp._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            #include_dirs=include_dirs
        )
    )

    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x
                         for x in ext.libraries]
 
    return extensions

if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='>=3.8',

        # Package info
        packages=['wisp'] + find_packages(),
        #package_dir={'':'wisp'},
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)    
        }

    )
