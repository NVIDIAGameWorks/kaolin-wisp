# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import glob
import numpy as np
import torch
import torchvision

""" A module for reading / writing various image formats. """


def write_exr(path, data):
    """Writes an EXR image to some path.

    Data is a dict of form { "default" = rgb_array, "depth" = depth_array }

    Args:
        path (str): Path to save the EXR
        data (dict): Dictionary of EXR buffers.

    Returns:
        (void): Writes to path.
    """
    try:
        import pyexr
    except:
        raise Exception(
            "Module pyexr is not available. To install, run `pip install pyexr`. "
            "You will likely also need `libopenexr`, which through apt you can install with "
            "`apt-get install libopenexr-dev` and on Windows you can install with "
            "`pipwin install openexr`.")
    pyexr.write(path, data,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)

def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    torchvision.io.write_png(hwc_to_chw(data), path)

def glob_imgs(path, exts=['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']):
    """Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path, normalize=True):
    """Loads an image.

    Args:
        path (str): Path to the image.
        noramlize (bool): If True, will return [0,1] floating point values. Otherwise returns [0,255] ints.

    Returns:
        (np.array): Image as an array of shape [H,W,C]
    """
    img = torchvision.io.read_image(path)
    if normalize:
        img = img.float() / 255.0
    return np.array(chw_to_hwc(img))

def hwc_to_chw(img):
    """Converts [H,W,C] to [C,H,W] for TensorBoard output.

    Args:
        img (torch.Tensor): [H,W,C] image.

    Returns:
        (torch.Tensor): [C,H,W] image.
    """
    return img.permute(2, 0, 1)

def chw_to_hwc(img):
    """Converts [C,H,W] to [H,W,C].

    Args:
        img (torch.Tensor): [C,H,W] image.

    Returns:
        (torch.Tensor): [H,W,C] image.
    """
    return img.permute(1, 2, 0)
