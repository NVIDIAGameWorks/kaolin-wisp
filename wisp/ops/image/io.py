# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import glob
import pyexr
import cv2
import skimage
import imageio
from PIL import Image
import numpy as np
import torch

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
    Image.fromarray(data).save(path)


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


def load_rgb(path):
    """Loads an image.

    TODO(ttakikawa): Currently ignores the alpha channel.

    Args:
        path (str): Path to the image.

    Returns:
        (np.array): Image as an array.
    """
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    img = img[:, :, :3]
    return img


def load_mask(path):
    """Loads an alpha mask.

    Args:
        path (str): Path to the image.

    Returns:
        (np.array): Image as an array.
    """
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5
    object_mask = object_mask.transpose(1, 0)

    return object_mask


def load_exr(path, use_depth=False, mip=None, srgb=False, bg_color='white',
             loader_type='pyexr'):
    """Loads a EXR by path.

    Args:
        path (str): path to the .exr file
        use_depth (bool): if True, loads the depth data
                          by default, this assumes the depth is stored in the "depth" buffer
        mip (int): if not None, then each image will be resized by 2^mip
        srgb (bool): if True, convert to SRGB
        loader_type (str): options [cv2, pyexr, imageio].
                           TODO(ttakikawa): Not sure quite yet what options should be supported here

    Returns:
        (dictionary of torch.Tensors)

        Keys:
            image : torch.FloatTensor of size [H,W,3]
            alpha : torch.FloatTensor of size [H,W,1]
            depth : torch.FloatTensor of size [H,W,1]
            ray_o : torch.FloatTensor of size [H,W,3]
            ray_d : torch.FloatTensor of size [H,W,3]
    """
    # TODO(ttakikawa): There is a lot that this function does... break this up
    from wisp.ops.image import resize_mip, linear_to_srgb

    # Load RGB and Depth
    if loader_type == 'cv2':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if use_depth:
            depth = cv2.imread(path.replace(".exr", ".depth.exr"), cv2.IMREAD_UNCHANGED)[:, :, 0]

    elif loader_type == 'pyexr':
        f = pyexr.open(path)
        img = f.get("default")

        if use_depth:
            if len(f.channel_map['depth']) > 0:
                depth = f.get("depth")
            else:
                f = pyexr.open(path.replace(".exr", ".depth.exr"))
                depth = f.get('default')[:, :, 0]

    elif loader_type == 'imageio':
        img = imageio.imread(path)
        if use_depth:
            depth = imageio.imread(path.replace(".exr", ".depth.exr"))[:, :, 0]
    else:
        raise ValueError(f'Invalid loader_type: {loader_type}')

    alpha = img[..., 3:4]

    if bg_color == 'black':
        img[..., :3] -= (1 - alpha)
        img = np.clip(img, 0.0, 1.0)
    else:
        img[..., :3] *= alpha
        img[..., :3] += (1 - alpha)
        img = np.clip(img, 0.0, 1.0)

    if mip is not None:
        # TODO(ttakikawa): resize_mip causes the mask to be squuezed... why?
        img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        if use_depth:
            depth = resize_mip(depth, mip, interpolation=cv2.INTER_NEAREST)
            # mask_depth = resize_mip(mask_depth[...,None].astype(np.float), mip, interpolation=cv2.INTER_NEAREST)

    img = torch.from_numpy(img)
    if use_depth:
        depth = torch.from_numpy(depth)

    if use_depth:
        mask_depth = torch.logical_and(depth > -1000, depth < 1000)
        depth[~mask_depth] = -1.0
        depth = depth[:, :, np.newaxis]

    if loader_type == 'cv2' or loader_type == 'imageio':
        # BGR to RGB
        img[..., :3] = img[..., :3][..., ::-1]

    if srgb:
        img = linear_to_srgb(img)

    alpha = mask_depth

    return img, alpha, depth


def hwc_to_chw(img):
    """Converts [H,W,C] to [C,H,W] for TensorBoard output.

    Args:
        img (np.array): [H,W,C] image.

    Returns:
        (np.array): [C,H,W] image.
    """
    return np.array(img).transpose(2, 0, 1)
