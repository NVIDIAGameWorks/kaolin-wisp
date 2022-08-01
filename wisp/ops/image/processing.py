# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import cv2
import torch


def srgb_to_linear(img):
    """Converts from SRGB to Linear colorspace.

    Args:
        img (torch.FloatTensor): SRGB image.

    Returns:
        (torch.FloatTensor): Linear image.
    """
    limit = 0.04045
    return torch.where(img > limit, torch.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    """Converts from Linear to SRGB colorspace.

    Args:
        img (torch.FloatTensor): Linear image.

    Returns:
        (torch.FloatTensor): SRGB image.
    """
    limit = 0.0031308
    img = torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    img[img > 1] = 1
    return img


def resize_mip(img, mip, interpolation=cv2.INTER_LINEAR):
    """Resize image with cv2.

    Args:
        img (torch.FloatTensor): Image of shape [H, W, 3]
        mip (int): Rescaling factor. Will rescale by 2**mip.
        interpolation: Interpolation modes used by `cv2.resize`.

    Returns:
        (torch.FloatTensor): Rescaled image of shape [H/(2**mip), W/(2**mip), 3]
    """
    resize_factor = 2**mip
    # WARNING: cv2 expects (w,h) for the shape. God knows why :)
    shape = (int(img.shape[1] // resize_factor), int(img.shape[0] // resize_factor))
    img = cv2.resize(img, dsize=shape, interpolation=interpolation)
    return img
