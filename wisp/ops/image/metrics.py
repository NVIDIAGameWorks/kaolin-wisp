# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import skimage
import skimage.metrics
import numpy as np
import torch
from lpips import LPIPS


""" A module for image based metrics """


def psnr(rgb, gts):
    """Calculate the PSNR metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The PSNR score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse = torch.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
    return 10 * np.log10(1.0 / mse)


def lpips(rgb, gts, lpips_model=None):
    """Calculate the LPIPS metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The LPIPS score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    if lpips_model is None:
        lpips_model = LPIPS(net='vgg').cuda()

    return lpips_model(
        (2.0 * rgb[..., :3] - 1.0).permute(2, 0, 1),
        (2.0 * gts[..., :3] - 1.0).permute(2, 0, 1)).mean().item()


def ssim(rgb, gts):
    """Calculate the SSIM metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The SSIM score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return skimage.metrics.structural_similarity(
        rgb[..., :3].cpu().numpy(),
        gts[..., :3].cpu().numpy(),
        multichannel=True,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5)
