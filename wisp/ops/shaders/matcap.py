# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from wisp.core import RenderBuffer, Rays
from wisp.ops.geometric import spherical_envmap

""" A collection of shader functions for matcaps """


def matcap_shader(rb: RenderBuffer, rays: Rays, matcap_path, mm=None) -> RenderBuffer:
    """Apply matcap shading.

    Args:
        rb (wisp.core.RenderBuffer): The RenderBuffer.
        rays (wisp.core.Rays): The rays object.
        matcap_path (str): Path to a matcap.
        mm (torch.FloatTensor): A 3x3 rotation matrix.

    Returns:
        (wisp.core.RenderBuffer): The output RenderBuffer.
    """
    if not os.path.exists(matcap_path):
        raise Exception(f"The path [{matcap_path}] does not exist. Check your working directory or use an absolute path to the matcap with --matcap-path")


    # TODO: Write a GPU version of the sampler...
    matcap = matcap_sampler(matcap_path)
    matcap_normal = rb.normal.clone()
    matcap_view = rays.dirs.clone()
    if mm is not None:
        mm = mm.to(matcap_normal.device)
        #matcap_normal = torch.mm(matcap_normal.reshape(-1, 3), mm.transpose(1,0))
        #matcap_normal = matcap_normal.reshape(self.width, self.height, 3)
        shape = matcap_view.shape
        matcap_view = torch.mm(matcap_view.reshape(-1, 3), mm.transpose(1,0))
        matcap_view = matcap_view.reshape(*shape)
    vN = spherical_envmap(matcap_view, matcap_normal).cpu().numpy()
    rb.rgb = torch.FloatTensor(matcap(vN)[...,:3].reshape(*matcap_view.shape)).to(matcap_normal.device) / 255.0
    return rb


def matcap_sampler(path, interpolate=True):
    """Fetches MatCap texture & converts to a interpolation function (if needed).

    TODO(ttakikawa): Replace this with something GPU compatible.

    Args:
        path (str): path to MatCap texture
        interpolate (bool): perform interpolation (default: True)

    Returns:
        (np.array) or (scipy.interpolate.Interpolator)
        - The matcap texture
        - A SciPy interpolator function to be used for CPU texture fetch.
    """

    matcap = np.array(Image.open(path)).transpose(1, 0, 2)
    if interpolate:
        return RegularGridInterpolator((np.linspace(0, 1, matcap.shape[0]),
                                        np.linspace(0, 1, matcap.shape[1])), matcap)
    else:
        return matcap
