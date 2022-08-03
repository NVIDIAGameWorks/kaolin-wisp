# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import wisp._C as _C

# This is a collection of utility functions that broadly work with different 3D constructs.

def find_depth_bound(query, nug_depth, info, curr_idxes=None):
    r"""Associate query points to the closest depth bound in-order.
    
    TODO: Document the input.
    """
    if curr_idxes is None:
        curr_idxes = torch.nonzero(info).contiguous()
    return _C.render.find_depth_bound_cuda(query.contiguous(), curr_idxes.contiguous(), nug_depth.contiguous())


def sample_unif_sphere(n):
    """Sample uniformly random points on a sphere.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """
    u = np.random.rand(2, n)
    z = 1 - 2*u[0,:]
    r = np.sqrt(1. - z * z)
    phi = 2 * np.pi * u[1,:]
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).transpose()
    return xyz


def sample_fib_sphere(n):
    """
    Evenly distributed points on sphere using Fibonnaci sequence.
    From <http://extremelearning.com.au/evenly-distributing-points-on-a-sphere>
    WARNING: Order is not randomized.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """

    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n)
    golden_ratio = (1 + 5**0.5)/2
    theta = 2. * np.pi * i / golden_ratio
    xyz = np.array([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)]).transpose()
    return xyz


def normalized_grid(height, width, jitter=False, device='cuda', use_aspect=True):
    """Returns grid[x,y] -> coordinates for a normalized window.

    This is generally confusing and terrible, but in the [XYZ] 3D space, the width generally corresponds to
    the XZ axis and the height corresponds to the Y axis. So in the normalized [XY] space, width also
    corresponds to the X and the height corresponds to Y.

    However, PyTorch follows HW ordering. So that means, given a [HW] image, the [H+1, W] coordinate will
    see an increase in the Y axis (the 2nd axis) of the actual global coordinates.

    Args:
        height (int): grid height
        width (int): grid width
        jitter (bool): If True, will jitter the coordinates.
        device (str): Device to allocate the grid on.
        use_aspect (bool): If True, will scale the coords by the aspect ratio.

    Returns:
        (torch.FloatTensor): Coords tensor of shape [H, W, 2]
    """
    window_x = torch.linspace(-1, 1, steps=width, device=device)
    window_y = torch.linspace(1, -1, steps=height, device=device)

    if jitter:
        window_x += (2.0 * torch.rand(*window_x.shape, device=device) - 1.0) * (1. / width)
        window_y += (2.0 * torch.rand(*window_y.shape, device=device) - 1.0) * (1. / height)

    if use_aspect:
        if width > height:
            window_x = window_x * (width / height)
        elif height > width:
            window_y = window_y * (height / width)

    coord = torch.stack(torch.meshgrid(window_x, window_y)).permute(2, 1, 0)  # torch>=1.10: indexing='ij' (default)
    return coord


def normalized_slice(height, width, dim=0, depth=0.0, device='cuda'):
    """Returns a set of 3D coordinates for a slicing plane.

    Args:
        height (int): Grid height.
        width (int): Grid width.
        dim (int): Dimension to slice along.
        depth (float): The depth (from the 0 on the axis) for which the slicing will happen.
        device (str): Device to allocate the grid on.

    Returns:
        (torch.FloatTensor): Coords tensor of shape [height, width, 3].
    """
    window = normalized_grid(height, width, device)
    depth_pts = torch.ones(height, width, 1, device=device) * depth

    if dim == 0:
        pts = torch.cat([depth_pts, window[..., 0:1], window[..., 1:2]], dim=-1)
    elif dim == 1:
        pts = torch.cat([window[..., 0:1], depth_pts, window[..., 1:2]], dim=-1)
    elif dim == 2:
        pts = torch.cat([window[..., 0:1], window[..., 1:2], depth_pts], dim=-1)
    else:
        assert (False, "dim is invalid!")
    pts[..., 1] *= -1
    return pts


def spherical_envmap(ray_dir, normal):
    """Computes matcap UV-coordinates from the ray direction and normal.

    Args:
        ray_dir (torch.Tensor): incoming ray direction of shape [...., 3]
        normal (torch.Tensor): surface normal of shape [..., 3]

    Returns:
        (torch.FloatTensor): UV coordinates of shape [..., 2]
    """
    # Input should be size [...,3]
    # Returns [N,2] # Might want to make this [...,2]

    # TODO(ttakikawa): Probably should implement all this on GPU
    ray_dir_screen = ray_dir.clone()
    ray_dir_screen[..., 2] *= -1
    ray_dir_normal_dot = torch.sum(normal * ray_dir_screen, dim=-1, keepdim=True)
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    r[..., 2] -= 1.0
    m = 2.0 * torch.sqrt(torch.sum(r ** 2, dim=-1, keepdim=True))
    vN = (r[..., :2] / m) + 0.5
    vN = 1.0 - vN
    vN = vN[..., :2].reshape(-1, 2)
    vN = torch.clip(vN, 0.0, 1.0)
    vN[torch.isnan(vN)] = 0
    return vN


def spherical_envmap_numpy(ray_dir, normal):
    """Computes matcap UV-coordinates from the ray direction and normal.

    Args:
        ray_dir (torch.Tensor): incoming ray direction of shape [...., 3]
        normal (torch.Tensor): surface normal of shape [..., 3]

    Returns:
        (torch.FloatTensor): UV coordinates of shape [..., 2]
    """
    ray_dir_screen = ray_dir * np.array([1, 1, -1])
    # Calculate reflection
    ray_dir_normal_dot = np.sum(normal * ray_dir_screen, axis=-1)[..., np.newaxis]
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    m = 2.0 * np.sqrt(r[..., 0] ** 2 + r[..., 1] ** 2 + (r[..., 2] - 1) ** 2)
    vN = (r[..., :2] / m[..., np.newaxis]) + 0.5
    vN = 1.0 - vN
    vN = vN[..., :2].reshape(-1, 2)
    vN = np.clip(vN, 0, 1)
    vN[np.isnan(vN)] = 0
    return vN
