# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from wisp.core import RenderBuffer, Rays

""" A collection of shader functions for shadow rays """


def pointlight_shadow_shader(rb: RenderBuffer, rays: Rays, pipeline,
                             point_light=[1.5, 4.5, 1.5], min_y=-2.0) -> RenderBuffer:
    """Apply shadow rays with one secondary ray towards the pointlight.

    Args:
        rb (wisp.core.RenderBuffer): The RenderBuffer.
        rays (wisp.core.Rays): The rays object.
        pipeline (wisp.core.Pipeline): The neural field.
        point_light (list[3] of float): Position of the point light.
        min_y (float): The location of the xz plane.

    Returns:
        (wisp.core.RenderBuffer): The output RenderBuffer.
    """
    rb.shadow = torch.zeros_like(rb.depth)[:, 0].bool().to(rays.origins.device)

    with torch.no_grad():
        plane_hit = torch.zeros_like(rb.depth)[:, 0].bool().to(rays.origins.device)
        rate = -rays.dirs[:, 1]  # check negative sign probably lol
        plane_hit[torch.abs(rate) < 0.00001] = False
        delta = rays.origins[:, 1] - min_y
        plane_t = delta / rate
        plane_hit[(plane_t > 0) & (plane_t < 500)] = True
        plane_hit = plane_hit & (plane_t < rb.depth[..., 0])

        rb.hit = rb.hit & ~plane_hit

        rb.depth[plane_hit] = plane_t[plane_hit].unsqueeze(1)
        rb.xyz[plane_hit] = rays.origins[plane_hit] + rays.dirs[plane_hit] * plane_t[plane_hit].unsqueeze(1)
        rb.normal[plane_hit] = 0
        rb.normal[plane_hit, 1] = 1

        # x is shadow ray origin
        light_o = torch.FloatTensor([[point_light]]).to(rays.origins.device)

        shadow_ray_o = rb.xyz + 0.01 * rb.normal
        shadow_ray_d = torch.zeros_like(rb.xyz).normal_(0.0, 0.01) + \
                       light_o - shadow_ray_o
        shadow_ray_d = F.normalize(shadow_ray_d, dim=1)[0]

        light_hit = ((shadow_ray_d * rb.normal).sum(-1) > 0.0)

        shadow_rays = Rays(origins=shadow_ray_o, dirs=shadow_ray_d, dist_min=0, dist_max=rays.dist_max)
        rb.shadow = pipeline.tracer(pipeline.nef, rays=shadow_rays).hit

        # rb.shadow[~plane_hit] = 0.0
        rb.shadow[~light_hit] = 0.0
        # rb.hit = rb.hit | plane_hit
        shadow_map = torch.clamp((1.0 - rb.shadow.float()) + 0.7, 0.0, 1.0).cpu().numpy()[..., 0]
        shadow_map = torch.from_numpy(gaussian_filter(shadow_map, sigma=2)).unsqueeze(-1)
        rb.rgb[..., :3] *= shadow_map.cuda()
    return rb