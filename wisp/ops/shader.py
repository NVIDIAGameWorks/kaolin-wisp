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
import torch.nn.functional as F
from wisp.core import RenderBuffer, Rays
from wisp.ops.geometric import matcap_sampler, spherical_envmap
from scipy.ndimage import gaussian_filter

# Collection of shaders

def matcap_shader(rb, rays, matcap_path, mm=None):
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

def pointlight_shadow_shader(rb, rays, pipeline, point_light=[1.5,4.5,1.5], min_y=-2.0):
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
    rb.shadow = torch.zeros_like(rb.depth)[:,0].bool().to(rays.origins.device)
    
    with torch.no_grad():
        plane_hit = torch.zeros_like(rb.depth)[:,0].bool().to(rays.origins.device)
        rate = -rays.dirs[:,1] # check negative sign probably lol
        plane_hit[torch.abs(rate) < 0.00001] = False
        delta = rays.origins[:,1] - min_y
        plane_t = delta / rate
        plane_hit[(plane_t > 0) & (plane_t < 500)] = True
        plane_hit = plane_hit & (plane_t < rb.depth[...,0])

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
        rb.shadow = pipeline.tracer(pipeline.nef, shadow_rays).hit

        #rb.shadow[~plane_hit] = 0.0
        rb.shadow[~light_hit] = 0.0
        #rb.hit = rb.hit | plane_hit
        shadow_map = torch.clamp((1.0 - rb.shadow.float()) + 0.7, 0.0, 1.0).cpu().numpy()[...,0]
        shadow_map = torch.from_numpy(gaussian_filter(shadow_map, sigma=2)).unsqueeze(-1)
        rb.rgb[...,:3] *= shadow_map.cuda()
    return rb
