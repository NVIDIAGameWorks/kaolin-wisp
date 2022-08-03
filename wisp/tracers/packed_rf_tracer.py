# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.utils import PsDebugger, PerfTimer
from wisp.tracers import BaseTracer


class PackedRFTracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.

    This tracer class expects the use of a feature grid that has a BLAS (i.e. inherits the BLASGrid
    class). It also expects a `nef.rgba` method to be implemented. 
    """

    def forward(self, nef, rays, lod_idx=None):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        supported_channels = nef.get_supported_channels()
        assert "rgb" in supported_channels and "this tracer requires rgb channels"
        assert "density" in supported_channels and "this tracer requires density channels"
        assert nef.grid is not None and "this tracer requires a grid"

        timer = PerfTimer(activate=False, show_memory=False)
        N = rays.origins.shape[0]
        

        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        # By default, PackedRFTracer will attempt to use the highest level of detail for the ray sampling.
        # This however may not actually do anything; the ray sampling behaviours are often single-LOD
        # and is governed by however the underlying feature grid class uses the BLAS to implement the sampling.
        ridx, pidx, samples, depths, deltas, boundary = nef.grid.raymarch(rays, 
                level=nef.grid.active_lods[lod_idx], num_samples=self.num_steps, raymarch_type=self.raymarch_type)

        timer.check("Raymarch")

        # Check for the base case where the BLAS traversal hits nothing
        if ridx.shape[0] == 0:
            if self.bg_color == 'white':
                hit = torch.zeros(N, device=ridx.device).bool()
                rgb = torch.ones(N, 3, device=ridx.device)
                alpha = torch.zeros(N, 1, device=ridx.device)
            else:
                hit = torch.zeros(N, 1, device=ridx.device).bool()
                rgb = torch.zeros(N, 3, device=ridx.device)
                alpha = torch.zeros(N, 1, device=ridx.device)
            
            depth = torch.zeros(N, 1, device=ridx.device)
            
            return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=alpha)
        
        timer.check("Boundary")
        
        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        
        # Compute the color and density for each ray and their samples
        color, density = nef(coords=samples, ray_d=rays.dirs.index_select(0, ridx), pidx=pidx, lod_idx=lod_idx,
                             channels=["rgb", "density"])

        timer.check("RGBA")        
        del ridx, pidx, rays

        # Compute optical thickness
        tau = density.reshape(-1, 1) * deltas
        del density, deltas

        # Perform volumetric integration
        ray_colors, transmittance = spc_render.exponential_integration(color.reshape(-1, 3), tau, boundary, exclusive=True)
        ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
        depth = torch.zeros(N, 1, device=ray_depth.device)
        depth[ridx_hit.long(), :] = ray_depth
        timer.check("Integration")
        alpha = spc_render.sum_reduce(transmittance, boundary)
        timer.check("Sum Reduce")
        
        hit = torch.zeros(N, device=color.device).bool()

        # Populate the background
        if self.bg_color == 'white':
            rgb = torch.ones(N, 3, device=color.device)
            out_alpha = torch.zeros(N, 1, device=color.device)
            bg = torch.ones([ray_colors.shape[0], 3], device=ray_colors.device)
            color = (1.0-alpha) * bg + alpha * ray_colors
        else:
            rgb = torch.zeros(N, 3, device=color.device)
            out_alpha = torch.zeros(N, 1, device=color.device)
            color = alpha * ray_colors
        
        hit[ridx_hit.long()] = alpha[...,0] > 0.0

        rgb[ridx_hit.long(), :3] = color
        out_alpha[ridx_hit.long()] = alpha
        
        timer.check("Composit")

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha)
