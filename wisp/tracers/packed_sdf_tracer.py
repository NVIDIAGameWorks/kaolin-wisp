# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.differential import finitediff_gradient
from wisp.ops.geometric import find_depth_bound
from wisp.tracers import BaseTracer

class PackedSDFTracer(BaseTracer):

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
        assert "sdf" in supported_channels and "this tracer requires sdf channels"
        assert nef.grid is not None and "this tracer requires a grid"
        
        if lod_idx is None:
            lod_idx = nef.num_lods-1

        timer = PerfTimer(activate=False)

        res = float(2**(lod_idx+nef.base_lod))
        #invres = 1.0 / res
        invres = 1.0

        # Trace SPC
        ridx, pidx, depth = nef.grid.raytrace(rays, nef.grid.active_lods[lod_idx], with_exit=True)
        depth[...,0:1] += 1e-5

        first_hit = spc_render.mark_pack_boundaries(ridx)
        curr_idxes = torch.nonzero(first_hit)[...,0].int()

        first_ridx = ridx[first_hit].long()
        nug_o = rays.origins[first_ridx]
        nug_d = rays.dirs[first_ridx]

        mask = torch.ones([first_ridx.shape[0]], device=nug_o.device).bool()
        hit = torch.zeros_like(mask).bool()

        t = depth[first_hit][...,0:1]
        x = torch.addcmul(nug_o, nug_d, t)
        dist = torch.zeros_like(t)
        
        curr_pidx = pidx[first_hit].long()
        
        timer.check("initial")
        # Doing things with where is not super efficient, but we have to make do with what we have...
        with torch.no_grad():

            # Calculate SDF for current set of query points   
            dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels="sdf") * invres * self.step_size
            dist[~mask] = 20
            dist_prev = dist.clone()
            timer.check("first")

            for i in range(self.num_steps):
                # Two-stage Ray Marching
                
                # Step 1: Use SDF to march
                t += dist
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                hit = torch.where(mask, torch.abs(dist)[...,0] < self.min_dis * invres, hit)
                hit |= torch.where(mask, 
                                   torch.abs(dist+dist_prev)[...,0] * 0.5 < (self.min_dis*5) * invres, hit)
                mask = torch.where(mask, (t < rays.dist_max)[...,0], mask)
                mask &= ~hit
                if not mask.any():
                    break
                dist_prev = torch.where(mask.view(mask.shape[0], 1), dist, dist_prev)
                
                # Step 2: Use AABBs to march
                next_idxes = find_depth_bound(t, depth, first_hit, curr_idxes=curr_idxes)
                mask &= (next_idxes != -1)
                aabb_mask = (next_idxes != curr_idxes)
                curr_idxes = torch.where(mask, next_idxes, curr_idxes)

                t = torch.where((mask & aabb_mask).view(mask.shape[0], 1), depth[curr_idxes.long(), 0:1], t)
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                
                curr_pidx = torch.where(mask, pidx[curr_idxes.long()].long(), curr_pidx)
                if not mask.any():
                    break
                dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels="sdf") * invres * self.step_size
            timer.check("step done")
    
        x_buffer = torch.zeros_like(rays.origins)
        depth_buffer = torch.zeros_like(rays.origins[...,0:1])
        hit_buffer = torch.zeros_like(rays.origins[...,0]).bool()
        normal_buffer = torch.zeros_like(rays.origins)
        rgb_buffer = torch.zeros(*rays.origins.shape[:-1], 3, device=rays.origins.device)
        alpha_buffer = torch.zeros(*rays.origins.shape[:-1], 1, device=rays.origins.device)

        hit_buffer[first_ridx] = hit
        x_buffer[hit_buffer] = x[hit]
        depth_buffer[hit_buffer] = t[hit]
        
        grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
        normal_buffer[hit_buffer] = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        rgb_buffer[..., :3] = (normal_buffer + 1.0) / 2.0
        alpha_buffer[hit_buffer] = 1.0
        timer.check("populate buffers")
        return RenderBuffer(xyz=x_buffer, depth=depth_buffer, hit=hit_buffer, normal=normal_buffer, rgb=rgb_buffer,
                            alpha=alpha_buffer)
