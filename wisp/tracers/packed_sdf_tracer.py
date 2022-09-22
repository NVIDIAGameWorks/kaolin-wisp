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
    """Tracer class for sparse SDFs.

    This tracer class expects the use of a feature grid that has a BLAS (i.e. inherits the BLASGrid
    class).
    """

    def __init__(self, num_steps=64, step_size=1.0, min_dis=1e-4, **kwargs):
        """Set the default trace() arguments. """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.step_size = step_size
        self.min_dis = min_dis
    
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "normal", "xyz", "hit", "rgb", "alpha"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"sdf"}

    def trace(self, nef, channels, extra_channels, rays, lod_idx=None, num_steps=64, step_size=1.0, min_dis=1e-4):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  query those extra channels at surface intersection points.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            num_steps (int): The number of steps to use for sphere tracing.
            step_size (float): The multiplier for the sphere tracing steps. 
                               Use a value <1.0 for conservative tracing.
            min_dis (float): The termination distance for sphere tracing.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        #TODO(ttakikawa): Use a more robust method
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
            dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels="sdf") * invres * step_size
            dist[~mask] = 20
            dist_prev = dist.clone()
            timer.check("first")

            for i in range(num_steps):
                # Two-stage Ray Marching
                
                # Step 1: Use SDF to march
                t += dist
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                hit = torch.where(mask, torch.abs(dist)[...,0] < min_dis * invres, hit)
                hit |= torch.where(mask, 
                                   torch.abs(dist+dist_prev)[...,0] * 0.5 < (min_dis*5) * invres, hit)
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
                dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels="sdf") * invres * step_size
            timer.check("step done")
    
        x_buffer = torch.zeros_like(rays.origins)
        depth_buffer = torch.zeros_like(rays.origins[...,0:1])
        hit_buffer = torch.zeros_like(rays.origins[...,0]).bool()
        normal_buffer = torch.zeros_like(rays.origins)
        rgb_buffer = torch.zeros(*rays.origins.shape[:-1], 3, device=rays.origins.device)
        alpha_buffer = torch.zeros(*rays.origins.shape[:-1], 1, device=rays.origins.device)
        hit_buffer[first_ridx] = hit
        
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=x[hit], lod_idx=lod_idx, channels=channel)
            extra_buffer = torch.zeros(*rays.origins.shape[:-1], feats.shape[-1], device=feats.device)
            extra_buffer[hit_buffer] = feats

        x_buffer[hit_buffer] = x[hit]
        depth_buffer[hit_buffer] = t[hit]
        
        # TODO(ttakikawa): RGB channel should _not_ be the normals. This should be only the case
        # if some shader object defines this to be the case.
        if "rgb" in channels or "normal" in channels:
            grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
            normal_buffer[hit_buffer] = F.normalize(grad, p=2, dim=-1, eps=1e-5)
            rgb_buffer[..., :3] = (normal_buffer + 1.0) / 2.0
        
        alpha_buffer[hit_buffer] = 1.0
        timer.check("populate buffers")
        return RenderBuffer(xyz=x_buffer, depth=depth_buffer, hit=hit_buffer, normal=normal_buffer,
                            rgb=rgb_buffer, alpha=alpha_buffer, **extra_outputs)
