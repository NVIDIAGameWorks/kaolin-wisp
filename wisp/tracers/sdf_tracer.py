# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
from wisp.core import RenderBuffer
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.differential import finitediff_gradient
from wisp.tracers import BaseTracer

class SDFTracer(BaseTracer):

    def __init__(self,  num_steps=64, step_size=1.0, min_dis=1e-4, raymarch_type='voxel', **kwargs):
        """Set the default trace() arguments. """
        super().__init__(**kwargs)
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.min_dis = min_dis
   
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "normal", "xyz", "hit"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"sdf"}

    def trace(self, nef, channels, extra_channels, rays, num_steps=64, step_size=1.0, min_dis=1e-4):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  query those extra channels at surface intersection points.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            num_steps (int): The number of steps to use for sphere tracing.
            step_size (float): The multiplier for the sphere tracing steps. 
                               Use a value <1.0 for conservative tracing.
            min_dis (float): The termination distance for sphere tracing.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        timer = PerfTimer(activate=False)

        # Distanace from ray origin
        t = torch.zeros(rays.origins.shape[0], 1, device=rays.origins.device)

        # Position in model space
        x = torch.addcmul(rays.origins, rays.dirs, t)

        cond = torch.ones_like(t).bool()[:, 0]
        
        normal = torch.zeros_like(x)
        # This function is in fact differentiable, but we treat it as if it's not, because
        # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
        # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
        # locations, where additional quantities (normal, depth, segmentation) can be determined. The
        # gradients will propagate only to these locations. 
        with torch.no_grad():

            d = nef(coords=x, channels="sdf")
            
            dprev = d.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            #cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(d).byte()
            
            for i in range(num_steps):
                timer.check("start")
                # 1. Check if ray hits.
                #hit = (torch.abs(d) < self._MIN_DIS)[:,0] 
                # 2. Check that the sphere tracing is not oscillating
                #hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]
                
                # 3. Check that the ray has not exit the far clipping plane.
                #cond = (torch.abs(t) < clamp[1])[:,0]
                
                hit = (torch.abs(t) < rays.dist_max)[:,0]
                
                # 1. not hit surface
                cond = cond & (torch.abs(d) > min_dis)[:,0] 

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > min_dis * 3)[:,0]
                
                # 3. not a hit
                cond = cond & hit
                
                #cond = cond & ~hit
                
                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break

                # Advance the x, by updating with a new t
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(rays.origins, rays.dirs, t), x)
                
                # Store the previous distance
                dprev = torch.where(cond.unsqueeze(1), d, dprev)

                # Update the distance to surface at x
                d[cond] = nef(coords=x[cond], channels="sdf") * step_size

                # Update the distance from origin 
                t = torch.where(cond.view(cond.shape[0], 1), t+d, t)
                timer.check("end")
    
        # AABB cull 

        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
        #hit = torch.ones_like(d).byte()[...,0]
        
        # The function will return 
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=x[hit], lod_idx=lod_idx, channels=channel)
            extra_buffer = torch.zeros(*x.shape[:-1], feats.shape[-1], device=feats.device)
            extra_buffer[hit] = feats
            extra_outputs[channel] = extra_buffer

        if "normal" in channels:
            if hit.any():
                grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
                _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
                normal[hit] = _normal
        else:
            normal = None

        return RenderBuffer(xyz=x, depth=t, hit=hit, normal=normal, **extra_outputs)
