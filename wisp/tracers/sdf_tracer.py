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

    def forward(self, nef, ray_o, ray_d):
        """PyTorch implementation of sphere tracing."""
        timer = PerfTimer(activate=False)
        supported_channels = nef.get_supported_channels()
        assert "sdf" in supported_channels and "this tracer requires sdf channels"

        # Distanace from ray origin
        t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

        # Position in model space
        x = torch.addcmul(ray_o, ray_d, t)

        cond = torch.ones_like(t).bool()[:,0]
        
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
            
            for i in range(self.num_steps):
                timer.check("start")
                # 1. Check if ray hits.
                #hit = (torch.abs(d) < self._MIN_DIS)[:,0] 
                # 2. Check that the sphere tracing is not oscillating
                #hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]
                
                # 3. Check that the ray has not exit the far clipping plane.
                #cond = (torch.abs(t) < self.clamp[1])[:,0]
                
                hit = (torch.abs(t) < self.camera_clamp[1])[:,0]
                
                # 1. not hit surface
                cond = cond & (torch.abs(d) > self.min_dis)[:,0] 

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > self.min_dis * 3)[:,0]
                
                # 3. not a hit
                cond = cond & hit
                
                #cond = cond & ~hit
                
                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break

                # Advance the x, by updating with a new t
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                
                # Store the previous distance
                dprev = torch.where(cond.unsqueeze(1), d, dprev)

                # Update the distance to surface at x
                d[cond] = nef(coords=x[cond], channels="sdf") * self.step_size

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
        
        if hit.any():
            grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
            _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
            normal[hit] = _normal
        
        return RenderBuffer(x=x, depth=t, hit=hit, normal=normal)
