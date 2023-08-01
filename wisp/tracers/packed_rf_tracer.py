# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.tracers import BaseTracer
from typing import Tuple

class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply

def exponential_integration(feats, tau, boundaries, exclusive=True):
    r"""Exponential transmittance integration across packs using the optical thickness (tau).

    Exponential transmittance is derived from the Beer-Lambert law. Typical implementations of
    exponential transmittance is calculated with :func:`cumprod`, but the exponential allows a reformulation
    as a :func:`cumsum` which its gradient is more stable and faster to compute. We opt to use the :func:`cumsum`
    formulation.

    For more details, we recommend "Monte Carlo Methods for Volumetric Light Transport" by Novak et al.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
        tau (torch.FloatTensor): optical thickness of shape :math:`(\text{num_rays}, 1)`.
        boundaries (torch.BoolTensor): bools of shape :math:`(\text{num_rays})`.
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`.
        exclusive (bool): Compute exclusive exponential integration if true. (default: True)

    Returns:
        (torch.FloatTensor, torch.FloatTensor)
        - Integrated features of shape :math:`(\text{num_packs}, \text{num_feats})`.
        - Transmittance of shape :math:`(\text{num_rays}, 1)`.

    """
    # TODO(ttakikawa): This should be a fused kernel... we're iterating over packs, so might as well
    #                  also perform the integration in the same manner.
    alpha = 1.0 - trunc_exp(-tau.contiguous())
    # Uses the reformulation as a cumsum and not a cumprod (faster and more stable gradients)
    transmittance = trunc_exp(-1.0 * spc_render.cumsum(tau.contiguous(), boundaries.contiguous(), exclusive=exclusive))
    transmittance = transmittance * alpha
    feats_out = spc_render.sum_reduce(transmittance * feats.contiguous(), boundaries.contiguous())
    return feats_out, transmittance

class PackedRFTracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is differentiable, and can be employed within training loops.

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """
    def __init__(self,
        raymarch_type : str = 'ray',  # options: 'voxel', 'ray'
        num_steps     : int = 1024,
        step_size     : float = 1.0,
        bg_color      : Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Set the default trace() arguments.

        Args:
            raymarch_type (str): Sample generation strategy to use for raymarch.
                'voxel' - intersects the rays with the acceleration structure cells.
                    Then among the intersected cells, each cell is sampled `num_steps` times.
                'ray' - samples `num_steps` along each ray, and then filters out samples which falls outside of occupied
                    cells of the acceleration structure.
            num_steps (int): The number of steps to use for the sampling. The meaning of this parameter changes
                depending on `raymarch_type`:
                'voxel' - each acceleration structure cell which intersects a ray is sampled `num_steps` times.
                'ray' - number of samples generated per ray, before culling away samples which don't fall
                    within occupied cells.
                The exact number of samples generated, therefore, depends on this parameter but also the occupancy
                status of the acceleration structure.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (Tuple[float, float, float]): The background color to use.
        """
        super().__init__(bg_color=bg_color)
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)
        self.prev_num_samples = None

    def get_prev_num_samples(self):
        """Returns the number of ray samples that were executed.
        
        Returns None if the tracer has never ran.

        Returns:
            (int): The number of ray samples.
        """
        return self.prev_num_samples

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "hit", "rgb", "alpha"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"rgb", "density"}

    def trace(self, nef, rays, channels, extra_channels,
              lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white'):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (Tuple[float, float, float]): The background color to use.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        #TODO(ttakikawa): Use a more robust method
        assert nef.grid is not None and "this tracer requires a grid"

        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1
        
        # By default, PackedRFTracer will attempt to use the highest level of detail for the ray sampling.
        # This however may not actually do anything; the ray sampling behaviours are often single-LOD
        # and is governed by however the underlying feature grid class uses the BLAS to implement the sampling.
        raymarch_results = nef.grid.raymarch(rays,
                                             level=nef.grid.active_lods[lod_idx],
                                             num_samples=num_steps,
                                             raymarch_type=raymarch_type)
        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        depths = raymarch_results.depth_samples
        self.prev_num_samples = samples.shape[0]
     
        pack_info = raymarch_results.pack_info
        boundary = raymarch_results.boundary
        
        hit_ray_d = rays.dirs.index_select(0, ridx)
        # Compute the color and density for each ray and their samples
        num_samples = samples.shape[0]
        color, density = nef(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=["rgb", "density"])
        density = density.reshape(num_samples, 1)    # Protect against squeezed return shape
        extra_outputs = {}
        self.bg_color = self.bg_color.to(rays.origins.device)

        if "depth" in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else: 
            depth = None
        
        rgb = torch.zeros(N, 3, device=rays.origins.device) + self.bg_color
        
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[boundary]

        # Compute optical thickness
        tau = density * deltas
        del density, deltas
        ray_colors, transmittance = exponential_integration(color, tau, boundary, exclusive=True)

        if "depth" in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(num_samples, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth

        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[...,0] > 0.0
        
        # Populate the background
        rgb[ridx_hit] = (self.bg_color * (1.0-alpha)) + ray_colors

        for channel in extra_channels:
            feats = nef(coords=samples,
                        ray_d=hit_ray_d,
                        lod_idx=lod_idx,
                        channels=channel)
            num_channels = feats.shape[-1]
            ray_feats, transmittance = exponential_integration(
                feats.view(num_samples, num_channels), tau, boundary, exclusive=True
            )
            composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, **extra_outputs)

