# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Optional, Dict
import torch
from wisp.core import RenderBuffer
from wisp.renderer.core.api import RayTracedRenderer, FramePayload, field_renderer
from wisp.core import Rays
from wisp.models.nefs.neural_sdf import NeuralSDF, BaseNeuralField
from wisp.tracers import PackedSDFTracer
from wisp.accelstructs import OctreeAS
from wisp.gfx.datalayers import Datalayers, OctreeDatalayers


@field_renderer(BaseNeuralField, PackedSDFTracer)
class NeuralSDFPackedRenderer(RayTracedRenderer):
    """ A neural field renderers for pipelines of NeuralSDF + PackedSDFTracer.
        The renderer is registered with the general BaseNeuralField to make a default fallback for future neural field
        subclasses which use the PackedSDFTracer and don't implement a dedicated renderer.
    """

    def __init__(self, nef: NeuralSDF, tracer_type=None,
                 samples_per_ray=None, min_distance=None, raymarch_type=None, *args, **kwargs):
        super().__init__(nef, *args, **kwargs)
        if min_distance is None:
            min_distance = 0.0003
        if samples_per_ray is None:
            samples_per_ray = 32
        self.samples_per_ray = samples_per_ray
        if raymarch_type is None:
            raymarch_type = 'voxel'

        self.tracer = tracer_type() if tracer_type is not None else PackedSDFTracer()
        self.render_res_x = None
        self.render_res_y = None
        self.output_width = None
        self.output_height = None
        self.far_clipping = None
        self.channels = None
        self._last_state = dict()

        self._data_layers = self.regenerate_data_layers()

    @classmethod
    def create_layers_painter(cls, nef: BaseNeuralField) -> Optional[Datalayers]:
        if nef.grid_type in ('OctreeGrid', 'CodebookOctreeGrid', 'HashGrid'):
            return OctreeDatalayers()
        else:
            return None

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        super().pre_render(payload)
        self.render_res_x = payload.render_res_x
        self.render_res_y = payload.render_res_y
        self.output_width = payload.camera.width
        self.output_height = payload.camera.height
        self.far_clipping = payload.camera.far
        self.tracer.num_steps = self.samples_per_ray
        self.tracer.bg_color = 'black' if payload.clear_color == (0.0, 0.0, 0.0) else 'white'
        self.channels = payload.channels

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        return self._last_state.get('num_steps', 0) < self.samples_per_ray or \
               self._last_state.get('channels') != self.channels

    def render(self, rays: Optional[Rays] = None) -> RenderBuffer:
        rb = self.tracer(self.nef, channels=self.channels, rays=rays)

        # Rescale renderbuffer to original size
        rb = rb.reshape(self.render_res_y, self.render_res_x, -1)
        if self.render_res_x != self.output_width or self.render_res_y != self.output_height:
            rb = rb.scale(size=(self.output_height, self.output_width))
        return rb

    def post_render(self) -> None:
        self._last_state['num_steps'] = self.tracer.num_steps
        self._last_state['channels'] = self.channels

    @property
    def device(self) -> torch.device:
        return next(self.nef.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def model_matrix(self) -> torch.Tensor:
        return torch.eye(4, device=self.device)

    @property
    def aabb(self) -> torch.Tensor:
        # (center_x, center_y, center_z, width, height, depth)
        return torch.tensor((0.0, 0.0, 0.0, 2.0, 2.0, 2.0), device=self.device)

    def acceleration_structure(self):
        if isinstance(self.nef.grid.blas, OctreeAS):
            return "Octree"
        else:
            return "None"

    def features_structure(self):
        if self.nef.grid_type == "OctreeGrid":
            return "Octree Grid"
        elif self.nef.grid_type == "CodebookOctreeGrid":
            return "Codebook Grid"
        elif self.nef.grid_type == "TriplanarGrid":
            return "Triplanar Grid"
        elif self.nef.grid_type == "HashGrid":
            return "Hash Grid"
        else:
            return "Unknown"



