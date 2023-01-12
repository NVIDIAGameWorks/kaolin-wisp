# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from wisp.core import RenderBuffer, Rays, PrimitivesPack
from wisp.renderer.core.api import RayTracedRenderer, FramePayload, field_renderer
from wisp.models.nefs.nerf import NeuralRadianceField, BaseNeuralField
from wisp.tracers import PackedRFTracer


@field_renderer(BaseNeuralField, PackedRFTracer)
class NeuralRadianceFieldPackedRenderer(RayTracedRenderer):
    """ A neural field renderers for pipelines of NeuralRadianceField + PackedRFTracer.
        The renderer is registered with the general BaseNeuralField to make a default fallback for future neural field
        subclasses which use the PackedRFTracer and don't implement a dedicated renderer.

        NeuralRadianceFieldPackedRenderer is capable of reducing the tracer settings to lower quality when interactive
        mode is on.
    """

    def __init__(self, nef: NeuralRadianceField, tracer: PackedRFTracer, batch_size=2**14, num_steps=None,
                 raymarch_type=None, *args, **kwargs):
        super().__init__(nef, tracer, batch_size, *args, **kwargs)
        if num_steps is None:
            num_steps = 2
        self.num_steps = num_steps
        self.num_steps_movement = max(num_steps // 4, 1)
        if raymarch_type is None:
            raymarch_type = 'voxel'
        self.raymarch_type = raymarch_type
        self.bg_color = 'black'

        self._last_state = dict()

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        super().pre_render(payload)
        self.bg_color = 'black' if payload.clear_color == (0.0, 0.0, 0.0) else 'white'
        if payload.interactive_mode:
            self.tracer.num_steps = self.num_steps_movement
        else:
            self.tracer.num_steps = self.num_steps

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        return self._last_state.get('num_steps', 0) < self.num_steps or \
               self._last_state.get('channels') != self.channels

    def render(self, rays: Rays) -> RenderBuffer:
        rb = RenderBuffer(hit=None)
        for ray_batch in rays.split(self.batch_size):
            rb += self.tracer(self.nef,
                              rays=ray_batch,
                              channels=self.channels,
                              lod_idx=None,  # TODO(ttakikawa): Add a way to control the LOD in the GUI
                              raymarch_type=self.raymarch_type,
                              num_steps=self.num_steps,
                              bg_color=self.bg_color)

        # Rescale renderbuffer to original size
        rb = rb.reshape(self.render_res_y, self.render_res_x, -1)
        if self.render_res_x != self.output_width or self.render_res_y != self.output_height:
            rb = rb.scale(size=(self.output_height, self.output_width))
        return rb

    def post_render(self) -> None:
        self._last_state['num_steps'] = self.tracer.num_steps
        self._last_state['channels'] = self.channels
