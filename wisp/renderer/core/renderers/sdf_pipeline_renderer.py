# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from wisp.renderer.core.api import RayTracedRenderer, FramePayload, field_renderer
from wisp.models.nefs.neural_sdf import NeuralSDF, BaseNeuralField
from wisp.tracers import PackedSDFTracer


@field_renderer(BaseNeuralField, PackedSDFTracer)
class NeuralSDFPackedRenderer(RayTracedRenderer):
    """ A neural field renderers for pipelines of NeuralSDF + PackedSDFTracer.
        The renderer is registered with the general BaseNeuralField to make a default fallback for future neural field
        subclasses which use the PackedSDFTracer and don't implement a dedicated renderer.
    """

    def __init__(self, nef: NeuralSDF, tracer: PackedSDFTracer, samples_per_ray=32, *args, **kwargs):
        super().__init__(nef, tracer, *args, **kwargs)
        self.samples_per_ray = samples_per_ray
        self._last_state = dict()

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        return self._last_state.get('num_steps', 0) < self.samples_per_ray or \
               self._last_state.get('channels') != self.channels

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        super().pre_render(payload)
        self.tracer.num_steps = self.samples_per_ray
        self.tracer.bg_color = 'black' if payload.clear_color == (0.0, 0.0, 0.0) else 'white'

    def post_render(self) -> None:
        self._last_state['num_steps'] = self.tracer.num_steps
        self._last_state['channels'] = self.channels
