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

    def __init__(self, nef: NeuralSDF, tracer: PackedSDFTracer, num_steps=None, step_size=None, min_dis=None,
                 *args, **kwargs):
        """
        Construct a new neural signed distance field from the nef + tracer pipeline.
        By default, tracing will use the tracer args, unless specific values are specified for overriding those
        defaults.

        Args:
            nef (BaseNeuralField): Neural field component of the pipeline. The neural field is expected to be coordinate
             based.
            tracer (PackedSDFTracer): Tracer component of the pipeline, assumed to trace signed distance fields.
            num_steps (int): Number of steps used for sphere tracing.
                By default, the tracer num_steps will be used. Specify an explicit value to override.
            step_size (float): The multiplier for the sphere tracing steps. Use a value <1.0 for conservative tracing.
                By default, the tracer step_size will be used. Specify an explicit value to override.
            min_dis (float): The termination distance for sphere tracing.
                By default, the tracer min_dis will be used. Specify an explicit value to override.
        """
        super().__init__(nef, tracer, *args, **kwargs)
        self.num_steps = num_steps if num_steps is not None else tracer.num_steps
        self.step_size = step_size if step_size is not None else tracer.step_size
        self.min_dis = min_dis if min_dis is not None else tracer.min_dis
        self._last_state = dict()

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        return self._last_state.get('num_steps', 0) != self.num_steps or \
               self._last_state.get('step_size', 0) != self.step_size or \
               self._last_state.get('min_dis', 0) != self.min_dis or \
               self._last_state.get('channels') != self.channels

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        super().pre_render(payload)
        self.tracer.num_steps = self.num_steps
        self.tracer.step_size = self.step_size
        self.tracer.min_dis = self.min_dis
        self.tracer.bg_color = 'black' if payload.clear_color == (0.0, 0.0, 0.0) else 'white'

    def post_render(self) -> None:
        self._last_state['num_steps'] = self.tracer.num_steps
        self._last_state['step_size'] = self.tracer.step_size
        self._last_state['min_dis'] = self.tracer.min_dis
        self._last_state['channels'] = self.channels
