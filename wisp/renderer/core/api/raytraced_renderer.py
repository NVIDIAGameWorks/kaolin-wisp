# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Optional, Dict
import copy
import torch
from wisp.core import RenderBuffer, Rays, PrimitivesPack
from wisp.models import Pipeline
from wisp.models.nefs import BaseNeuralField
from wisp.tracers import BaseTracer
from wisp.renderer.core.api.base_renderer import BottomLevelRenderer, FramePayload
from wisp.renderer.core.api.decorators import field_renderer
from wisp.gfx.datalayers import Datalayers, OctreeDatalayers, AABBDatalayers
from wisp.accelstructs import OctreeAS, AxisAlignedBBoxAS


@field_renderer(BaseNeuralField, BaseTracer)
class RayTracedRenderer(BottomLevelRenderer):
    """ A default neural field renderers for all neural ray-based pipelines.
        The renderer is registered with the general BaseNeuralField & BaseTracer
        to make a default fallback for future combos of neural field / tracer which
        don't implement a dedicated renderer.

        Renderers represent "visualizable" objects in the interactive renderer system.
        These include anything that could potentially be painted on the canvas, how to tune it, and what data-layers
        it supports.
        This specific renderer is concerned with neural objects which rely on ray tracing.

        This class also serves as a convenience super-class for other renderers which shouldn't implement the entire
        BottomLevelRenderer functionality from scratch.
    """
    def __init__(self, nef: BaseNeuralField, tracer: BaseTracer, batch_size=None, *args, **kwargs):
        super().__init__(nef, *args, **kwargs)
        self.nef = nef.eval()
        self.tracer = tracer
        self.layers_painter = self.create_layers_painter(self.nef)

        self.batch_size = batch_size
        self._data_layers = self.regenerate_data_layers()

        # Per frame cached info
        self.render_res_x = None
        self.render_res_y = None
        self.output_width = None
        self.output_height = None
        self.far_clipping = None
        self.channels = None

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, **kwargs):
        """ Builds a bottom level renderer from the building block of a pipeline. """
        # Create a copy of the pipeline's tracer, with the same type and values
        # This allows the renderer to modify the interactive tracer without affecting the pipeline
        tracer = copy.deepcopy(pipeline.tracer)

        # Pass the kwargs to the renderer also, in case it wants to further modify the tracer with them.
        return cls(nef=pipeline.nef, tracer=tracer, **kwargs)

    @classmethod
    def create_layers_painter(cls, nef: BaseNeuralField) -> Optional[Datalayers]:
        """ NeuralRadianceFieldPackedRenderer can draw datalayers showing the occupancy status.
        These depend on the bottom level acceleration structure.
        """
        if not hasattr(nef.grid, 'blas'):
            return None
        elif isinstance(nef.grid.blas, AxisAlignedBBoxAS):
            return AABBDatalayers()
        elif isinstance(nef.grid.blas, OctreeAS):
            return OctreeDatalayers()
        else:
            return None

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        """ Should the neural field be "refreshed" (meaning: retraced to a new RenderBuffer).
            Retracing here generally means invoking calls that do not upload new data to the GPU.
            The general logic cannot assume anything about the state of the neural field and therefore opts
            to always refresh it.
            WARNING: This is suboptimal for neural fields which take time to retrace.
            Users are recommended to subclass and override this function by keeping track of the last state
            and what fields changed since the last invocation.
        """
        return True

    def needs_redraw(self) -> bool:
        """ Returns if a full redrawing of the object should occur.
            This includes regeneration of buffers that may get uploaded to the GPU, such as vectorial data layers.
        """
        if self.layers_painter is not None:
            return self.layers_painter.needs_redraw(self.nef.grid.blas)
        else:
            return True

    def regenerate_data_layers(self) -> Dict[str, PrimitivesPack]:
        """ Regenarates the vectorial data layers depicting additional information about the object. """
        if self.layers_painter is not None:
            return self.layers_painter.regenerate_data_layers(self.nef.grid.blas)
        else:
            return dict()

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        """ Actions which should take place before rendering:
        Store information from payload, adjust resolution, and so forth.
        """
        super().pre_render(payload)
        self.render_res_x = payload.render_res_x
        self.render_res_y = payload.render_res_y
        self.output_width = payload.camera.width
        self.output_height = payload.camera.height
        self.far_clipping = payload.camera.far
        self.channels = payload.channels

    def render(self, rays: Rays) -> RenderBuffer:
        rb = RenderBuffer(hit=None)

        # Feed as single batch
        if self.batch_size is None:
            rb = self.tracer(self.nef, rays=rays, channels=self.channels)
        else:   # Apply in batches
            for ray_batch in rays.split(self.batch_size):
                rb += self.tracer(self.nef, rays=ray_batch, channels=self.channels)

        # Rescale renderbuffer to original size
        rb = rb.reshape(self.render_res_y, self.render_res_x, -1)
        if self.render_res_x != self.output_width or self.render_res_y != self.output_height:
            rb = rb.scale(size=(self.output_height, self.output_width))
        return rb

    def post_render(self) -> None:
        """ Actions which should take place after rendering:
        Store information for next frame.
        """
        pass

    def acceleration_structure(self) -> str:
        """ Returns a human readable name of the bottom level acceleration structure used by this renderer """
        if getattr(self.nef, 'grid') is None or getattr(self.nef.grid, 'blas') is None:
            return "None"
        elif hasattr(self.nef.grid.blas, 'name'):
            return self.nef.grid.blas.name()
        else:
            return "Unknown"

    def features_structure(self) -> str:
        """ Returns a human readable name of the feature structure used by this renderer """
        if getattr(self.nef, 'grid') is None:
            return "None"
        elif hasattr(self.nef.grid, 'name'):
            return self.nef.grid.name()
        else:
            return "Unknown"

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return self.nef.device
