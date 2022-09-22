# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Set, Tuple
import torch
from kaolin.render.camera import Camera
from wisp.core import RenderBuffer, Rays, PrimitivesPack
from wisp.models import Pipeline
from wisp.models.nefs import BaseNeuralField
from wisp.gfx.datalayers import Datalayers


@dataclass
class FramePayload:
    """This is a dataclass which holds metadata for the current frame.
    """
    camera: Camera
    visible_objects: Set[str]
    interactive_mode: bool
    render_res_x: int
    render_res_y: int
    time_delta: float   # In seconds
    clear_color: Tuple[float, float, float]
    channels: Set[str] # Channels requested for the render.

class BottomLevelRenderer(ABC):

    def __init__(self, *args, **kwargs):
        self._model_matrix = None
        self._bbox = None
        self._data_layers = dict()

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        """ General frame setup occurs here, for example:
            1. Ray Tracers - ray generation.
            2. Update shader uniforms.
        """
        pass

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        """ Override to optimize cases when the neural field does not require rendering from scratch. """
        return True

    @abstractmethod
    def render(self, *args, **kwargs) -> RenderBuffer:
        raise NotImplementedError('BottomLevelRenderer subclasses must implement render() logic.')

    def redraw(self):
        if self.needs_redraw():
            self._data_layers = self.regenerate_data_layers()

    def post_render(self, *args, **kwargs) -> None:
        """ General frame teardown logic takes place, for example:
            1. Cleanup of temporary information generated during the frame
            2. Caching of information relevant for the next frames.
        """
        pass

    def needs_redraw(self) -> bool:
        return True

    def regenerate_data_layers(self) -> Dict[str, PrimitivesPack]:
        return dict()

    def data_layers(self) -> Dict[str, PrimitivesPack]:
        """ Returns layers of information made of primitives, visually describing the renderer internal structures """
        return self._data_layers

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError('BottomLevelRenderer subclasses must implement device')

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        raise NotImplementedError('BottomLevelRenderer subclasses must implement dtype')

    @property
    @abstractmethod
    def model_matrix(self) -> torch.Tensor:
        """ torch.Tensor of 4x4 matrix defining how the object local coordinates should transform to world coordinates.
        """
        raise NotImplementedError('BottomLevelRenderer subclasses must implement model_matrix() logic.')

    @property
    @abstractmethod
    def aabb(self) -> torch.Tensor:
        """ torch.Tensor defining the axis-aligned bounding box of object as:
            (center_x, center_y, center_z, width, height, depth) """
        raise NotImplementedError('BottomLevelRenderer subclasses must implement bbox() logic.')


class RayTracedRenderer(BottomLevelRenderer):
    def __init__(self, nef: BaseNeuralField, *args, **kwargs):
        super().__init__(nef, *args, **kwargs)
        self.nef = nef.eval()
        self.layers_painter = self.create_layers_painter(self.nef)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, *args, **kwargs):
        """ Builds a bottom level renderer from the building block of a pipeline. """
        # Build the renderer using the pipeline's tracer args
        # The renderer is protected against extra **kwargs that may exist inside the tracer
        tracer_args = pipeline.tracer.__dict__

        # Allow the constructor to override the tracer's args with **kwargs specified manually
        tracer_args.update(**kwargs)
        tracer_args['tracer_type'] = type(pipeline.tracer)
        return cls(nef=pipeline.nef, *args, **tracer_args)

    @classmethod
    @abstractmethod
    def create_layers_painter(cls, nef: BaseNeuralField) -> Optional[Datalayers]:
        return None

    def needs_redraw(self) -> bool:
        if self.layers_painter is not None:
            return self.layers_painter.needs_redraw(self.nef.grid)
        else:
            return True

    def regenerate_data_layers(self) -> Dict[str, PrimitivesPack]:
        if self.layers_painter is not None:
            return self.layers_painter.regenerate_data_layers(self.nef.grid)
        else:
            return dict()

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        """ Prepare primary rays to render """
        pass

    @abstractmethod
    def render(self, rays: Optional[Rays] = None) -> RenderBuffer:
        pass

    def post_render(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return self.nef.device


class RasterizedRenderer(BottomLevelRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def render(self, camera: Camera) -> RenderBuffer:
        raise NotImplementedError('RasterizedRenderer subclasses must implement render() logic.')
