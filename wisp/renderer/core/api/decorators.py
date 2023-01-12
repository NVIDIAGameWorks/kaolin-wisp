# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Type
from wisp.models.nefs import BaseNeuralField
from wisp.tracers import BaseTracer
from wisp.renderer.core.api.base_renderer import BottomLevelRenderer
from wisp.renderer.core.api.renderers_factory import register_neural_field_type


def field_renderer(field_type: Type[BaseNeuralField], tracer_type: Type[BaseTracer]):
    """ A decorator that registers a neural field type with a renderer.
        By registering the renderer type, the interactive renderer knows what type of renderer to create
        when dealing with this type of field.
        Essentially, this allows displaying custom types of objects on the canvas.
    """
    def _register_renderer_fn(renderer_class: Type[BottomLevelRenderer]):
        register_neural_field_type(field_type, tracer_type, renderer_class)
        return renderer_class
    return _register_renderer_fn
