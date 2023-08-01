# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from collections import defaultdict, deque
from typing import Type, TYPE_CHECKING, Union
from wisp.models import Pipeline, RasterizationPipeline
from wisp.models.nefs import BaseNeuralField
from wisp.tracers import BaseTracer
if TYPE_CHECKING:  # Prevent circular imports mess due to typed annotations
    from wisp.renderer.core.api.base_renderer import BottomLevelRenderer, RasterizedRenderer
    from wisp.renderer.core.api.raytraced_renderer import RayTracedRenderer

# All renderers supported by the interactive renderer should be registered here
# Registration can be quickly achieved via the @field_renderer decorator.
# Make sure custom modules are imported before neural renderers are constructed!
_REGISTERED_RENDERABLE_NEURAL_FIELDS = defaultdict(dict)
_REGISTERED_RENDERABLE_RASTERIZERS = defaultdict()

def register_neural_field_type(neural_field_type: Type[BaseNeuralField],
                               tracer_type: Type[BaseTracer],
                               renderer_type: Type[BottomLevelRenderer]):
    """ Register new types of neural fields with their associated bottom level renderers using this function.
        This allows the interactive renderer to display this neural field type on the canvas.
    """
    field_name = neural_field_type.__name__
    tracer_name = tracer_type.__name__
    _REGISTERED_RENDERABLE_NEURAL_FIELDS[field_name][tracer_name] = renderer_type


def register_rasterizer_type(rasterizer_type: Type, renderer_type: Type[BottomLevelRenderer]):
    """ Register new types of rasterizers with their associated bottom level renderers using this function.
        This allows the interactive renderer to display this rasterizer type on the canvas.
    """
    _REGISTERED_RENDERABLE_RASTERIZERS[rasterizer_type.__name__] = renderer_type


def _neural_field_to_renderer_cls(pipeline: Pipeline) -> Type[RayTracedRenderer]:
    tracer_type = type(pipeline.tracer)

    # Start by iterating the current tracer type - look for renderers compatible with the current nef type
    # or any of its parents (the hierarchy of nefs take precedence over the hierarchy of tracers).
    renderer_cls = None
    while tracer_type:
        tracer_name = tracer_type.__name__

        # Look for a renderer compatible with the current tracer type and nef classes
        type_queue = deque([type(pipeline.nef)])

        renderer_cls = None
        while type_queue:
            # Query nef + tracer combo
            field_type = type_queue.popleft()
            field_name = field_type.__name__
            supported_tracers = _REGISTERED_RENDERABLE_NEURAL_FIELDS.get(field_name)
            renderer_cls = supported_tracers.get(tracer_name) if supported_tracers is not None else None

            # Current nef + tracer pair doesn't match any registered renderer
            if renderer_cls is not None:
                break
            else:   # Try querying all parent(nef) + tracer combos for compatibility
                bases = field_type.__bases__
                if len(bases) > 0:
                    for b in bases:
                        type_queue.append(b)

        if renderer_cls is not None:
            break   # Found a renderer class
        else:
            # Didn't find a renderer class - repeat the process with the parent class of the tracer
            tracer_base_types = tracer_type.__bases__
            if len(tracer_base_types) > 0:
                # Does tracer have a single parent class which inherits from BaseTracer?
                # If so, keep looking
                tracer_base_types = [base for base in tracer_type.__bases__ if issubclass(base, BaseTracer)]
                tracer_type = tracer_base_types[0] if len(tracer_base_types) == 1 else None
            else:
                # Reached end of tracers hierarchy or it is too ambiguous, quit and fail gracefully
                tracer_type = None

    if tracer_type is None:
        raise ValueError(f'Renderer factory encountered an unknown neural pipeline: '
                         f'Neural Field {type(pipeline.nef).__name__} with tracer {type(pipeline.tracer).__name__}. '
                         'Please register the factory to reflect what kind of renderer should be created for this '
                         'type of neural field/tracer.')
    return renderer_cls

def _neural_rasterizer_to_renderer_cls(pipeline: RasterizationPipeline) -> Type[RasterizedRenderer]:
    # Start by iterating the current rasterizer_type type -
    # look for renderers compatible with the current rasterizer_type type or any of its parents
    renderer_cls = None
    type_queue = deque([type(pipeline.rasterizer)])
    while type_queue:
        # Query rasterizer
        rast_type = type_queue.popleft()
        rast_name = rast_type.__name__
        # Look for a renderer compatible with the current rasterizer type
        renderer_cls = _REGISTERED_RENDERABLE_RASTERIZERS.get(rast_name)

        if renderer_cls is not None:
            break   # Matching renderer found
        else:   # Current rasterizer class doesn't match any registered renderer
            # Try querying all parent(rast_type) for compatibility
            bases = rast_type.__bases__
            if len(bases) > 0:
                for b in bases:
                    type_queue.append(b)

    if renderer_cls is None:
        raise ValueError(f'Renderer factory encountered an unknown neural rasterization pipeline: '
                         f'{type(pipeline.rasterizer).__name__}. '
                         'Please register the factory to reflect what kind of renderer should be created for this '
                         'type of rasterization pipeline.')
    return renderer_cls

def create_neural_field_renderer(neural_object: Union[Pipeline, RasterizationPipeline], **kwargs) -> BottomLevelRenderer:
    # Fetch the neural field object. If given a pipeline, query the nef field.
    # Otherwise assume we're given the neural field explicitly.
    if isinstance(neural_object, Pipeline):
        # Obtain the appropriate renderer class, compatible with this neural field type and tracer
        renderer_cls = _neural_field_to_renderer_cls(neural_object)

        # In the case of a pipeline, we use a specialized constructor which builds the renderer from its components
        # kwargs may override the pipeline default settings
        renderer_instance = renderer_cls.from_pipeline(neural_object, **kwargs)
    elif isinstance(neural_object, RasterizationPipeline):
        renderer_cls = _neural_rasterizer_to_renderer_cls(neural_object)
        # In the case of a rasterization pipeline,
        # we use a specialized constructor which builds the renderer from its components
        # kwargs may override the pipeline default settings
        renderer_instance = renderer_cls.from_pipeline(neural_object, **kwargs)
    else:
        raise NotImplementedError(f"Unknown pipeline type: {neural_object}")
    return renderer_instance
