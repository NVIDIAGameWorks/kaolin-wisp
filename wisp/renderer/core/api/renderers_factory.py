# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from collections import defaultdict, deque
from typing import Type
from wisp.models import Pipeline
from wisp.models.nefs import BaseNeuralField
from wisp.tracers import BaseTracer
from wisp.renderer.core.api.base_renderer import BottomLevelRenderer, RayTracedRenderer

# All renderers supported by the interactive renderer should be registered here
# Registration can be quickly achieved via the @field_renderer decorator.
# Make sure custom modules are imported before neural renderers are constructed!
_REGISTERED_RENDERABLE_NEURAL_FIELDS = defaultdict(dict)


def register_neural_field_type(neural_field_type: Type[BaseNeuralField],
                               tracer_type: Type[BaseTracer],
                               renderer_type: Type[BottomLevelRenderer]):
    """ Register new types of neural fields with their associated bottom level renderers using this function.
        This allowes the interactive renderer to display this neural field type on the canvas.
    """
    field_name = neural_field_type.__name__
    tracer_name = tracer_type.__name__
    _REGISTERED_RENDERABLE_NEURAL_FIELDS[field_name][tracer_name] = renderer_type


def _neural_field_to_renderer_cls(pipeline: Pipeline) -> Type[RayTracedRenderer]:
    tracer_name = type(pipeline.tracer).__name__

    # Start by looking for a renderer compatible with the pipeline tracer and nef classes
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
            type_queue.append(*field_type.__bases__)

    if renderer_cls is None:
        raise ValueError(f'Renderer factory encountered an unknown neural pipeline: '
                         f'Neural Field {type(pipeline.nef).__name__} with tracer {tracer_name}. '
                         'Please register the factory to reflect what kind of renderer should be created for this '
                         'type of neural field/tracer.')
    return renderer_cls


def create_neural_field_renderer(neural_object: Pipeline, **kwargs) -> RayTracedRenderer:
    # TODO (operel): support creation of non ray traced renderers

    # Fetch the neural field object. If given a pipeline, query the nef field.
    # Otherwise assume we're given the neural field explicitly.
    if isinstance(neural_object, Pipeline):
        # Obtain the appropriate renderer class, compatible with this neural field type and tracer
        renderer_cls = _neural_field_to_renderer_cls(neural_object)

        # In the case of a pipeline, we use a specialized constructor which builds the renderer from its components
        # kwargs may override the pipeline default settings
        renderer_instance = renderer_cls.from_pipeline(neural_object, **kwargs)
    # else:
        # TODO (operel): not sure we even need this path..
        #   Would a NeF ever exist without a tracer? If needed we can just choose any tracer combo
        # # Obtain the appropriate renderer class, compatible with this neural field type
        # nef = neural_object
        # renderer_cls = _neural_field_to_renderer_cls(nef)
        # renderer_args = kwargs
        #
        # # Create an instance of a renderer, wrapping the neural field and using the given arguments
        # renderer_instance = renderer_cls(nef, **renderer_args)

    return renderer_instance


# TODO (operel): Double check the interactive app can loads a model from file and call create_renderer
# Model loading should now be at the app's responsibility
# # Load nerf if model_path is available and nerf is not
# if nerf is None:
#     model_path = kwargs['model_path']
#     nerf = torch.load(model_path)
