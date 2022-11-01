# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
from wisp.core import ObjectTransform
from wisp.models import Pipeline
from wisp.framework import WispState, BottomLevelRendererState


def request_redraw(state):
    """ Marks the canvas as dirty,
    forcing the renderer core to refresh the object renderers on the next rendering iteration.
    """
    state.renderer.canvas_dirty = True


def add_pipeline_to_scene_graph(state: WispState,
                                name: str,
                                pipeline: Pipeline,
                                transform: ObjectTransform = None,
                                **setup_args):
    """ Adds a new object pipeline to the scene graph.
    The pipeline contains all components required to trace this object.
    The object is represented in the SceneGraph by a newly created BottomLevelRenderer.

    Args:
        state (WispState): A wisp state object, containing the scene graph information.
        name (str): Unique name of object added to the scene graph
        pipeline (Pipeline): A pipeline consisting of a field (and possibly a tracer) of the added object.
        transform (ObjectTransform): The object transform, containing a 4x4 transformation matrix which transforms
        the object from local object space to world space.
        setup_args (Dict): Optional setup args which control how the BottomLevelRenderer will be created.
    """
    if transform is None:
        transform = ObjectTransform(device=pipeline.nef.device)
    state.graph.neural_pipelines[name] = pipeline
    state.graph.bl_renderers[name] = BottomLevelRendererState(status='pending', transform=transform,
                                                              setup_args=setup_args)
    request_redraw(state)   # Let renderer core know it should refresh next frame


def add_to_scene_graph(state: WispState,
                       name: str,
                       obj,
                       transform: ObjectTransform = None,
                       **setup_args):
    """ Adds a new object to the scene graph.
    obj can be any supported object type, neural or non-neural.
    This is the most general function used to manage adding new objects to the scene graph.

    Args:
        state (WispState): A wisp state object, containing the scene graph information.
        name (str): Unique name of object added to the scene graph
        obj (object): Any object supported by the scene-graph.
        i.e: for neural fields, obj is a Pipeline.
        transform (ObjectTransform): The object transform, containing a 4x4 transformation matrix which transforms
        the object from local object space to world space.
        setup_args (Dict): Optional setup args which control how the BottomLevelRenderer will be created.
    """
    if isinstance(obj, Pipeline):
        add_pipeline_to_scene_graph(state, name, obj, transform, **setup_args)
    else:   # TODO (operel): Currently only neural pipelines are supported
        raise NotImplementedError(f'Unsupported object type added to scene graph: {obj}')


def remove_from_scene_graph(state: WispState, name: str):
    """ Removes an existing pipeline from the scene graph.

        Args:
            state (WispState): A wisp state object, containing the scene graph information.
            name (str): Unique name of object added to the scene graph
    """
    assert name in state.graph.neural_pipelines, f'Scene graph requested to remove non-existing object: {name}'
    del state.graph.neural_pipelines[name]
    del state.graph.bl_renderers[name]
    request_redraw(state)  # Let renderer core know it should refresh next frame
