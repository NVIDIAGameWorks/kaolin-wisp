# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from wisp.models import Pipeline
from wisp.framework import WispState


def request_redraw(state):
    """ Marks the canvas as dirty,
    forcing the renderer core to refresh the object renderers on the next rendering iteration.
    """
    state.renderer.canvas_dirty = True


def add_to_scene_graph(state: WispState, name: str, pipeline: Pipeline):
    """ Adds a new object pipeline to the scene graph.
    The pipeline contains all components required to trace this object.

    Args:
        state (WispState): A wisp state object, containing the scene graph information.
        name (str): Unique name of object added to the scene graph
        pipeline (Pipeline): A pipeline consisting of a field (and possibly a tracer) of the added object.
    """
    state.graph.neural_pipelines[name] = pipeline
    request_redraw(state)   # Let renderer core know it should refresh next frame


def remove_from_scene_graph(state: WispState, name: str):
    """ Removes an existing pipeline from the scene graph.

        Args:
            state (WispState): A wisp state object, containing the scene graph information.
            name (str): Unique name of object added to the scene graph
    """
    assert name in state.graph.neural_pipelines, f'Scene graph requested to remove non-existing object: {name}'
    del state.graph.neural_pipelines[name]
    request_redraw(state)  # Let renderer core know it should refresh next frame
