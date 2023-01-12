# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Callable, Dict, List
from wisp.renderer.gui import WidgetImgui
from wisp.renderer.gui import WidgetInteractiveVisualizerProperties, WidgetGPUStats, WidgetSceneGraph, WidgetOptimization
from wisp.renderer.gizmos.gizmo import Gizmo
from wisp.renderer.app.wisp_app import WispApp
from wisp.renderer.core.api import request_redraw
from wisp.framework import WispState, watch
from wisp.datasets import MultiviewDataset, SDFDataset


class OptimizationApp(WispApp):
    """ An app for running an optimization and visualizing it's progress interactively in real time. """

    def __init__(self, wisp_state: WispState, trainer_step_func: Callable[[], None], experiment_name: str):
        super().__init__(wisp_state, experiment_name)

        # Tell the renderer to invoke the optimization step() function on every background iteration.
        # The background tasks are constantly invoked by glumpy within the on_idle() event.
        # The actual rendering will occur in-between these calls, invoked by the on_draw() event (which checks if
        # it's time to render the scene again).
        self.register_background_task(trainer_step_func)

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
        This function is called at the very beginning of WispApp initialization, hence the initialized fields can
        be customized to affect the behaviour of the renderer.
        """
        # Channels available to view over the canvas
        wisp_state.renderer.available_canvas_channels = ["rgb", "depth"]
        wisp_state.renderer.selected_canvas_channel = "rgb"

        # For this app, we'll use only a world grid which resides in the 'xz' plane.
        # This attribute will signal the renderer that some camera controllers should align with this plane,
        # as well as what gizmos (e.g: reference grids, axes) to draw on the canvas.
        wisp_state.renderer.reference_grids = ['xz']

        # MultiviewDatasets come from images with a predefined background color.
        # The following lines can be uncommented to initialize the renderer canvas background color
        # to the train data bg color if it is black or white.
        #
        # train_sets = self.wisp_state.optimization.train_data
        # if train_sets is not None and len(train_sets) > 0:
        #     train_set = train_sets[0]
        #     if isinstance(train_set, MultiviewDataset):
        #         if train_set.bg_color == 'white':
        #             wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)
        #         elif train_set.bg_color == 'black':
        #             wisp_state.renderer.clear_color_value = (0.0, 0.0, 0.0)

    def create_widgets(self) -> List[WidgetImgui]:
        """ Returns the list of widgets the gui will display, in order. """
        return [WidgetGPUStats(),            # Current FPS, memory occupancy, GPU Model
                WidgetOptimization(),        # Live status of optimization, epochs / iterations count, loss curve
                WidgetInteractiveVisualizerProperties(),  # Canvas dims, user camera controller & definitions
                WidgetSceneGraph()]          # A scene graph tree of the entire hierarchy of objects in the scene
                                             # and their properties

    def create_gizmos(self) -> Dict[str, Gizmo]:
        """ Override to control which gizmos appear on the canvas.
            For example:
            gizmos = dict(
                world_grid_xy=WorldGrid(squares_per_axis=20, grid_size=10,
                                        line_color=(128, 128, 128), line_size=1, plane="xy")
                grid_axis_painter=AxisPainter(axes_length=10, line_width=1, axes=('x', 'y'), is_bidirectional=True)
            )
        """
        return super().create_gizmos()

    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
            Possible choices: 'First Person View', 'Turntable', 'Trackball' """
        return "Turntable"

    def register_event_handlers(self) -> None:
        """ Register event handlers for various events that occur in a wisp app.
            For example, the renderer is able to listen on updates on fields of WispState objects.
            (note: events will not prompt when iterables like lists, dicts and tensors are internally updated!)
        """
        # Register default events, such as updating the renderer camera controller when the wisp state field changes
        super().register_event_handlers()

        # For this app, we define a custom event that prompts when an optimization epoch is done
        watch(watched_obj=self.wisp_state.optimization, field="epoch", status="changed", handler=self.on_epoch_ended)
        watch(watched_obj=self.wisp_state.optimization, field="running", status="changed",
              handler=self.on_optimization_running_changed)

    def update_renderer_state(self, wisp_state, dt) -> None:
        """
        Populate the scene state object with the most recent information about the interactive renderer.
        The scene state, for example, may be used by the GUI widgets to display up to date information.
        This function is invoked in the beginning of the render() function, before the gui and the canvas are drawn.
        :param wisp_state The WispState object holding shared information between components about the wisp app.
        :param dt Amount of time elapsed since the last update.
        """
        # Update the wisp state with new information about this frame.
        # i.e.: Current FPS, time elapsed.
        super().update_renderer_state(wisp_state, dt)

    def on_epoch_ended(self):
        """ A custom event used by the optimization renderer.
            When an epoch ends, this handler is invoked to force a redraw() and render() of the canvas if needed.
        """
        request_redraw(self.wisp_state)

        # Force render if target FPS is 0 (renderer only responds to specific events) or too much time have elapsed
        if self.is_time_to_render() or self.wisp_state.renderer.target_fps == 0:
            self.render()

    def on_optimization_running_changed(self, value: bool):
        # When training starts / resumes, invoke a redraw() to refresh the renderer core with newly
        # added objects to the scene graph (i.e. the optimized object, or some objects from the dataset).
        if value:
            self.redraw()
