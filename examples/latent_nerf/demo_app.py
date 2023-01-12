# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Callable, Dict, List
from kaolin.render.camera import Camera
from wisp.framework import WispState, watch
from wisp.renderer.app import WispApp
from wisp.renderer.gui import WidgetImgui
from wisp.renderer.gizmos import Gizmo
from wisp.renderer.core.api import request_redraw


class DemoApp(WispApp):
    """ A demo app for optimizing an object with a latent channel.
    The app itself is responsible for the interactive visualization of the demo.
    That is, this app exposes the new latent channel to the interactive visualizer,
    allowing users to choose it from the gui and view it over the canvas.
    """

    def __init__(self,
                 wisp_state: WispState,
                 background_task: Callable[[], None] = None,
                 window_name: str = 'SIGGRAPH 2022 Demo'):
        super().__init__(wisp_state, window_name)

        # Tell the renderer to invoke a background task (i.e. a training iteration function)
        # in conjunction to rendering.
        # Power users: The background tasks are constantly invoked by glumpy within the on_idle() event.
        # The actual rendering will occur in-between these calls, invoked by the on_draw() event (which checks if
        # it's time to render the scene again).
        self.register_background_task(background_task)

        # from wisp.models.nefs import FunnyNeuralField
        # from wisp.tracer import DebuggablePackedRFTracer
        # from wisp.renderer.core.api import register_neural_field_type, NeuralRadianceFieldPackedRenderer
        # register_neural_field_type(neural_field_type=FunnyNeuralField,
        #                            tracer_type=DebuggablePackedRFTracer,
        #                            renderer_type=NeuralRadianceFieldPackedRenderer)


    ## --------------------------------------------------------------------
    ## ------------------------------ Setup -------------------------------
    ## --------------------------------------------------------------------

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
            This function is called before the WispApp is initialized, hence the initialized fields can
            be customized to affect the behaviour of the renderer.
        """
        # For convenience, we divide the initialization to sections:
        self._init_scene_graph(wisp_state)
        self._init_interactive_renderer_properties(wisp_state)

    def _init_scene_graph(self, wisp_state: WispState) -> None:
        """ -- wisp_state.graph holds the scene graph information: -- """
        # Define which channels can be traced by this wisp app, and how they behave.
        # The starter kit has some predefined channels (RGB, depth, and more..).
        # Wisp allows the definition of custom channels by augmenting or replacing this dictionary.
        # See wisp.core.channel_fn for blending and normalization functions
        from wisp.core import channels_starter_kit, Channel, blend_normal, normalize
        wisp_state.graph.channels = channels_starter_kit()
        wisp_state.graph.channels['color_feature'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                             normalize_fn=normalize,  # Map to [0.0, 1.0]
                                                             min_val=0, max_val=1.0)

    def _init_interactive_renderer_properties(self, wisp_state: WispState) -> None:
        """ -- wisp_state.renderer holds the interactive renderer configuration, let's explore it: -- """

        # Set the initial window dimensions
        wisp_state.renderer.canvas_width = 1200
        wisp_state.renderer.canvas_height = 800

        # Set which world grid planes should be displayed on the canvas.
        # Options: any combination of 'xy', 'xz', 'yz'. Use [] to turn off the grid.
        wisp_state.renderer.reference_grids = ['xz']

        # Decide which channels can be displayed over the canvas (channel names are NOT case sensitive).
        # See also wisp_state.graph.channels and wisp.core.channels.channels_starter_kit for configuring channels.
        # Options: Any subset of channel names defined in wisp_state.graph.channels
        wisp_state.renderer.available_canvas_channels = ["RGB", "Depth", "Color_Feature"]
        wisp_state.renderer.selected_canvas_channel = "RGB"  # Channel selected by default

        # Lens mode for camera used to view the canvas.
        # Choices: 'perspective', 'orthographic'
        wisp_state.renderer.selected_camera_lens = 'perspective'

        # Set the canvas background color (RGB)
        wisp_state.renderer.clear_color_value = (0.0, 0.0, 0.0)

    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
        Possible choices: 'First Person View', 'Turntable', 'Trackball'
        """
        return "Turntable"

    def create_widgets(self) -> List[WidgetImgui]:
        """ Customizes the gui: Defines which widgets the gui will display, in order. """
        from wisp.renderer.gui import WidgetInteractiveVisualizerProperties, WidgetGPUStats, WidgetSceneGraph, WidgetOptimization
        widgets = [WidgetGPUStats(),            # Current FPS, memory occupancy, GPU Model
                   WidgetOptimization(),        # Live status of optimization, epochs / iterations count, loss curve
                   WidgetInteractiveVisualizerProperties(),  # Canvas dims, user camera controller & definitions
                   WidgetSceneGraph()]          # A scene graph tree with the objects hierarchy and their properties

        return widgets

    ## --------------------------------------------------------------------
    ## ---------------------------- App Events ----------------------------
    ## --------------------------------------------------------------------

    def register_event_handlers(self) -> None:
        """ Register event handlers for various events that occur in a wisp app.
            For example, the renderer is able to listen on changes to fields of WispState objects.
            (note: events will not prompt when iterables like lists, dicts and tensors are internally updated!)
        """
        # Register default events, such as updating the renderer camera controller when the wisp state field changes
        super().register_event_handlers()

        # For this app, we define a custom event that prompts when an optimization epoch is done,
        # or when the optimization is paused / unpaused
        watch(watched_obj=self.wisp_state.optimization, field="epoch", status="changed", handler=self.on_epoch_ended)
        watch(watched_obj=self.wisp_state.optimization, field="running", status="changed",
              handler=self.on_optimization_running_changed)

    def on_epoch_ended(self):
        """ A custom event defined for this app.
            When an epoch ends, this handler is invoked to force a redraw() and render() of the canvas if needed.
        """
        # Request a redraw from the renderer core.
        # redraw() will:
        # - Refresh the scene graph (new objects added will be created within the renderer-core if needed)
        # - Data layers will regenerate according to up-to-date state.
        request_redraw(self.wisp_state)

        # Request a render if:
        # 1. Too much time have elapsed since the last frame
        # 2. Target FPS is 0 (rendering loop is stalled and the renderer only renders when explicitly requested)
        # render() ensures the most up to date neural field is displayed.
        if self.is_time_to_render() or self.wisp_state.renderer.target_fps == 0:
            self.render()

    def on_optimization_running_changed(self, value: bool):
        # When training starts / resumes, invoke a redraw() to refresh the renderer core with newly
        # added objects to the scene graph (i.e. the optimized object, or some objects from the dataset).
        if value:
            self.redraw()
