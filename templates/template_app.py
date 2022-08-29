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


class TemplateApp(WispApp):
    """ An exemplary template for quick creation of new user interactive apps.
    Clone this file and modify the fields to create your own customized wisp app.
    """

    def __init__(self,
                 wisp_state: WispState,
                 background_task: Callable[[], None] = None,
                 window_name: str = 'YourWindowTitleHere'):
        super().__init__(wisp_state, window_name)

        # Tell the renderer to invoke a background task (i.e. a training iteration function)
        # in conjunction to rendering.
        # Power users: The background tasks are constantly invoked by glumpy within the on_idle() event.
        # The actual rendering will occur in-between these calls, invoked by the on_draw() event (which checks if
        # it's time to render the scene again).
        self.register_background_task(background_task)

    ## --------------------------------------------------------------------
    ## ------------------------------ Setup -------------------------------
    ## --------------------------------------------------------------------

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
        This function is called at the very beginning of WispApp initialization, hence the initialized fields can
        be customized to affect the behaviour of the renderer.
        """
        # For convenience, we divide the initialization to sections:
        self._init_scene_graph(wisp_state)
        self._init_interactive_renderer_properties(wisp_state)
        self._init_user_state(wisp_state)

    def _init_scene_graph(self, wisp_state: WispState) -> None:
        """ -- wisp_state.graph holds the scene graph information: -- """
        # Define which channels can be traced by this wisp app, and how they behave.
        # The starter kit has some predefined channels (RGB, depth, and more..).
        # Wisp allows the definition of custom channels by augmenting or replacing this dictionary.
        # See wisp.core.channel_fn for blending and normalization functions
        from wisp.core import channels_starter_kit, Channel, blend_normal, normalize
        wisp_state.graph.channels = channels_starter_kit()
        # For example, this is how a latent channel within the range [-1, 1] could be set
        # Your implementation of BaseNeuralField and BaseTracer should support such channels, otherwise
        # they'll appear empty when you visualize them.
        wisp_state.graph.channels['my_latent'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                         normalize_fn=normalize,  # Map to [0.0, 1.0]
                                                         min_val=-1.0, max_val=1.0)

        # Here you may populate the scene graph with pre loaded objects.
        # When optimizing an object, there is no need to explicitly add it here as the Trainers already
        # adds it to the scene graph.
        # from wisp.renderer.core.api import add_object_to_scene_graph
        # add_object_to_scene_graph(state=wisp_state, name='New Object', pipeline=Pipeline(nef=..., tracer=...))

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
        wisp_state.renderer.available_canvas_channels = ["rgb", "depth", "my_latent"]
        wisp_state.renderer.selected_canvas_channel = "rgb"  # Channel selected by default

        # Lens mode for camera used to view the canvas.
        # Choices: 'perspective', 'orthographic'
        wisp_state.renderer.selected_camera_lens = 'perspective'

        # Set the canvas background color (RGB)
        wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)

        # For optimization apps -
        # Some MultiviewDatasets come from images with a predefined background color.
        # The following lines can be uncommented to initialize the renderer canvas background color
        # to the train data bg color if it is black or white.
        #
        # from wisp.datasets import MultiviewDataset, SDFDataset
        # train_sets = self.wisp_state.optimization.train_data
        # if train_sets is not None and len(train_sets) > 0:
        #     train_set = train_sets[0]  # If multiple datasets are available, use the first one
        #     if isinstance(train_set, MultiviewDataset):
        #         if train_set.bg_color == 'white':
        #             wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)
        #         elif train_set.bg_color == 'black':
        #             wisp_state.renderer.clear_color_value = (0.0, 0.0, 0.0)

    def _init_user_state(self, wisp_state: WispState) -> None:
        """ -- wisp_state.extent allows users to store whatever custom information they want to share -- """

        # For example: let's add a frame counter which increments every time a frame is rendered.
        user_state = wisp_state.extent
        user_state['frame_counter'] = 0

    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
        Possible choices: 'First Person View', 'Turntable', 'Trackball'
        """
        return "Turntable"

    def create_widgets(self) -> List[WidgetImgui]:
        """ Customizes the gui: Returns which widgets the gui will display, in order. """
        from wisp.renderer.gui import WidgetRendererProperties, WidgetGPUStats, WidgetSceneGraph, WidgetOptimization
        widgets = [WidgetGPUStats(),            # Current FPS, memory occupancy, GPU Model
                   WidgetOptimization(),        # Live status of optimization, epochs / iterations count, loss curve
                   WidgetRendererProperties(),  # Canvas dims, user camera controller & definitions
                   WidgetSceneGraph()]          # A scene graph tree with the objects hierarchy and their properties

        # Create new types of widgets with imgui by implementing the following interface:
        class WidgetCounter(WidgetImgui):
            def paint(self, wisp_state: WispState, *args, **kwargs):
                import imgui
                number_of_neural_objects = len(wisp_state.graph.neural_pipelines)
                frames_counter = wisp_state.extent['frame_counter']
                imgui.text(f'The scene graph has {number_of_neural_objects} neural objects.')
                imgui.text(f'The app rendered {frames_counter} frames so far.')
        widgets.insert(1, WidgetCounter())

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
        self.canvas_dirty = True    # Request a redraw from the renderer core

        # Request a render if:
        # 1. Too much time have elapsed since the last frame
        # 2. Target FPS is 0 (rendering loop is stalled and the renderer only renders when explicitly requested)
        if self.is_time_to_render() or self.wisp_state.renderer.target_fps == 0:
            self.render()

    def on_optimization_running_changed(self, value: bool):
        # When training starts / resumes, invoke a redraw() to refresh the renderer core with newly
        # added objects to the scene graph (i.e. the optimized object, or some objects from the dataset).
        if value:
            self.redraw()

    ## --------------------------------------------------------------------
    ## -------------------------- Advanced usage --------------------------
    ## --------------------------------------------------------------------

    # Implement the following functions for even more control

    def create_gizmos(self) -> Dict[str, Gizmo]:
        """ Gizmos are transient rasterized objects rendered by OpenGL on top of the canvas.
        For example: world grid, axes painter.
        Here you may add new types of gizmos to paint over the canvas.
        """
        gizmos = super().create_gizmos()

        # Use glumpy and OpenGL to paint over the canvas
        # For brevity, a custom gizmo implementation is omitted here,
        # see wisp.renderer.gizmos.ogl.axis_painter for a working example

        from glumpy import gloo, gl
        class CustomGizom(Gizmo):
            def render(self, camera: Camera):
                """ Renders the gizmo using the graphics api. """
                pass

            def destroy(self):
                """ Release GL resources, must be called from the rendering thread which owns the GL context """
                pass

        gizmos['my_custom_gizmo'] = CustomGizom()
        return gizmos

    def update_renderer_state(self, wisp_state, dt) -> None:
        """
        Called every time the rendering loop iterates.
        This function is invoked in the beginning of the render() function, before the gui and the canvas are drawn.
        Here you may populate the scene state object with information that updates per frame.
        The scene state, for example, may be used by the GUI widgets to display up to date information.
        :param wisp_state The WispState object holding shared information between components about the wisp app.
        :param dt Amount of time elapsed since the last update.
        """
        # Update the default wisp state with new information about this frame.
        # i.e.: Current FPS, time elapsed.
        super().update_renderer_state(wisp_state, dt)
        wisp_state.extent['frame_counter'] += 1
