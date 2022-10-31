# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from __future__ import annotations
import torch
from typing import List, Dict, Type, DefaultDict, Tuple, TYPE_CHECKING
from collections import defaultdict
from dataclasses import dataclass, field
from kaolin.render.camera import Camera
from wisp.framework.event import watchedfields
from wisp.core import Channel, PrimitivesPack, ObjectTransform, channels_starter_kit
if TYPE_CHECKING:  # Prevent circular imports mess due to typed annotations
    from wisp.models import Pipeline
    from wisp.renderer.core.control import CameraControlMode
    from wisp.renderer.core import BottomLevelRenderer

"""
The WispState object is a shared resource between the various wisp components,
and is used to keep the various modules decoupled and synced on transient information, current configurations, and
state of the scene.

Wisp components share information and sync through the WispState subcomponents and their fields:
1. Wisp Components may write to certain fields directly, which are periodically read by other components.
   For example: The interactive renderer reads the InteractiveRendererState.canvas_dirty to determine when
   the canvas should be redrawn. Other components may request a redraw by setting this flag.
2. The @watchedfields decorator prompts an event when the state field values changes. 
   Components may register event handlers to immediately respond on such changes by using wisp.framework.event.watch
   For example: The interactive renderer listens on the OptimizationState.epoch field to redraw the canvas
   when an epoch ends.
   Note: Iterable fields will not respond to objects being added / removed.
   
Wisp applications may extend the state object via the WispState.extent field.
"""

@watchedfields
@dataclass
class InteractiveRendererState:
    """
    Holds settings directly used by the interactive renderer of wisp apps.
    """

    fps: float = 0.0
    """ FPS measured by renderer clocked """

    target_fps: float = None
    """ Ideal FPS the renderer should attempt to be on par with.
        If None, the ideal FPS is unbounded.
    """

    dt: float = 0.0
    """ Time elapsed since last renderer clock tick """

    canvas_width: int = 1600
    """ Default canvas width """

    canvas_height: int = 1200
    """ Default canvas height """

    clear_color_value: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """ Clear value for canvas background color channel, in RGB format """

    clear_depth_value: float = 1.0
    """ Clear value for canvas background depth channel """

    available_canvas_channels: List[str] = field(default_factory=list)
    """ List of available output channels to view on the canvas.
        This field should be set by the app, and should be compatible with the objects expected to be traced.
    """

    selected_canvas_channel: str = 'rgb'
    """ Currently selected output channel to view on the canvas.
        Choices: any of the channels listed in available_canvas_channels. 
    """

    canvas_dirty = False
    """ Setting this flag forces the renderer to redraw the canvas on the next frame. """

    cam_controller: Type[CameraControlMode] = None
    """ The current camera controller which manipulates the canvas camera.
        Choices: First Person, Trackball or Turntable. 
    """

    selected_camera: Camera = None
    """ The camera currently used by renderer to view the scene.
        If None, the renderer will create a default camera upon demand.
    """

    selected_camera_lens: str = 'perspective'
    """ Projection mode for selected camera used to view the canvas.
        Choices: 'perspective', 'orthographic'
    """

    interactive_mode: bool = False
    """ Whether the renderer is currently set in interactive mode: resolution may be adjusted to maintain FPS.
    """

    background_tasks_paused: bool = True
    """ When true, background tasks (i.e. optimizations) are paused. """

    antialiasing: str = "msaa_4x"
    """ Antialising method to use for the canvas.
        Due to the universal nature of the renderer, this setting may not affect object renderers, but
        mostly primitives drawn directly on the canvas with the graphics api
        (such as, i.e. vectorial layers drawn with OpenGL).
        Choices:
        'msaa_4x' - Framebuffer will be generated with 4x multisampling. 
                    More fragments will be generated to eliminate jagged borders, at the cost of additional computation. 
        'none' - No inherent antialising mode will be activated.
    """

    reference_grids: List[str] = field(default_factory=list)
    """ List of world grids to use as reference planes, for both rendering and some camera controllers.
        Choices: Any combination of the values: 'xy', 'xz', 'yz'. An empty list will turn off the grid.
    """

    device: torch.device = 'cpu'
    """ Default device for interactive renderer and bottom level renderers to use """


@watchedfields
@dataclass
class BottomLevelRendererState:
    """
    Holds the settings of a single Bottom-Level renderer.
    Wisp supports joint rendering of various pipelines (NeRF, SDFs, meshes, and so forth),
    where each pipeline is wrapped by a bottom level renderer configured by this state object.
    The state object exists throughout the lifecycle of the renderer / pipeline,
    and is used to determine how the bottom level renderer of an existing pipeline is constructed.
    """

    renderer: BottomLevelRenderer = None
    """ The instance of the bottom level renderer. Initialized only if the status field is 'loaded' """

    data_layers: Dict[str, PrimitivesPack] = field(default_factory=dict)  # layer id -> prims
    toggled_data_layers: Dict[str, bool] = field(default_factory=dict)    # layer id -> bool

    transform: ObjectTransform = ObjectTransform()
    """ The object transform maintains the 4x4 model matrix, which describes the object transformation from
        local object coordinates to world space.
        Manipulating the transform results in moving, scaling and orienting the object.
    """

    ### Lifecycle fields ###

    status: str = 'pending'
    """ The current status of this renderer instance:
        'pending' - the renderer is part of the scene graph but was not initialized yet.
        'loaded' - the renderer is part of the scene graph, and the instance was initialized (renderer field exists).
        'ignored' - this renderer should be excluded from the scene graph.  
    """

    setup_args: Dict[str, object] = field(default_factory=dict)
    """ Optional args, used to load or construct the renderer instance """


@watchedfields
@dataclass
class SceneGraphState:
    """
    Holds the entire scene graph of objects.
    """
    neural_pipelines: Dict[str, Pipeline] = field(default_factory=dict)
    """ Wisp objects are represented by neural pipelines which pair a BaseNeuralField and a BaseTracer.
        A BottomLevelRenderer knows how to invoke the pipeline to render the nef and obtain a 
        Renderbuffer of the object. 
    """

    bl_renderers: Dict[str, BottomLevelRendererState] = field(default_factory=dict)
    """ Renderable wisp objects are represented by BottomLevelRenderers which wrap neural pipelines.
        A BottomLevelRenderer knows how to invoke the rendering pipeline of some object type and obtain
        a Renderbuffer of the object. For example, a BottomLevelRenderer may wrap a NeRF object and its RFTracer.
    """

    cameras: Dict[str, Camera] = field(default_factory=dict)
    """ Set of cameras currently available for this scene. """

    visible_objects: Dict[str, bool] = field(default_factory=dict)
    """ Which objects are currently toggled on as visible in the scene.
        This mapping includes all object types: bl_renderers, cameras, etc.
    """

    channels: Dict[str, Channel] = field(default_factory=channels_starter_kit)
    """ The channels kit containing information per channel, such as blending function, normalization function, etc.
    """

@watchedfields
@dataclass
class OptimizationState:
    """
    Holds information about an optimization task wisp is currently running.
    This is used to cache information in trainers and share them with other components like the renderer.
    """

    running: bool = False
    """ Is the optimization currently running or paused / stopped. """

    validation_enabled: bool = False
    """ If true, validation will run every N epochs. """

    epoch: int = 0
    """ Current epoch of the optimization. An epoch is task / dataset dependent, but is normally
        equivalent to a single pass over a finite dataset.
    """

    iteration: int = 0
    """ Current iteration within an epoch. An iteration is task / dataset dependent, but usually refers
        to a step of the trainer on a single batch.
    """

    max_epochs: int = 0
    """ Total number of epochs set for this optimization task. """

    iterations_per_epoch: int = 0
    """ Total iterations used within each epoch / the current epoch. """

    elapsed_time: float = 0
    """ Total time elapsed for running the optimization. Measured in seconds. """

    lr: float = 0.0
    """ The learning rate currently set by the optimizaer. """

    losses: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    """ Losses currently reported by this optimization task (name to list of values per epoch) """

    metrics: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(float))
    """ Metrics currently reported by this optimization task (name to list of values per epoch) """

    train_data: List[torch.utils.data.Dataset] = field(default_factory=list)
    """ List of training sets employed by this optimization task """

    validation_data: List[torch.utils.data.Dataset] = field(default_factory=list)
    """ List of validation sets employed by this optimization task """


@dataclass
class WispState:
    """ Holds all information shared between components of a single wisp app. """

    renderer: InteractiveRendererState = InteractiveRendererState()
    """ Holds settings directly used by the interactive renderer of wisp apps. """

    graph: SceneGraphState = SceneGraphState()
    """ Holds the entire scene graph of objects. """

    optimization: OptimizationState = OptimizationState()
    """ Holds information about an optimization task wisp is currently running. """

    extent: Dict[str, object] = field(default_factory=dict)
    """ Extensible field: custom wisp applications information can be added here. """
