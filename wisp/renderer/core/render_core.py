# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import abc
import numpy as np
import torch
import copy
from collections import defaultdict
from typing import Dict, List, Iterable
from kaolin.render.camera import Camera, PinholeIntrinsics, OrthographicIntrinsics
from wisp.framework import WispState, BottomLevelRendererState
from wisp.core import RenderBuffer, Rays, PrimitivesPack, create_default_channel
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.renderer.core.api import BottomLevelRenderer, RayTracedRenderer, create_neural_field_renderer
from wisp.renderer.core.api import FramePayload
from wisp.gfx.datalayers import CameraDatalayers


class RendererCore:
    def __init__(self, state: WispState):
        self.state = state
        self.device = state.renderer.device

        # Create a camera for user view
        self.camera = self._setup_camera(state)
        self._camera_layers = CameraDatalayers()

        # Set up the list of available bottom level object renderers, according to scene graph
        self._renderers = None
        self._tlas = None
        self.refresh_bl_renderers(state)

        self.res_x, self.res_y = None, None
        self.set_full_resolution()

        self._last_state = dict()
        self._last_renderbuffer = None

        # Minimal resolution supported by RendererCore
        self.MIN_RES = 128

    def _default_camera(self, lens="perspective"):
        # TODO: move defaults elsewhere
        if lens == 'perspective':
            return Camera.from_args(
                eye=torch.tensor([4.0, 4.0, 4.0]),
                at=torch.tensor([0.0, 0.0, 0.0]),
                up=torch.tensor([0.0, 1.0, 0.0]),
                fov=30 * np.pi / 180,  # In radians
                x0=0.0, y0=0.0,
                width=800, height=800,
                near=1e-2, far=20,
                dtype=torch.float64,
                device=self.device
            )
        elif lens == 'orthographic':
            return Camera.from_args(
                eye=torch.tensor([4.0, 4.0, 4.0]),
                at=torch.tensor([0.0, 0.0, 0.0]),
                up=torch.tensor([0.0, 1.0, 0.0]),
                width=800, height=800,
                near=-800, far=800,
                fov_distance=1.0,
                dtype=torch.float64,
                device=self.device
            )

    def change_camera_projection_mode(self, lens: str) -> None:
        # Need to update only the intrinsic component of the camera
        # TODO (operel): in the future, kaolin should implement a capability to switch lens and approximate params
        # from one type to another where applicable
        if lens == 'Perspective':
            intrinsics = PinholeIntrinsics.from_fov(
                fov=30 * np.pi / 180,       # In radians
                x0=0.0, y0=0.0,
                width=self.camera.width, height=self.camera.height,
                near=1e-2, far=1e2,
                dtype=self.camera.dtype,
                device=self.camera.device
            )
        elif lens == 'Orthographic':
            intrinsics = OrthographicIntrinsics.from_frustum(
                width=self.camera.width, height=self.camera.height,
                near=-self.camera.width, far=self.camera.height,
                fov_distance=1.0,
                dtype=self.camera.dtype,
                device=self.camera.device
            )
        else:
            raise ValueError(f"Unknown lens type: {lens} given to render_core.change_camera_projection_mode")
        self.camera.intrinsics = intrinsics
        del self._last_state['camera']  # The previous camera is now obsolete

    def _setup_camera(self, state: WispState):
        # Use selected camera to control canvas
        camera = state.renderer.selected_camera
        if camera is None:
            # Create a default camera
            lens_type = self.state.renderer.selected_camera_lens
            camera = self._default_camera(lens_type)

        camera = camera.to(self.device)
        return camera

    def refresh_bl_renderers(self, state: WispState) -> None:
        renderers = dict()
        scene_graph = state.graph

        # Remove obsolete bottom level renderers for pipelines that no longer exist
        for obj_name in list(scene_graph.bl_renderers.keys()):
            if obj_name not in scene_graph.neural_pipelines:
                del scene_graph.bl_renderers[obj_name]

        # Set up a renderer for all neural pipelines in the scene
        for renderer_id, neural_pipeline in scene_graph.neural_pipelines.items():
            # See if a descriptor for the renderer already exists.
            bl_state = scene_graph.bl_renderers.get(renderer_id)
            if bl_state is None:
                # If not, create a default one
                bl_state = BottomLevelRendererState(status='pending', setup_args=dict())
                scene_graph.bl_renderers[renderer_id] = bl_state

            if bl_state.status == 'loaded':
                assert bl_state.renderer is not None, \
                       f'status of renderer {renderer_id} shows it was loaded, but renderer instance is None.'
                renderers[renderer_id] = bl_state.renderer
            elif bl_state.status == 'pending':
                bl_state.renderer = create_neural_field_renderer(neural_object=neural_pipeline, **bl_state.setup_args)
                bl_state.status = 'loaded'
                renderers[renderer_id] = bl_state.renderer
                scene_graph.visible_objects[renderer_id] = True  # Mark as visible when first loaded
            elif bl_state.status == 'ignored':
                pass    # Skip renderer on purpose
            else:
                raise ValueError(f'Invalid bottom level renderer state: {bl_state.status}')
        self._renderers = renderers

        # Refresh TLAS
        self._tlas = self._setup_tlas(state)

    def _setup_tlas(self, state: WispState):
        # Currently the top-level acceleration structure uses a straightforward ordered list stub
        return ListTLAS(self._renderers)

    def set_full_resolution(self):
        self.res_x = self.camera.width
        self.res_y = self.camera.height
        self.interactive_mode = False

    def set_low_resolution(self):
        self.res_x = self.camera.width // 4
        self.res_y = self.camera.height // 4
        self.interactive_mode = True

    def resize_canvas(self, width, height):
        self.camera.intrinsics.width = width
        self.camera.intrinsics.height = height
        self.set_low_resolution()

    def _update_scene_graph(self):
        """ Update scene graph information about objects and their data layers """
        # New data layers maybe have been generated, update the scene graph about their existence.
        # Some existing layers may have been regenerated, in which case we copy their previous "toggled on/off" status.
        data_layers = self._bl_renderer_data_layers()
        for object_id, bl_renderer_state in self.state.graph.bl_renderers.items():
            object_layers = data_layers[object_id]
            bl_renderer_state.data_layers = object_layers
            toggled_data_layers = defaultdict(bool)
            for layer_name, layer in object_layers.items():  # Copy over previous toggles if such exist
                toggled_data_layers[layer_name] = bl_renderer_state.toggled_data_layers.get(layer_name, False)
            bl_renderer_state.toggled_data_layers = toggled_data_layers

    @torch.no_grad()
    def redraw(self) -> None:
        """ Allow bottom-level renderers to refresh internal information, such as data layers. """
        # Read phase: sync with scene graph, create renderers for new objects added
        self.refresh_bl_renderers(self.state)

        # Invoke internal redraw() on all current visible renderers, to imply it's time to refresh content
        # (i.e. data layers may get generate here behind the scenes)
        scene_graph = self.state.graph
        for obj_id, renderer in self._renderers.items():
            if obj_id in scene_graph.visible_objects:
                renderer.redraw()

        # Write phase: update scene graph back with latest info from render core, i.e. new data layers generated
        self._update_scene_graph()

    @torch.no_grad()
    def render(self, time_delta=None, force_render=False) -> RenderBuffer:
        """Render a frame.

        Args:
            time_delta (float): The time delta from the previous frame, used to control renderer parameters
                                based on the amount of detected lag. 
            force_render (bool): If True, will always output a fresh new RenderBuffer. 
                                 Otherwise the RenderBuffer can be a stale copy of the the previous frame
                                 if no updates are detected.

        Returns: 
            (wisp.core.RenderBuffer): The rendered buffer.
        """
        payload = self._prepare_payload(time_delta)
        rb = self.render_payload(payload, force_render)
        output_rb = self._post_render(payload, rb)
        return output_rb

    def _prepare_payload(self, time_delta=None) -> FramePayload:
        """This function will prepare the FramePayload for the current frame.

        The FramePayload contains metadata for the current frame, from which the RenderBuffer will be 
        generated from. 

        Args:
            time_delta (float): The time delta from the previous frame, used to control renderer parameters
                                based on the amount of detected lag. 

        Returns:
            (wisp.renderer.core.api.FramePayload): The metadata for the frame.
        """
        # Adjust resolution of all renderers to maintain FPS
        camera = self.camera
        clear_color = self.state.renderer.clear_color_value
        res_x, res_y = self.res_x, self.res_y

        # If the FPS is slow, downscale the resolution for the render.
        is_fps_lagging = time_delta is not None and \
                         self.target_fps is not None and \
                         self.target_fps > 0 and \
                         time_delta > (1.0 / self.target_fps)
        if self.interactive_mode and is_fps_lagging:
            if res_x > self.MIN_RES and res_y > self.MIN_RES:
               res_x //= 2
               res_y //= 2

        # TODO(ttakikawa): Leaving a note here to think about whether this should be the case...
        # The renderer always needs depth, alpha, and rgb
        required_channels = {"rgb", "depth", "alpha"}
        selected_canvas_channel = self.state.renderer.selected_canvas_channel.lower()
        visible_objects = set([k for k,v in self.state.graph.visible_objects.items() if v])
        payload = FramePayload(camera=camera, interactive_mode=self.interactive_mode,
                               render_res_x=res_x, render_res_y=res_y, time_delta=time_delta,
                               visible_objects=visible_objects, clear_color=clear_color,
                               channels={selected_canvas_channel}.union(required_channels))
        for renderer_id, renderer in self._renderers.items():
            if renderer_id in payload.visible_objects:
                renderer.pre_render(payload)

        return payload

    def raygen(self, camera, res_x, res_y):
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, res_x, res_y, device=self.device)
        if camera.lens_type == 'pinhole':
            rays = generate_pinhole_rays(camera, ray_grid)
        elif camera.lens_type == 'ortho':
            rays = generate_ortho_rays(camera, ray_grid)
        else:
            raise ValueError(f'RendererCore failed to raygen on unknown camera lens type: {camera.lens_type}')
        return rays

    def _create_empty_rb(self, height, width, dtype=torch.float32) -> RenderBuffer:
        clear_color = self.state.renderer.clear_color_value
        clear_depth = self.state.renderer.clear_depth_value

        return RenderBuffer(
            rgb=torch.tensor(clear_color, dtype=dtype, device=self.device).repeat(height, width, 1),
            alpha=torch.zeros((height, width, 1), dtype=dtype, device=self.device),
            depth=torch.full((height, width, 1), fill_value=clear_depth, dtype=dtype, device=self.device),
            hit=None
        )

    def render_payload(self, payload: FramePayload, force_render: bool) -> RenderBuffer:
        """Renders a RenderBuffer using a FramePayload which contains metadata.

        Args:
            payload (wisp.renderer.core.api.FramePayload): Metadata for the frame to be renderered.
            force_render (bool): If True, will always output a fresh new RenderBuffer. 
                                 Otherwise the RenderBuffer can be a stale copy of the the previous frame
                                 if no updates are detected.
        
        Returns:
            (wisp.core.RenderBuffer): The rendered buffer.
        """
        camera = payload.camera
        res_x, res_y = payload.render_res_x, payload.render_res_y

        visible_renderers = [r for r_id, r in self._renderers.items() if r_id in payload.visible_objects]
        renderers_to_refresh = list(filter(lambda renderer: renderer.needs_refresh(payload), visible_renderers))
        if not self.needs_refresh() and len(renderers_to_refresh) == 0 and not force_render:
            return self._last_renderbuffer  # No need to regenerate..

        # Generate rays
        rays = self.raygen(camera, res_x, res_y)
        renderer_to_hit_rays = self._tlas.traverse(rays, payload)
        renderers_in_view = renderer_to_hit_rays.keys()

        rb_dtype = torch.float32
        clear_depth = self.state.renderer.clear_depth_value

        out_rb = self._create_empty_rb(height=camera.height, width=camera.width, dtype=rb_dtype)
        for renderer in renderers_in_view:
            if isinstance(renderer, RayTracedRenderer):
                in_rays = rays.to(device=renderer.device, dtype=renderer.dtype)
                rb = renderer.render(in_rays)
            else:   # RasterizedRenderer
                in_cam = self.camera.to(device=renderer.device, dtype=renderer.dtype)
                rb = renderer.render(in_cam)

            rb = rb.to(device=self.device)
            rb.rgb = rb.rgb.to(dtype=rb_dtype)
            
            rb.alpha = rb.alpha.to(dtype=rb_dtype)
            rb.depth = rb.depth.to(dtype=rb_dtype)

            # TODO (operel): if rb.depth is None -> painters algorithm
            # Normalize ray-traced depth buffer to graphics api range
            img_dims = rb.depth.shape

            # Clip depth values which fall outside of the view frustum
            clip_mask = camera.clip_mask(rb.depth.squeeze(-1))
            rb.alpha[~clip_mask] = 0.0

            # Normalize depth from [0, inf] to NDC space according to camera settings
            # (depends on near / far and NDC space)
            ndc_depth = camera.normalize_depth(rb.depth.reshape(-1, 1))
            rb.depth = ndc_depth.reshape(img_dims)

            # Set depth of missed rays to far clipping plane, as PackedRFTracer initializes depth to 0 and writes values
            # only for hit rays
            alpha_mask = ~rb.hit[...,0]
            rb.depth[alpha_mask] = clear_depth

            rb.depth = rb.depth.to(rb_dtype)
            out_rb = out_rb.blend(rb, channel_kit=self.state.graph.channels)

        return out_rb

    def _post_render(self, payload: FramePayload, rb: RenderBuffer) -> RenderBuffer:
        # Update current resolution in case it was decreased to maintain fps
        self.res_x, self.res_y = payload.render_res_x, payload.render_res_y

        # Cache information to accelerate next frames
        self._last_renderbuffer = rb

        # Record last state, to, i.e, calculate if needs to redraw future frames
        self._last_state['camera'] = copy.deepcopy(payload.camera)
        self._last_state['res_x'] = payload.render_res_x
        self._last_state['res_y'] = payload.render_res_y
        for renderer_id, renderer in self._renderers.items():
            if renderer_id in payload.visible_objects:
                renderer.post_render()

        # Create an output renderbuffer to contain the currently viewed mode as rgba channel
        output_rb = self.map_output_channels_to_rgba(rb)
        return output_rb

    def needs_refresh(self) -> bool:
        if len(self._last_state) == 0:
            return True

        # Resolution check: if not full resolution - canvas is dirty
        if self._last_state['res_x'] != self.camera.width or self._last_state['res_y'] != self.camera.height:
            return True

        for att_name, prev_val in self._last_state.items():
            if not hasattr(self, att_name):
                continue
            curr_val = self.__getattribute__(att_name)
            if isinstance(curr_val, Camera):
                if not torch.allclose(curr_val, prev_val):
                    return True
            elif curr_val != prev_val:
                return True
        return False

    def _bl_renderer_data_layers(self) -> Dict[str, PrimitivesPack]:
        """ Returns the bottom level object data layers"""
        layers = dict()
        for renderer_id, renderer in self._renderers.items():
            layers[renderer_id] = renderer.data_layers()
        return layers

    def _cameras_data_layers(self) -> Iterable[PrimitivesPack]:
        """ Returns the available cameras data layer (all visible cameras layers are packed together) """
        cameras_to_redraw = {camera_id: camera for camera_id, camera in self.state.graph.cameras.items()
                             if self.state.graph.visible_objects.get(camera_id, False)}
        layers = self._camera_layers.regenerate_data_layers(cameras_to_redraw, self.state.renderer.clear_color_value)
        return layers.values()

    def active_data_layers(self) -> List[PrimitivesPack]:
        layers_to_draw = []
        for obj_state in self.state.graph.bl_renderers.values():
            for layer_id, layer in obj_state.data_layers.items():
                if obj_state.toggled_data_layers[layer_id]:
                    layers_to_draw.append(layer)
        camera_data_layers = self._cameras_data_layers()
        layers_to_draw.extend(camera_data_layers)
        return layers_to_draw

    def map_output_channels_to_rgba(self, rb: RenderBuffer):
        selected_output_channel = self.state.renderer.selected_canvas_channel.lower()
        rb_channel = rb.get_channel(selected_output_channel)

        if rb_channel is None:
            # Unknown channel type configured to view over the canvas.
            # That can happen if, i.e. no object have traced a RenderBuffer with this channel.
            # Instead of failing, create an empty rb
            height, width = rb.rgb.shape[:2]
            return self._create_empty_rb(height=height, width=width, dtype=rb.rgb.dtype)

        # Normalize channel to [0, 1]
        channels_kit = self.state.graph.channels
        channel_info = channels_kit.get(selected_output_channel, create_default_channel())
        normalized_channel = channel_info.normalize_fn(rb_channel.clone())  # Clone to protect from modifications

        # To RGB (in normalized space)
        # TODO (operel): incorporate color maps
        channel_dim = normalized_channel.shape[-1]
        if channel_dim == 1:
            rgb = torch.cat((normalized_channel, normalized_channel, normalized_channel), dim=-1)
        elif channel_dim == 2:
            rgb = torch.cat((normalized_channel, normalized_channel, torch.zeros_like(normalized_channel)), dim=-1)
        elif channel_dim == 3:
            rgb = normalized_channel
        else:
            raise ValueError('Cannot display channels with more than 3 dimensions over the canvas.')

        canvas_rb = RenderBuffer(rgb=rgb, depth=rb.depth, alpha=rb.alpha)
        return canvas_rb

    @property
    def renderers(self) -> Dict[str, BottomLevelRenderer]:
        """ All loaded bottom level renderers currently employed by the renderer core """
        return self._renderers

    @property
    def camera(self) -> Camera:
        return self.state.renderer.selected_camera

    @camera.setter
    def camera(self, camera: Camera) -> None:
        self.state.renderer.selected_camera = camera

    @property
    def target_fps(self) -> float:
        return self.state.renderer.target_fps

    @property
    def interactive_mode(self) -> bool:
        return self.state.renderer.interactive_mode

    @interactive_mode.setter
    def interactive_mode(self, mode: bool) -> None:
        self.state.renderer.interactive_mode = mode

    @property
    def selected_camera_lens(self) -> str:
        return self.state.renderer.selected_camera_lens

    @selected_camera_lens.setter
    def selected_camera_lens(self, lens: str):
        self.state.renderer.selected_camera_lens = lens


class TLAS(abc.ABC):
    def traverse(self, payload: FramePayload) -> Dict[BottomLevelRenderer, Rays]:
        pass


class ListTLAS(abc.ABC):
    def __init__(self, bl_renderers: Dict[str, BottomLevelRenderer]):
        self.bl_renderers = list(bl_renderers.values())
        self.bl_renderer_ids = list(bl_renderers.keys())

    def traverse(self, rays: Rays, payload: FramePayload) -> Dict[BottomLevelRenderer, Rays]:
        # TODO (operel): invoke aabb test on bottom level renderers
        return {bl_renderer: rays for renderer_id, bl_renderer in zip(self.bl_renderer_ids, self.bl_renderers)
                if renderer_id in payload.visible_objects}
