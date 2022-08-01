import torch
from typing import Optional
from wisp.core import RenderBuffer, Rays
from wisp.accelstructs import OctreeAS
from wisp.models.nefs import SPCField
from wisp.tracers import PackedSPCTracer
from wisp.gfx.datalayers import Datalayers, OctreeDatalayers
from wisp.renderer.core.api import field_renderer, RayTracedRenderer, FramePayload


@field_renderer(SPCField, PackedSPCTracer)
class SPCRenderer(RayTracedRenderer):
    """ A renderer for SPC objects, used with pipelines of SPCField + PackedSPCTracer. """

    def __init__(self, nef: SPCField, batch_size=4000, *args, **kwargs):
        super().__init__(nef, *args, **kwargs)

        self.tracer = PackedSPCTracer()
        self.batch_size = batch_size
        self.render_res_x = None
        self.render_res_y = None
        self.output_width = None
        self.output_height = None
        self._data_layers = self.regenerate_data_layers()
        self.channels = None

    @classmethod
    def create_layers_painter(cls, nef: SPCField) -> Optional[Datalayers]:
        return OctreeDatalayers()   # Always assume an octree grid object exists for this field

    def pre_render(self, payload: FramePayload, *args, **kwargs) -> None:
        super().pre_render(payload)
        self.render_res_x = payload.render_res_x
        self.render_res_y = payload.render_res_y
        self.output_width = payload.camera.width
        self.output_height = payload.camera.height
        self.channels = payload.channels

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        """ SPC never requires a refresh due to internal state changes """
        return False

    def render(self, rays: Optional[Rays] = None) -> RenderBuffer:
        rb = RenderBuffer(hit=None)
        for ray_batch in rays.split(self.batch_size):
            rb += self.tracer(self.nef, channels=self.channels, rays=ray_batch, lod_idx=None)

        # Rescale renderbuffer to original size
        rb = rb.reshape(self.render_res_y, self.render_res_x, -1)
        if self.render_res_x != self.output_width or self.render_res_y != self.output_height:
            rb = rb.scale(size=(self.output_height, self.output_width))
        return rb

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def model_matrix(self) -> torch.Tensor:
        return torch.eye(4, device=self.device)

    @property
    def aabb(self) -> torch.Tensor:
        # (center_x, center_y, center_z, width, height, depth)
        return torch.tensor((0.0, 0.0, 0.0, 2.0, 2.0, 2.0), device=self.device)

    def acceleration_structure(self):
        return "Octree"  # Assumes to always use OctreeAS

    def features_structure(self):
        return "Octree Grid"  # Assumes to always use OctreeGrid for storing features
