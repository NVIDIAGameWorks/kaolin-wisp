# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Optional
from wisp.models.nefs import SPCField
from wisp.tracers import PackedSPCTracer
from wisp.gfx.datalayers import Datalayers, OctreeDatalayers
from wisp.renderer.core.api import RayTracedRenderer, field_renderer, FramePayload


@field_renderer(SPCField, PackedSPCTracer)
class SPCRenderer(RayTracedRenderer):
    """ A renderer for SPC objects, used with pipelines of SPCField + PackedSPCTracer. """

    def __init__(self, nef: SPCField, tracer: PackedSPCTracer, batch_size=4000, *args, **kwargs):
        super().__init__(nef, tracer, batch_size, *args, **kwargs)

    @classmethod
    def create_layers_painter(cls, nef: SPCField) -> Optional[Datalayers]:
        return OctreeDatalayers()   # Always assume an octree grid object exists for this field

    def needs_refresh(self, payload: FramePayload, *args, **kwargs) -> bool:
        """ SPC never requires a refresh due to internal state changes """
        return False

    def acceleration_structure(self) -> str:
        """ Returns a human readable name of the bottom level acceleration structure used by this renderer """
        return "Octree"  # Assumes to always use OctreeAS

    def features_structure(self) -> str:
        """ Returns a human readable name of the feature structure used by this renderer """
        return "Octree Grid"  # Assumes to always use OctreeGrid for storing features
