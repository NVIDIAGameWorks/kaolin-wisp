# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.framework import WispState
from wisp.core.transforms import ObjectTransform
from .widget_imgui import WidgetImgui


class WidgetObjectTransform(WidgetImgui):

    def __init__(self):
        super().__init__()

    def paint(self, state: WispState, object_transform: ObjectTransform = None, *args, **kwargs):
        if object_transform is None:
            return

        button_width = 100
        if imgui.button("Reset", width=button_width):
            object_transform.reset()

        changed, values = imgui.slider_float3(
            "Translate",
            object_transform.tx, object_transform.ty, object_transform.tz,
            min_value=-15.0, max_value=15.0,
            format="%.2f",
            power=1.0
        )
        if changed:
            object_transform.tx = values[0]
            object_transform.ty = values[1]
            object_transform.tz = values[2]

        changed, values = imgui.slider_float3(
            "Rotate",
            object_transform.rx, object_transform.ry, object_transform.rz,
            min_value=-180.0, max_value=180.0,
            format="%.2f",
            power=1.0
        )
        if changed:
            object_transform.rx = values[0]
            object_transform.ry = values[1]
            object_transform.rz = values[2]

        changed, values = imgui.slider_float3(
            "Scale",
            object_transform.sx, object_transform.sy, object_transform.sz,
            min_value=-15.0, max_value=15.0,
            format="%.2f",
            power=1.0
        )
        if changed:
            object_transform.sx = values[0]
            object_transform.sy = values[1]
            object_transform.sz = values[2]
