# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Dict
import imgui
from wisp.framework import WispState
from .widget_imgui import WidgetImgui


class WidgetPropertyEditor(WidgetImgui):

    running_idx = 0 # Used to distinguish multiple instances of property editor widget

    def __init__(self):
        super().__init__()

    def paint_row(self, title, value):
        imgui.text(title)
        imgui.next_column()
        imgui.separator()
        imgui.push_item_width(-1)
        if callable(value):  # Values may be editable widgets, in which case they're callable
            value()
        else:
            imgui.text(str(value))
        imgui.pop_item_width()

    def paint(self, state: WispState, properties: Dict[str, object] = None, *args, **kwargs):
        if properties is None:
            return
        try:
            imgui.begin_group()
            WidgetPropertyEditor.running_idx += 1
            running_idx = WidgetPropertyEditor.running_idx
            imgui.columns(2, f'properties_{running_idx}', border=True)

            # Draw properties table
            for row_idx, (title, value) in enumerate(properties.items()):
                self.paint_row(title, value)
                if row_idx < len(properties) - 1:
                    imgui.next_column()
        finally:
            imgui.columns(1)
            imgui.separator()
            WidgetPropertyEditor.running_idx -= 1
            imgui.end_group()
