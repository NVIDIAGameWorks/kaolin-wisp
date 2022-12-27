# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Dict
import imgui
import numpy as np
from wisp.framework import WispState
from .widget_imgui import WidgetImgui


class WidgetPropertyEditor(WidgetImgui):

    running_idx = 0  # Used to distinguish multiple instances of property editor widget

    def __init__(self):
        super().__init__()

    def paint_row(self, title, value):
        imgui.text(title)
        imgui.next_column()
        imgui.separator()
        imgui.push_item_width(-1)
        if value is None:
            imgui.text('None')
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            # Value is a sequence, concatenate entries as string
            if len(value) == 0:
                txt = 'None'
            else:
                txt = ', '.join([str(e) for e in value])
            imgui.text(txt)
        elif isinstance(value, float):          # Round and format float
            imgui.text("{:.3f}".format(value))
        elif callable(value):                   # Values may be editable widgets, in which case they're callable
            try:
                value()
            except Exception as e1:
                try:
                    imgui.text(type(value).__name__)    # Callable is probably not an editable widget, use typename
                except Exception as e2:
                    raise ValueError(f"Gui failed to paint property - '{title}: {value}'. "
                                     f"This error message can occur, i.e., if a wisp-module returns a public property of "
                                     f"some unsupported type.") from e1
        else:                                   # Other types are cast to str
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
