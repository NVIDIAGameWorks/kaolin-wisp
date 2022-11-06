# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import imgui
from .widget_imgui import WidgetImgui
from wisp.framework import WispState
import pynvml
pynvml.nvmlInit()


class WidgetGPUStats(WidgetImgui):
    def __init__(self):
        super().__init__()
        self.text_color = (0.2, 1., 0.)  # RGB

    def paint(self, state: WispState, *args, **kwargs):

        imgui.text_colored(f"Frame time: {int(state.renderer.dt*1000)}ms / FPS: {int(state.renderer.fps)}", *self.text_color)
        imgui.text(f"Interactive Mode: {'on' if state.renderer.interactive_mode else 'off'}")
        imgui.separator()

        device = state.renderer.device
        if device == 'cpu':
            device_name = 'Running on CPU'
            imgui.text(f"{device_name}")
        else:
            device_name = torch.cuda.get_device_name(device=device)
            imgui.text(f"{device_name}, CUDA v{torch.version.cuda}")
            width, height = imgui.get_content_region_available()

            device_index = getattr(device, 'index', None)
            device_index = device_index if device_index is not None else torch.cuda.current_device()
            nvml_device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            nvml_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device_handle)
            used_mem = nvml_info.used
            total_mem = nvml_info.total
            dec_str = '{:.2f}'
            imgui.text("Global GPU Mem: ")
            imgui.same_line()
            imgui.progress_bar(fraction=used_mem / total_mem, size=(width / 2, 0),
                               overlay=f"{dec_str.format(used_mem / 1024**3)} / "
                                       f"{dec_str.format(total_mem / 1024**3)} GB")
