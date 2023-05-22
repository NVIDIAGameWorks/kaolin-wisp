# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import datetime
import imgui
from wisp.framework import WispState
from .widget_imgui import WidgetImgui


class WidgetOptimization(WidgetImgui):
    def __init__(self):
        super().__init__()
        self.active_text_color = (0.2, 1., 0.)         # RGB (green)
        self.paused_text_color = (1.0, 0.917, 0.164)   # RGB (yellow)
        self.stopped_text_color = (1.0, 0.0, 0.0)      # RGB (yellow)

    def paint(self, state: WispState, *args, **kwargs):
        expanded, _ = imgui.collapsing_header("Optimization", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if not expanded:
            return

        is_training_stopped = not state.optimization.running or state.renderer.background_tasks_paused
        is_training_paused = state.renderer.background_tasks_paused

        if is_training_stopped:
            imgui.text_colored(f"Training stopped.", *self.stopped_text_color)
        elif is_training_paused:
            imgui.text_colored(f"Training paused.", *self.paused_text_color)
        else:
            imgui.text_colored(f"Training running.", *self.active_text_color)

        curr_iteration = state.optimization.iteration
        curr_epoch = state.optimization.epoch
        button_width = 150
        if not is_training_stopped:
            if imgui.button("Stop Training", width=button_width):
                state.optimization.running = False
                state.renderer.background_tasks_paused = True
        else:
            if curr_epoch == 1 and curr_iteration == 1:
                if imgui.button("Start Training", width=button_width):
                    state.optimization.running = True
                    state.renderer.background_tasks_paused = False
            else:
                if imgui.button("Resume Training", width=button_width):
                    state.optimization.running = True
                    state.renderer.background_tasks_paused = False

        imgui.same_line()
        elapsed_time_seconds = state.optimization.elapsed_time
        formatted_time = str(datetime.timedelta(seconds=elapsed_time_seconds))
        formatted_time = formatted_time.split(".")[0]   # Remove microseconds
        imgui.text(f"Elapsed Time: {formatted_time}")

        if is_training_stopped:
            state.renderer.target_fps = None
        else:
            _, checkbox_enabled = imgui.checkbox(label=f"Limit rendering FPS",
                                                 state=state.renderer.target_fps is not None)

            # Would be better to have a disabled widget instead of an invisible one,
            # but as of June 2022 the official pyimgui release doesn't expose push_item_flag + ITEM_DISABLED
            if checkbox_enabled:
                imgui.same_line()
                imgui.push_item_width(-30)
                target_fps = state.renderer.target_fps if state.renderer.target_fps is not None else 0
                changed, value = imgui.slider_int("##target_fps", value=target_fps, min_value=0, max_value=60)
                state.renderer.target_fps = value
                imgui.pop_item_width()
            else:
                state.renderer.target_fps = None

        if not is_training_stopped or curr_epoch > 0 or curr_iteration > 1:
            width, height = imgui.get_content_region_available()
            total_epochs = state.optimization.max_epochs
            if total_epochs > 0:
                epoch_progress = curr_epoch / total_epochs
                imgui.text("Epoch:")
                imgui.progress_bar(fraction=epoch_progress, size=(width, 0), overlay=f"{curr_epoch} / {total_epochs}")

            total_iteration = state.optimization.iterations_per_epoch
            if total_iteration > 0:
                iterations_progress = curr_iteration / total_iteration
            else:
                iterations_progress = 0
            imgui.text("Iteration:")
            imgui.progress_bar(fraction=iterations_progress, size=(width, 0),
                               overlay=f"{curr_iteration} / {total_iteration}")

            if len(state.optimization.losses) > 0:
                if imgui.tree_node("Losses", imgui.TREE_NODE_DEFAULT_OPEN):
                    for loss_name, loss_vals in state.optimization.losses.items():
                        formatted_loss_name = loss_name.replace('_', ' ').capitalize()
                        imgui.text(f"{formatted_loss_name}:")
                        imgui.plot_lines("##losses", np.array(loss_vals, dtype=np.float32),
                                         graph_size=(width - 30, 35))
                    imgui.tree_pop()
