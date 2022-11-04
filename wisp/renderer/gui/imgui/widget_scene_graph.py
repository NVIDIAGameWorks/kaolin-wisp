# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import imgui
from wisp.core.colors import light_purple, white, lime, orange
from wisp.framework import WispState
from .widget_imgui import WidgetImgui
from .widget_radiance_pipeline_renderer import WidgetNeuralRadianceFieldRenderer
from .widget_sdf_pipeline_renderer import WidgetNeuralSDFRenderer
from .widget_cameras import WidgetCameraProperties
from .widget_object_transform import WidgetObjectTransform
from wisp.renderer.core.api import request_redraw


class WidgetSceneGraph(WidgetImgui):
    names_to_widgets = dict(
        NeuralSDFPackedRenderer=WidgetNeuralSDFRenderer,
        NeuralRadianceFieldPackedRenderer=WidgetNeuralRadianceFieldRenderer,
    )

    names_to_color = dict(
        NeuralSDFPackedRenderer=orange,
        NeuralRadianceFieldPackedRenderer=light_purple,
        Camera=lime,
    )

    names_to_title = dict(
        NeuralSDFPackedRenderer="Neural Signed Distance Field",
        NeuralRadianceFieldPackedRenderer="Neural Radiance Field",
        Camera="Camera",
    )

    def __init__(self):
        super().__init__()
        self.object_widgets = dict()
        self.transform_widget = WidgetObjectTransform()

    def add_object_type(self, name, widget, color, title):
        self.names_to_widgets[name] = widget
        self.names_to_color[name] = color
        self.names_to_title[name] = title

    def get_bl_renderer_widget(self, object_id, object_type):
        if object_id not in self.object_widgets:
            widget_type = self.names_to_widgets.get(object_type)
            self.object_widgets[object_id] = widget_type() if widget_type is not None else None
        return self.object_widgets[object_id]

    def get_object_color(self, object_type):
        return self.names_to_color.get(object_type, white)

    def get_object_title(self, object_type):
        return self.names_to_title.get(object_type, f"{object_type}")

    def refresh_object_widgets(self, state: WispState):
        # Remove widgets for obsolete objects
        for widget_id in list(self.object_widgets.keys()):
            if widget_id not in state.graph.bl_renderers:
                del self.object_widgets[widget_id]

    @staticmethod
    def paint_object_checkbox(state, obj_id):
        visible_objects = state.graph.visible_objects
        is_checked = visible_objects.get(obj_id, False)
        visibility_toggled, is_checked = imgui.checkbox(f"##{obj_id}", is_checked)
        visible_objects[obj_id] = is_checked
        if visibility_toggled:
            request_redraw(state)

    @staticmethod
    def paint_all_objects_checkbox(state):
        visible_objects = state.graph.visible_objects
        is_checked = any([visible_objects.get(obj_id, False) for obj_id in state.graph.bl_renderers.keys()])
        visibility_toggled, is_checked = imgui.checkbox(f"##sg_all_bl_renderers", is_checked)
        if visibility_toggled:
            for obj_id in state.graph.bl_renderers.keys():
                visible_objects[obj_id] = is_checked
            request_redraw(state)

    @staticmethod
    def paint_all_cameras_checkbox(state):
        visible_objects = state.graph.visible_objects
        is_checked = any([visible_objects.get(obj_id, False) for obj_id in state.graph.cameras.keys()])
        visibility_toggled, is_checked = imgui.checkbox(f"##sg_all_cameras", is_checked)
        if visibility_toggled:
            for cam_id in state.graph.cameras.keys():
                visible_objects[cam_id] = is_checked
            request_redraw(state)

    def paint(self, state: WispState, *args, **kwargs):
        expanded, _ = imgui.collapsing_header("Scene Objects", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded:
            bl_renderers = state.graph.bl_renderers
            self.refresh_object_widgets(state)

            if len(bl_renderers) > 0:
                self.paint_all_objects_checkbox(state)
                imgui.same_line()
                if imgui.tree_node("Objects", imgui.TREE_NODE_DEFAULT_OPEN):
                    for obj_id, obj in bl_renderers.items():
                        if obj.status != 'loaded':
                            continue
                        obj_type = type(obj.renderer).__name__
                        obj_color = self.get_object_color(obj_type)

                        self.paint_object_checkbox(state, obj_id)
                        imgui.same_line()
                        if imgui.tree_node(obj_id, imgui.TREE_NODE_DEFAULT_OPEN):
                            imgui.text(f"Type:")
                            imgui.same_line()
                            obj_title = self.get_object_title(obj_type)
                            imgui.text_colored(f"{obj_title}", *obj_color)

                            object_transform = obj.transform
                            if object_transform is not None:
                                if imgui.tree_node("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
                                    self.transform_widget.paint(state, object_transform)
                                imgui.tree_pop()

                            if imgui.tree_node("Properties", imgui.TREE_NODE_DEFAULT_OPEN):
                                bl_renderer_widget = self.get_bl_renderer_widget(obj_id, obj_type)
                                if bl_renderer_widget is not None:
                                    bl_renderer_widget.paint(state, obj.renderer)
                                imgui.tree_pop()

                            if imgui.tree_node("Layers"):
                                obj_data_layers = obj.data_layers
                                toggled_obj_data_layers = obj.toggled_data_layers
                                for layer in obj_data_layers.keys():
                                    is_prev_selected = toggled_obj_data_layers[layer]
                                    _, is_selected = imgui.checkbox(f"{layer}", is_prev_selected)
                                    toggled_obj_data_layers[layer] = is_selected
                                    if is_prev_selected != is_selected:
                                        request_redraw(state)
                                imgui.tree_pop()
                            imgui.tree_pop()
                    imgui.tree_pop()

            if len(state.graph.cameras) > 0:
                self.paint_all_cameras_checkbox(state)
                imgui.same_line()
                if imgui.tree_node("Cameras", imgui.TREE_NODE_DEFAULT_OPEN):
                    for cam_id, cam in state.graph.cameras.items():
                        self.paint_object_checkbox(state, cam_id)
                        imgui.same_line()
                        if imgui.tree_node(f"Camera {cam_id}"):
                            obj_type = type(cam).__name__
                            obj_color = self.get_object_color(obj_type)
                            imgui.text(f"Type:")
                            imgui.same_line()
                            imgui.text_colored(f"{obj_type}", *obj_color)
                            cam_widget = WidgetCameraProperties(camera_id=cam_id)
                            cam_widget.paint(state, cam)
                            imgui.tree_pop()
                    imgui.tree_pop()
                # TODO (operel): Add proper camera widget caching
