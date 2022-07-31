# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy

import numpy as np
from typing import Optional
from glumpy import gloo, gl
from kaolin.render.camera import Camera
from wisp.renderer.gizmos.gizmo import Gizmo


class AxisPainter(Gizmo):

    def __init__(self, axes_length, line_width, origin=None, axes=None, is_bidirectional=True):
        """
        :param axes_length Axes will occupy [origin, origin + axes_length] in world coordinates
        :param line_width Line width
        :param origin numpy array of 3 coordinates for the x,y,z of the origin
        :param axes Iterable of strings to specify which axes to draw. Possible values: 'x', 'y', 'z'
        :param is_bidirectional if True, will also draw [origin, origin - axes_length] in world coordinates
        """
        if origin is None:
            origin = np.array((0.0, 0.0, 0.0))
        if axes is None:
            axes = ('x', 'y', 'z')
        self.vbo, self.ibo = self.create_line_buffers(axes_length,
                                                      origin=origin, axes=axes, is_bidirectional=is_bidirectional)
        self.line_size = line_width
        self.canvas_program: Optional[gloo.Program] = self.create_gl_program()

    def destroy(self):
        """ Release GL resources, must be called from the rendering thread which owns the GL context """
        if self.vbo is not None:
            self.vbo.delete()
            self.vbo = None
        if self.ibo is not None:
            self.ibo.delete()
            self.ibo = None
        if self.canvas_program is not None:
            self.canvas_program.delete()
            self.canvas_program = None

    def create_gl_program(self):
        vertex = """
                    uniform mat4   u_viewprojection;
                    attribute vec3 position;
                    attribute vec4 color;
                    varying vec4 v_color;
                    void main()
                    {
                        v_color = color;
                        gl_Position = u_viewprojection * vec4(position, 1.0f);
                    } """

        fragment = """
                    varying vec4 v_color;
                    void main()
                    {
                        gl_FragColor = v_color;
                    } """

        # Compile GL program
        canvas = gloo.Program(vertex, fragment)
        return canvas

    def create_line_buffers(self, axes_length, origin, axes, is_bidirectional):

        if is_bidirectional:
            segment_ends = (-axes_length, axes_length)
        else:
            segment_ends = (axes_length,)

        num_lines = len(segment_ends) * len(axes)
        vertex_buffer = np.zeros(2 * num_lines, [("position", np.float32, 3), ("color", np.float32, 4)])

        blend_to_alpha = 0.6   # Makes the line appear more transparent towards the far tip
        idx = 0
        for axis in axes:
            for end_val in segment_ends:
                start = origin
                if axis == 'x':
                    end = np.array((end_val, 0.0, 0.0))
                    color_start = np.array((1.0, 0.0, 0.0, 1.0))
                    color_end = np.array((1.0, 0.0, 0.0, blend_to_alpha))
                elif axis == 'y':
                    end = np.array((0.0, end_val, 0.0))
                    color_start = np.array((0.0, 1.0, 0.0, 1.0))
                    color_end = np.array((0.0, 1.0, 0.0, blend_to_alpha))
                elif axis == 'z':
                    end = np.array((0.0, 0.0, end_val))
                    color_start = np.array((0.0, 0.0, 1.0, 1.0))
                    color_end = np.array((0.0, 0.0, 1.0, blend_to_alpha))
                vertex_buffer["position"][idx] = start
                vertex_buffer["position"][idx+1] = end
                vertex_buffer["color"][idx] = color_start
                vertex_buffer["color"][idx+1] = color_end
                idx += 2
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        index_buffer = np.arange(0, 2 * num_lines).astype(np.uint32)
        index_buffer = index_buffer.view(gloo.IndexBuffer)

        return vertex_buffer, index_buffer

    def render(self, camera: Camera):
        gl.glLineWidth(self.line_size)
        self.canvas_program["u_viewprojection"] = camera.view_projection_matrix()[0].cpu().numpy().T
        self.canvas_program.bind(self.vbo)
        self.canvas_program.draw(gl.GL_LINES, self.ibo)