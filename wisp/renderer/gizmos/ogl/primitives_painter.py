# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
from collections import defaultdict
import torch
from typing import Optional, List

import numpy as np
from glumpy import gloo, gl
from kaolin.render.camera import Camera

from wisp.core.primitives import PrimitivesPack
from wisp.renderer.gizmos.gizmo import Gizmo

# TODO (operel): release GL resources when instance is destroyed (and issue a warning if not taken care of explicitly)

class PrimitivesPainter(Gizmo):

    def __init__(self):
        # GL program used to paint GL primitives
        self.canvas_program: Optional[gloo.Program] = self.create_gl_program()
        self.lines = []
        self.points = []

    def destroy(self):
        """ Release GL resources, must be called from the rendering thread which owns the GL context """
        self.clear()
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

    def clear(self):
        for line in self.lines:
            vertex_buffer, index_buffer, _ = line
            vertex_buffer.delete()
            index_buffer.delete()

        self.lines = []
        self.points = []

    def create_points_buffers(self, points, point_size):
        # TODO (operel): gpu copy is better..
        pos, color = points
        count = len(pos)
        vertex_buffer = np.zeros(count, [("position", np.float32, 3), ("color", np.float32, 4)])
        vertex_buffer["position"] = torch.cat(pos).cpu().numpy()
        vertex_buffer["color"] = torch.cat(color).reshape(-1, 4).cpu().numpy()
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)
        return vertex_buffer

    def create_line_buffers(self, lines):
        # TODO (operel): gpu copy is better..
        start, end, color = lines
        count = len(start)
        vertex_buffer = np.zeros(2 * count, [("position", np.float32, 3), ("color", np.float32, 4)])
        vertex_buffer["position"] = torch.cat([start, end], dim=-1).reshape(-1, 3).cpu().numpy()
        vertex_buffer["color"] = torch.cat([color, color], dim=-1).reshape(-1, 4).cpu().numpy()
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        index_buffer = np.arange(0, 2 * count).astype(np.uint32)
        index_buffer = index_buffer.view(gloo.IndexBuffer)

        return vertex_buffer, index_buffer

    def redraw(self, primitives: List[PrimitivesPack]):
        self.clear()

        # Sort primitives by gl state changes - this reduces the amount of draw calls
        draw_calls = dict(
            lines=dict(
                line_width=defaultdict(PrimitivesPack)
            ),
            points=dict(
                point_size=defaultdict(PrimitivesPack)
            ),
        )

        for pack in primitives:
            if (len(pack._lines_start)):
                draw_calls['lines']['line_width'][pack.line_width].append(pack)
    
                for _line_width, pack in draw_calls['lines']['line_width'].items():
                    lines = pack.lines
                    vertex_buffer, index_buffer = self.create_line_buffers(lines)
                    self.lines.append((vertex_buffer, index_buffer, pack.line_width))
            elif (len(pack._points_pos)):
                draw_calls['points']['point_size'][pack.point_size].append(pack)

                for point_size, pack in draw_calls['points']['point_size'].items():
                    points = pack.points
                    vertex_buffer = self.create_points_buffers(points, point_size)
                    self.points.append((vertex_buffer, point_size))

    def render(self, camera: Camera):
        for line_entry in self.lines:
            vertex_buffer, index_buffer, line_width = line_entry
            gl.glLineWidth(line_width)
            self.canvas_program["u_viewprojection"] = camera.view_projection_matrix()[0].cpu().numpy().T
            self.canvas_program.bind(vertex_buffer)
            self.canvas_program.draw(gl.GL_LINES, index_buffer)
    
        for point_entry in self.points:
            vertex_buffer, point_size = point_entry
            gl.glPointSize(point_size)
            self.canvas_program["u_viewprojection"] = camera.view_projection_matrix()[0].cpu().numpy().T
            self.canvas_program.bind(vertex_buffer)
            self.canvas_program.draw(gl.GL_POINTS)