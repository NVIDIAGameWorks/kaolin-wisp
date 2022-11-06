# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import numpy as np
from typing import Optional, Tuple
from glumpy import gloo, gl
from kaolin.render.camera import Camera
from wisp.renderer.gizmos.gizmo import Gizmo


class WorldGrid(Gizmo):

    def __init__(self, squares_per_axis: int = 20, grid_size: float = 1.0,
            line_color: Tuple = None, line_size: int = 2, plane: str = 'xy'):
        """
        Args:
            squares_per_axis (int): Number of grid squares per row, column
            grid_size (float): Grid will occupy [-size, size] in world coordinates
            line_color (Tuple): Tuple of RGB values for line color, in range [0,255]. i.e: (255, 255, 255) for white.
            line_size (int): integer between [1,32], where each grid square can be thought of a 32 x 32 pixel grid,
               and line size specifies how many rows / columns each line occupies for each square.
               The actual amount of pixels each line occupies also depends on the grid_size and camera settings.
            plane (str): Plane to align the reference grid with. Valid choices: 'xy', 'xz', 'yz'.
        """
        # Size of each square texture, to be repeated, setting this one too low will decrease quality.
        # We may go above 32 x 32 and scale the line size accordingly
        self.tex_size = 64
        self.line_size = line_size * (self.tex_size / 64)
        if line_color is None:
            line_color = (255, 255, 255)
        self.line_color = np.array(line_color)
        self.squares_per_axis = squares_per_axis
        self.grid_size = grid_size
        dtype = np.uint8
        tex_data = np.zeros((self.tex_size, self.tex_size, 4), dtype=dtype)

        for j in range(0, self.tex_size):   # Height
            for i in range(0, self.tex_size):   # Width
                tex_data[j, i, :3] = self.line_color if (i < self.line_size or j < self.line_size) else 0
                tex_data[j, i, 3] = 255 if (i < self.line_size or j < self.line_size) else 0    # Alpha

        tex = tex_data.view(gloo.Texture2D)

        tex.activate()  # Binds texture
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        try:
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_ANISOTROPY, 16)
        except AttributeError as e:
            logging.warning('GL_TEXTURE_MAX_ANISOTROPY not available; appearance may suffer')
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        tex.deactivate()

        # GL program used to paint a single billboard
        self.canvas_program: Optional[gloo.Program] = self.create_gl_program(tex, plane)

    def destroy(self):
        """ Release GL resources, must be called from the rendering thread which owns the GL context """
        if self.canvas_program is not None:
            if self.canvas_program['tex'] is not None:
                self.canvas_program['tex'].delete()
                self.canvas_program['tex'] = None
            self.canvas_program.delete()
            self.canvas_program = None

    def create_gl_program(self, texture: np.ndarray, plane: str):
        vertex = """
                    uniform mat4   u_viewprojection;
                    attribute vec4 position;
                    attribute vec4 texcoord;
                    varying vec4 v_texcoord;
                    void main()
                    {
                        v_texcoord = texcoord;
                        gl_Position = u_viewprojection * position;
                    } """

        fragment = """
                    uniform sampler2D tex;
                    varying vec4 v_texcoord;
                        
                    void main()
                    {
                        vec4 sample = texture2DProj(tex, v_texcoord);
                        if (sample.a <= 0)
                            discard;
                        gl_FragColor = sample;
                    } """

        # Compile GL program
        canvas = gloo.Program(vertex, fragment, count=4)

        # Upload fixed values to GPU
        if plane == 'xz':
            canvas["position"] = [(-self.grid_size, 0.0, +self.grid_size, 1.0),
                                  (+self.grid_size, 0.0, +self.grid_size, 1.0),
                                  (-self.grid_size, 0.0, -self.grid_size, 1.0),
                                  (+self.grid_size, 0.0, -self.grid_size, 1.0)]
        elif plane == 'xy':
            canvas["position"] = [(-self.grid_size, +self.grid_size, 0.0, 1.0),
                                  (+self.grid_size, +self.grid_size, 0.0, 1.0),
                                  (-self.grid_size, -self.grid_size, 0.0, 1.0),
                                  (+self.grid_size, -self.grid_size, 0.0, 1.0)]
        elif plane == 'yz':
            canvas["position"] = [(0.0, -self.grid_size, +self.grid_size, 1.0),
                                  (0.0, +self.grid_size, +self.grid_size, 1.0),
                                  (0.0, -self.grid_size, -self.grid_size, 1.0),
                                  (0.0, +self.grid_size, -self.grid_size, 1.0)]

        # Repeat square pattern with texture coordinates > 1.0
        reps = self.squares_per_axis + float(self.line_size) / self.tex_size
        canvas['texcoord'] = [(0, 0, 0.0, 1.0), (reps, 0, 0.0, 1.0), (0, reps, 0.0, 1.0), (reps, reps, 0.0, 1.0)]
        canvas['tex'] = texture
        return canvas

    def render(self, camera: Camera):
        # TODO(operel): state control should be elsewhere
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.canvas_program["u_viewprojection"] = camera.view_projection_matrix()[0].cpu().numpy().T
        self.canvas_program.draw(gl.GL_TRIANGLE_STRIP)
