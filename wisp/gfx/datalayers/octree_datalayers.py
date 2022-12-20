# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from typing import Dict
import kaolin.ops.spc as spc_ops
from wisp.core import PrimitivesPack
from wisp.accelstructs import OctreeAS
from wisp.gfx.datalayers import Datalayers
from wisp.core.colors import soft_blue, soft_red, lime_green, purple, gold


class OctreeDatalayers(Datalayers):

    def __init__(self):
        self._last_state = dict()

    def needs_redraw(self, blas: OctreeAS) -> True:
        # Pyramids contain information about the number of cells per level,
        # it's a plausible heuristic to determine whether the frame should be redrawn
        return not ('pyramids' in self._last_state and
                    torch.equal(self._last_state['pyramids'], blas.pyramid))

    def regenerate_data_layers(self, blas: OctreeAS) -> Dict[str, PrimitivesPack]:
        data_layers = dict()
        lod_colors = [
            torch.tensor((*soft_blue, 1.0)),
            torch.tensor((*soft_red, 1.0)),
            torch.tensor((*lime_green, 1.0)),
            torch.tensor((*purple, 1.0)),
            torch.tensor((*gold, 1.0)),
        ]

        for lod in range(blas.max_level):
            cells = PrimitivesPack()
            
            level_points = spc_ops.unbatched_get_level_points(blas.points, blas.pyramid, lod)
            
            corners = spc_ops.points_to_corners(level_points) / (2 ** lod)
            
            corners = corners * 2.0 - 1.0

            grid_lines = corners[:, [(0, 1), (1, 3), (3, 2), (2, 0),
                                     (4, 5), (5, 7), (7, 6), (6, 4),
                                     (0, 4), (1, 5), (2, 6), (3, 7)]]

            grid_lines_start = grid_lines[:, :, 0].reshape(-1, 3)
            grid_lines_end = grid_lines[:, :, 1].reshape(-1, 3)
            color_tensor = lod_colors[lod % len(lod_colors)]
            grid_lines_color = color_tensor.repeat(grid_lines_start.shape[0], 1)
            cells.add_lines(grid_lines_start, grid_lines_end, grid_lines_color)

            data_layers[f'Octree LOD{lod}'] = cells

        self._last_state['pyramids'] = blas.pyramid
        return data_layers
