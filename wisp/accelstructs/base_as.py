# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from wisp.core import WispModule


@dataclass
class ASQueryResults:
    """ A data holder for keeping the results of a single acceleration structure query() call.
    A query receives a set of input coordinates and returns the cell indices of the acceleration structure where
    the query coordinates fall.
    """

    pidx: torch.LongTensor
    """ Holds the query results.
    - If the query is invoked with `with_parents=False`, this field is a tensor of shape [num_coords],
      containing indices of cells of the acceleration structure, where the query coordinates match.
    - If the query is invoked with `with_parents=True`, this field is a tensor of shape [num_coords, level+1],
      containing indices of the cells of the acceleration structure + the full parent hierarchy of each 
      cell query result.
    """


@dataclass
class ASRaytraceResults:
    """ A data holder for keeping the results of a single acceleration structure raytrace() call.
    A raytrace operation returns all intersections of the ray set with the acceleration structure cells.
    Ray/cell intersections are also referred to in Kaolin & Wisp as "nuggets".
    """

    ridx: torch.LongTensor
    """ A tensor containing the ray index of the ray that intersected each cell [num_nuggets]. 
    (can be used to index into rays.origins and rays.dirs)
    """

    pidx: torch.LongTensor
    """ Point indices into the cells of the acceleration structure, where the ray intersected a cell [num_nuggets] """

    depth: torch.FloatTensor
    """ Depths of each nugget, representing:
      - The first intersection point of the ray with the cell (entry), and 
      - Optionally also a second intersection point of the ray with the cell (exit).
      A tensor of [num_intersections, 1 or 2]. 
    """


@dataclass
class ASRaymarchResults:
    """ A data holder for keeping the results of a single acceleration structure raymarching() call.
    A raymarch operation advances the ray set within the acceleration structure and generates samples along the rays.
    """

    samples: torch.FloatTensor
    """ Sample coordinates of shape [num_hit_samples, 3]"""

    ridx: torch.LongTensor
    """ A tensor containing the ray index of the ray that generated each sample [num_hit_samples]. 
    Can be used to index into rays.origins and rays.dirs.
    """

    depth_samples: Optional[torch.FloatTensor]
    """ Depths of each sample, a tensor of shape [num_hit_samples, 1] """

    deltas: Optional[torch.FloatTensor]
    """ Depth diffs between each sample and the previous one, a tensor of shape [num_hit_samples, 1] """

    boundary: Optional[torch.BoolTensor]
    """ Boundary tensor which marks the beginning of each variable-sized sample pack of shape [num_hit_samples].
        That is: [True, False, False, False, True, False False] represents two rays of 4 and 3 samples respectively. 
    """


class BaseAS(WispModule, ABC):
    """
    A base interface for all acceleration structures within Wisp.
    """

    def __init__(self):
        super().__init__()

    def query(self, coords, level=None, with_parents=False) -> ASQueryResults:
        """Returns the ``cell index`` corresponding to the sample coordinates
        (indices of acceleration structure cells the query coords match).

        Args:
            coords (torch.FloatTensor) : 3D coordinates of shape [num_coords, 3].
                Certain acceleration structures may require the coords to be normalized within some range.
            level (int) : For acceleration structures which support multiple level-of-details,
                level specifies the desired LOD to query. If None, queries the highest level.
            with_parents (bool) : For acceleration structures which support multiple level-of-details,
                If `with_parents=True`, this call will return the hierarchical parent indices as well.

        Returns:
            (ASQueryResults): A plain data object carrying the results of the results of the query operation,
            containing the indices of cells matching the query coords.
        """
        raise NotImplementedError(f"{self.name} acceleration structure does not support the 'query' method.")

    def raytrace(self, rays, level=None, with_exit=False) -> ASRaytraceResults:
        """Traces rays against the SPC structure, returning all intersections along the ray with the SPC points
        (SPC points are quantized, and can be interpreted as octree cell centers or corners).

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            level (int) : The level of the octree to raytrace. If None, traces the highest level.
            with_exit (bool) : If True, also returns exit depth.

        Returns:
            (ASRaytraceResults): A plain data object carrying the results of the raytrace operation,
             containing the indices of ray and cell intersections.
        """
        raise NotImplementedError(f"{self.name} acceleration structure does not support the 'raytrace' method.")

    def raymarch(self, rays, *args, **kwargs) -> ASRaymarchResults:
        """Samples points along the ray within the acceleration structure.
        The exact algorithm used is under the responsibility of the acceleration structure.

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            *args, **kwargs: Optional args this specific acceleration structure requires for marching.
        Returns:
            (RaymarchResults): a plain data object carrying the results of raymarching,
            containing samples generated by the ray marching.
        """
        raise NotImplementedError(f"{self.name} acceleration structure does not support the 'raymarch' method.")

    def occupancy(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the number of cells occupied in that LOD.
        Used for logging, debugging and visualization purposes, for acceleration structures with
        a defined concept of occupancy.
        """
        return list()

    def capacity(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the total cell capacity in that LOD.
        Used for logging, debugging and visualization purposes, for acceleration structures with
        a defined concept of capacity.
        """
        return list()

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.

        Acceleration structures are encouraged to report:
        - 'occupancy' - a list of length [LODs], where each element contains the number of cells occupied in that LOD
        - 'capacity' - a list of length [LODs], where each element contains the total cell capacity in that LOD
        """
        return {
            '#Used Cells (LOD)': self.occupancy(),
            '#Capacity (LOD)': self.capacity()
        }
