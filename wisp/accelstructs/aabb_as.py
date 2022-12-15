# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from wisp.accelstructs.octree_as import OctreeAS
import wisp.ops.spc as wisp_spc_ops


class AxisAlignedBBoxAS(OctreeAS):
    """Axis Aligned Bounding Box, as a bottom-level acceleration structure class.
    Can be used to to quickly query cells occupancy, and trace rays against the volume.
    """
    
    def __init__(self):
        """Initializes a simple acceleration structure of an AABB (axis aligned bounding box). """
        # Builds a root-only octree, of one level, which is essentially a bounding box.
        # Useful for hacking together a quick AABB (axis aligned bounding box) tracer.
        octree = wisp_spc_ops.create_dense_octree(1)
        super().__init__(octree)

    def name(self) -> str:
        return "AABB"
