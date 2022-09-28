# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import wisp.ops.mesh as mesh_ops
from wisp.utils import PsDebugger, PerfTimer
import wisp.ops.spc as wisp_spc_ops
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render


class OctreeAS(object):
    """Octree bottom-level acceleration structure class implemented using Kaolin SPC.
       Can be used to to quickly query cells occupancy, and trace rays against the volume.
    """
    
    def __init__(self):
        self.initialized = False

    def init_from_mesh(self, mesh_path, level, sample_tex=False, num_samples=100000000):
        """Builds the grid from a path to the mesh.

        Only supports OBJ for now.

        Args:
            mesh_path (str): Path to the OBJ file.
            level (int): The depth of the octree.
            sample_tex (bool): If True, will also sample textures and store it in the accelstruct.
            num_samples (int): The number of samples to be generated on the mesh surface.

        Returns:
            (void): Will initialize the OctreeAS object.
        """

        if sample_tex:
            out = mesh_ops.load_obj(mesh_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = mesh_ops.load_obj(mesh_path)

        # For now only supports sphere normalization, which is a bit more robust for SDF type workloads
        # (although it will underutilize the voxels)
        self.V, self.F = mesh_ops.normalize(self.V, self.F, 'sphere')

        # Note: This function is not deterministic since it relies on sampling.
        #       Eventually this will be replaced by 3D rasterization.
        octree = wisp_spc_ops.mesh_to_octree(self.V, self.F, level, num_samples)
        self.init(octree)
    
    def init_from_pointcloud(self, pointcloud, level):
        """Builds the grid from a pointcloud.

        Args:
            pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords, 3] in 
                                            normalized space [-1, 1].
            level (int): The depth of the octree.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        octree = wisp_spc_ops.pointcloud_to_octree(pointcloud, level, dilate=0)
        self.init(octree)

    def init_dense(self, level):
        """Builds a dense octree grid.

        Args:
            level (int): The depth of the octree.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        octree = wisp_spc_ops.create_dense_octree(level)
        self.init(octree)

    def init_aabb(self):
        """Builds a root-only octree.

        Useful for hacking together a quick AABB tracer.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        octree = wisp_spc_ops.create_dense_octree(1)
        self.init(octree)

    def init(self, octree):
        """Initializes auxillary state from an octree tensor.

        Args:
            octree (torch.ByteTensor): SPC octree tensor.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        self.octree = octree
        self.points, self.pyramid, self.prefix = wisp_spc_ops.octree_to_spc(self.octree)
        self.initialized = True
        self.max_level = self.pyramid.shape[-1] - 2
    
    def query(self, coords, level=None, with_parents=False):
        """Returns the ``pidx`` for the sample coordinates.

        Args:
            coords (torch.FloatTensor) : 3D coordinates of shape [num_coords, 3] in normalized [-1, 1] space.
            level (int) : The depth of the octree to query. If None, queries the highest level.
            with_parents (bool) : If True, also returns hierarchical parent indices.
        
        Returns:
            (torch.LongTensor): The indices into the point hierarchy of shape [num_query].
                If with_parents is True, then the shape will be [num_query, level+1].
        """
        if level is None:
            level = self.max_level
        
        return spc_ops.unbatched_query(self.octree, self.prefix, coords, level, with_parents)

    def raytrace(self, rays, level=None, with_exit=False):
        """Traces rays against the SPC structure, returning all intersections along the ray with the SPC points
        (SPC points are quantized, and can be interpreted as octree cell centers or corners).

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            level (int) : The level of the octree to raytrace. If None, traces the highest level.
            with_exit (bool) : If True, also returns exit depth.

        Returns:
            (torch.LongTensor, torch.LongTensor, torch.FloatTensor):
                - Indices into rays.origins and rays.dirs of shape [num_intersections]
                - Indices into the point_hierarchy of shape [num_intersections]
                - Depths of shape [num_intersections, 1 or 2]
        """
        if level is None:
            level = self.max_level

        ridx, pidx, depth = spc_render.unbatched_raytrace(
                self.octree, self.points, self.pyramid, self.prefix,
                rays.origins, rays.dirs, level, return_depth=True, with_exit=with_exit)
        return ridx, pidx, depth

    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Samples points along the ray inside the SPC structure.

        TODO(ttakikawa): Maybe separate the sampling logic from the raymarch logic. 
                         Haven't decided yet if this inits sense.

        TODO(ttakikawa): Do the returned shapes actually init sense? Reevaluate.

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            level (int) : The level of the octree to raytrace. If None, traces the highest level.
            num_samples (int) : Number of samples per voxel
        
        Returns:
            (torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor):
                - Indices into rays.origins and rays.dirs of shape [num_intersections]
                - Indices into the point_hierarchy of shape [num_intersections]
                - Sample coordinates of shape [num_intersections, num_samples, 3]
                - Sample depths of shape [num_intersections*num_samples, 1]
                - Sample depth diffs of shape [num_intersections*num_samples, 1]
                - Boundary tensor which marks the beginning of each variable-sized sample pack of shape [num_intersections*num_samples]
        """
        timer = PerfTimer(activate=False, show_memory=False)
        if level is None:
            level = self.max_level

        # Samples points along the rays by first tracing it against the SPC object.
        # Then, given each SPC voxel hit, will sample some number of samples in each voxel.
        # This setting is pretty nice for getting decent outputs from outside-looking-in scenes, 
        # but in general it's not very robust or proper since the ray samples will be weirdly distributed
        # and or aliased. 
        if raymarch_type == 'voxel':
            ridx, pidx, depth = self.raytrace(rays, level, with_exit=True)
            
            timer.check("raytrace")

            depth_samples = wisp_spc_ops.sample_from_depth_intervals(depth, num_samples)[...,None]
            deltas = depth_samples[...,0].diff(dim=-1, prepend=depth[...,0:1]).reshape(-1, 1)
            timer.check("sample depth")

            samples = torch.addcmul(rays.origins.index_select(0, ridx)[:,None], 
                                    rays.dirs.index_select(0, ridx)[:,None], depth_samples)
            timer.check("generate samples coords")
            
            boundary = wisp_spc_ops.expand_pack_boundary(spc_render.mark_first_hit(ridx.int()), num_samples)
            #deltas = spc_render.diff(depth_samples, boundary).reshape(-1, 1)

        # Samples points along the rays, and then uses the SPC object the filter out samples that don't hit
        # the SPC objects. This is a much more well-spaced-out sampling scheme and will work well for 
        # inside-looking-out scenes. The camera near and far planes will have to be adjusted carefully, however.
        elif raymarch_type == 'ray':
            # Sample points along 1D line
            depth = torch.linspace(0, 1.0, num_samples, device=rays.origins.device)[None] + \
                    (torch.rand(rays.origins.shape[0], num_samples, device=rays.origins.device)/num_samples)
            #depth = torch.linspace(0, 1.0, num_samples, device=rays.origins.device) + \
            #        (torch.rand(num_samples, rays.origins.device)/num_samples)
            depth = depth ** 2
            
            # Normalize between near and far plane
            depth *= (rays.dist_max - rays.dist_min)
            depth += rays.dist_min

            # Batched generation of samples
            #samples = rays.origins[:, None] + rays.dirs[:, None] * depth[None, :, None]
            samples = rays.origins[:, None] + rays.dirs[:, None] * depth[..., None]
            deltas = depth.diff(dim=-1, prepend=(torch.zeros(depth.shape[0], 1, device=depth.device)+ rays.dist_min))
            # Hack together pidx, mask, ridx, boundaries, etc
            pidx = self.query(samples.reshape(-1, 3), level=level).reshape(-1, num_samples)
            mask = (pidx > -1)
            ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
            ridx = ridx[...,None].repeat(1, num_samples)[mask]
            boundary = spc_render.mark_pack_boundaries(ridx)
            pidx = pidx[mask]
            #depth_samples = depth[None].repeat(rays.origins.shape[0], 1)[mask][..., None]
            depth_samples = depth[mask][..., None]
            
            #deltas = spc_render.diff(depth_samples, boundary).reshape(-1, 1) 
            deltas = deltas[mask].reshape(-1, 1)

            samples = samples[mask][:,None]
        else:
            raise TypeError(f"raymarch type {raymarch_type} is wrong")

        return ridx, pidx, samples, depth_samples, deltas, boundary
