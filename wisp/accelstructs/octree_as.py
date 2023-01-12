# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import List
import torch
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render
import wisp.ops.mesh as mesh_ops
import wisp.ops.spc as wisp_spc_ops
from wisp.accelstructs.base_as import BaseAS, ASQueryResults, ASRaytraceResults, ASRaymarchResults


class OctreeAS(BaseAS):
    """Octree bottom-level acceleration structure class implemented using Kaolin SPC.
    Can be used to to quickly query cells occupancy, and trace rays against the volume.
    """
    
    def __init__(self, octree):
        """Initializes the acceleration structure from the topology of a sparse octree (Structured Point Cloud).
        Structured Point Cloud (SPC) is a compact data structure for organizing and efficiently pack sparse 3D geometric
        information.
        Intuitively, SPCs can also be described as sparse voxel-grids, quantized point clouds, or
        voxelized point clouds.

        Args:
            octree (torch.ByteTensor): SPC octree tensor, containing the acceleration structure topology.
            For more details about this format, see:
             https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
        """
        super().__init__()
        self.octree = octree
        self.points, self.pyramid, self.prefix = wisp_spc_ops.octree_to_spc(octree)
        self.max_level = self.pyramid.shape[-1] - 2
        self.extent = dict()    # Additional optional information which may be stored in this struct

    @classmethod
    def from_mesh(cls, mesh_path, level, sample_tex=False, num_samples=100000000) -> OctreeAS:
        """ Builds the acceleration structure and initializes occupancy of cells from samples over mesh faces.
        Assumes a path to the mesh, only OBJ is supported for now.

        Args:
            mesh_path (str): Path to the OBJ file.
            level (int): The depth of the octree.
            sample_tex (bool): If True, will also sample textures and store it in the accelstruct within
                self.texv, self.texf, self.mats fields.
                This feature is currently unused.
            num_samples (int): The number of samples to be generated on the mesh surface.
                More samples require additional processing time but are more likely to produce faithful occupancy
                output without "holes".
        """
        if sample_tex:
            out = mesh_ops.load_obj(mesh_path, load_materials=True)
            vertices, faces, texture_vertices, texture_faces, materials = out
        else:
            vertices, faces = mesh_ops.load_obj(mesh_path)

        # For now only supports sphere normalization, which is a bit more robust for SDF type workloads
        # (although it will underutilize the voxels)
        vertices, faces = mesh_ops.normalize(vertices, faces, 'sphere')

        # Note: This function is not deterministic since it relies on sampling.
        # Eventually this will be replaced by 3D rasterization.
        octree = wisp_spc_ops.mesh_to_octree(vertices, faces, level, num_samples)
        accel_struct = OctreeAS(octree)
        accel_struct.extent['vertices'] = vertices
        accel_struct.extent['faces'] = faces
        if sample_tex:
            accel_struct.extent['texv'] = texture_vertices
            accel_struct.extent['texf'] = texture_faces
            accel_struct.extent['mats'] = materials
        return accel_struct

    @classmethod
    def from_pointcloud(cls, pointcloud, level) -> OctreeAS:
        """ Builds the acceleration structure and initializes occupancy of cells from a pointcloud.

        Args:
            pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords, 3] in 
                                            normalized space [-1, 1].
            level (int): The depth of the octree.
        """
        octree = wisp_spc_ops.pointcloud_to_octree(pointcloud, level, dilate=0)
        return OctreeAS(octree)

    @classmethod
    def from_quantized_points(cls, quantized_points, level) -> OctreeAS:
        """ Builds the acceleration structure from quantized (integer) point coordinates.

        Args:
            quantized_points (torch.LongTensor): 3D coordinates of shape [num_coords, 3] in
                                                 integer coordinate space [0, 2**level]
            level (int): The depth of the octree.
        """
        octree = spc_ops.unbatched_points_to_octree(quantized_points, level, sorted=False)
        return OctreeAS(octree)

    @classmethod
    def make_dense(cls, level) -> OctreeAS:
        """ Builds the acceleration structure and initializes full occupancy of all cells.

        Args:
            level (int): The depth of the octree.
        """
        octree = wisp_spc_ops.create_dense_octree(level)
        return OctreeAS(octree)

    def query(self, coords, level=None, with_parents=False) -> ASQueryResults:
        """Returns the ``pidx`` for the sample coordinates (indices of acceleration structure cells returned by
        this query).

        Args:
            coords (torch.FloatTensor) : 3D coordinates of shape [num_coords, 3] in normalized [-1, 1] space.
            level (int) : The depth of the octree to query. If None, queries the highest level.
            with_parents (bool) : If True, also returns hierarchical parent indices.
        
        Returns:
            (ASQueryResults): containing the indices into the point hierarchy of shape [num_query].
                If with_parents is True, then the query result will be of shape [num_query, level+1].
        """
        if level is None:
            level = self.max_level
        
        pidx = spc_ops.unbatched_query(self.octree, self.prefix, coords, level, with_parents)
        return ASQueryResults(pidx=pidx)

    def raytrace(self, rays, level=None, with_exit=False) -> ASRaytraceResults:
        """Traces rays against the SPC structure, returning all intersections along the ray with the SPC points
        (SPC points are quantized, and can be interpreted as octree cell centers or corners).

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            level (int) : The level of the octree to raytrace. If None, traces the highest level.
            with_exit (bool) : If True, also returns exit depth.

        Returns:
            (ASRaytraceResults): with fields containing -
                - Indices into rays.origins and rays.dirs of shape [num_intersections]
                - Indices into the point_hierarchy of shape [num_intersections]
                - Depths of shape [num_intersections, 1 or 2]
        """
        if level is None:
            level = self.max_level

        ridx, pidx, depth = spc_render.unbatched_raytrace(
            self.octree, self.points, self.pyramid, self.prefix,
            rays.origins, rays.dirs, level, return_depth=True, with_exit=with_exit)
        return ASRaytraceResults(ridx=ridx, pidx=pidx, depth=depth)

    def _raymarch_voxel(self, rays, num_samples, level=None) -> ASRaymarchResults:
        """Samples points along the ray inside the SPC structure.
        Raymarch is achieved by intersecting the rays with the SPC cells.
        Then among the intersected cells, each cell is sampled num_samples times.
        In this scheme, num_hit_samples <= num_intersections*num_samples

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            num_samples (int) : Number of samples generated per voxel. The total number of samples generated will
                also depend on the number of cells a ray have intersected.
            level (int) : The level of the octree to raytrace. If None, traces the highest level.

        Returns:
            (ASRaymarchResults) with fields containing:
                - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
                - Sample coordinates of shape [num_hit_samples, 3]
                - Sample depths of shape [num_hit_samples, 1]
                - Sample depth diffs of shape [num_hit_samples, 1]
                - Boundary tensor which marks the beginning of each variable-sized
                  sample pack of shape [num_hit_samples]
        """
        # NUM_INTERSECTIONS = number of nuggets: ray / cell intersections
        # NUM_INTERSECTIONS can be 0!
        # ridx, pidx ~ (NUM_INTERSECTIONS,)
        # depth ~ (NUM_INTERSECTIONS, 2)
        raytrace_results = self.raytrace(rays, level, with_exit=True)
        ridx = raytrace_results.ridx.long()
        num_intersections = ridx.shape[0]

        # depth_samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
        depth = raytrace_results.depth
        depth_samples = wisp_spc_ops.sample_from_depth_intervals(depth, num_samples)[..., None]
        deltas = depth_samples[..., 0].diff(dim=-1, prepend=depth[..., 0:1]).reshape(num_intersections * num_samples, 1)

        # samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
        samples = torch.addcmul(rays.origins.index_select(0, ridx)[:, None],
                                rays.dirs.index_select(0, ridx)[:, None], depth_samples)

        # boundary ~ (NUM_INTERSECTIONS * NUM_SAMPLES,)
        # (each intersected cell is sampled NUM_SAMPLES times)
        boundary = wisp_spc_ops.expand_pack_boundary(spc_render.mark_first_hit(ridx.int()), num_samples)

        # ridx ~ (NUM_INTERSECTIONS * NUM_SAMPLES,)
        # samples ~ (NUM_INTERSECTIONS * NUM_SAMPLES, 3)
        # depth_samples ~ (NUM_INTERSECTIONS * NUM_SAMPLES, 1)
        ridx = ridx[:,None].expand(num_intersections, num_samples).reshape(num_intersections*num_samples)
        samples = samples.reshape(num_intersections * num_samples, 3)
        depth_samples = depth_samples.reshape(num_intersections * num_samples, 1)

        return ASRaymarchResults(
            ridx=ridx,
            samples=samples,
            depth_samples=depth_samples,
            deltas=deltas,
            boundary=boundary
        )

    def _raymarch_ray(self, rays, num_samples, level=None) -> ASRaymarchResults:
        """Samples points along the ray inside the SPC structure.
        Raymarch is achieved by sampling num_samples along each ray,
        and then filtering out samples which falls outside of occupied cells.
        In this scheme, num_hit_samples <= num_rays * num_samples.
        Ray boundaries are determined by the ray dist_min / dist_max values
        (which could, for example, be set by the near / far planes).

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            num_samples (int) : Number of samples generated per ray. The actual number of generated samples may be lower
                due to samples intersecting empty cells.
            level (int) : The level of the octree to raytrace. If None, traces the highest level.

        Returns:
            (ASRaymarchResults) with fields containing:
                - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
                - Sample coordinates of shape [num_hit_samples, 3]
                - Sample depths of shape [num_hit_samples, 1]
                - Sample depth diffs of shape [num_hit_samples, 1]
                - Boundary tensor which marks the beginning of each variable-sized
                  sample pack of shape [num_hit_samples]
        """
        # Sample points along 1D line
        # depth ~ (NUM_RAYS, NUM_SAMPLES)
        depth = torch.linspace(0, 1.0, num_samples, device=rays.origins.device)[None] + \
                (torch.rand(rays.origins.shape[0], num_samples, device=rays.origins.device) / num_samples)

        # Normalize between near and far plane
        depth *= (rays.dist_max - rays.dist_min)
        depth += rays.dist_min

        # Batched generation of samples
        # samples ~ (NUM_RAYS, NUM_SAMPLES, 3)
        # deltas, pidx, mask ~ (NUM_RAYS, NUM_SAMPLES)
        num_rays = rays.shape[0]
        samples = torch.addcmul(rays.origins[:, None], rays.dirs[:, None], depth[..., None])
        deltas = depth.diff(dim=-1,
                            prepend=(torch.zeros(rays.origins.shape[0], 1, device=depth.device) + rays.dist_min))
        query_results = self.query(samples.reshape(num_rays * num_samples, 3), level=level)
        pidx = query_results.pidx
        pidx = pidx.reshape(num_rays, num_samples)
        mask = pidx > -1

        # NUM_HIT_SAMPLES: number of samples sampled within occupied cells
        # NUM_HIT_SAMPLES can be 0!
        # depth_samples, deltas, ridx, boundary ~ (NUM_HIT_SAMPLES,)
        # samples ~ (NUM_HIT_SAMPLES, 3)
        depth_samples = depth[mask][:, None]
        num_hit_samples = depth_samples.shape[0]
        deltas = deltas[mask].reshape(num_hit_samples, 1)
        samples = samples[mask]
        ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
        ridx = ridx[..., None].repeat(1, num_samples)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)

        return ASRaymarchResults(
            ridx=ridx,
            samples=samples,
            depth_samples=depth_samples,
            deltas=deltas,
            boundary=boundary
        )

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Samples points along the ray inside the SPC structure.
        The exact algorithm employed for raymarching is determined by `raymarch_type`.

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            raymarch_type (str): Sampling strategy to use for raymarch.
                'voxel' - intersects the rays with the SPC cells. Then among the intersected cells, each cell
                    is sampled num_samples times.
                    In this scheme, num_hit_samples <= num_intersections*num_samples
                'ray' - samples num_samples along each ray, and then filters out samples which falls outside of occupied
                    cells.
                    In this scheme, num_hit_samples <= num_rays * num_samples
            num_samples (int) : Number of samples generated per voxel or ray. The exact meaning of this arg depends on
                the value of `raymarch_type`.
            level (int) : The level of the octree to raytrace. If None, traces the highest level.
        
        Returns:
            (ASRaymarchResults) with fields containing:
                - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
                - Sample coordinates of shape [num_hit_samples, 3]
                - Sample depths of shape [num_hit_samples, 1]
                - Sample depth diffs of shape [num_hit_samples, 1]
                - Boundary tensor which marks the beginning of each variable-sized
                  sample pack of shape [num_hit_samples]
        """
        if level is None:
            level = self.max_level

        # Samples points along the rays by first tracing it against the SPC object.
        # Then, given each SPC voxel hit, will sample some number of samples in each voxel.
        # This setting is pretty nice for getting decent outputs from outside-looking-in scenes, 
        # but in general it's not very robust or proper since the ray samples will be weirdly distributed
        # and or aliased. 
        if raymarch_type == 'voxel':
            raymarch_results = self._raymarch_voxel(rays=rays, num_samples=num_samples, level=level)

        # Samples points along the rays, and then uses the SPC object the filter out samples that don't hit
        # the SPC objects. This is a much more well-spaced-out sampling scheme and will work well for 
        # inside-looking-out scenes. The camera near and far planes will have to be adjusted carefully, however.
        elif raymarch_type == 'ray':
            raymarch_results = self._raymarch_ray(rays=rays, num_samples=num_samples, level=level)

        else:
            raise TypeError(f"Raymarch sampler type: {raymarch_type} is not supported by OctreeAS.")

        return raymarch_results

    def occupancy(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the number of cells occupied in that LOD """
        return self.pyramid[0, :-2].cpu().numpy().tolist()

    def capacity(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the total cell capacity in that LOD """
        return [8**lod for lod in range(self.max_level)]

    def occupancy(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the number of cells occupied in that LOD """
        return self.pyramid[0, :-2].cpu().numpy()

    def capacity(self) -> List[int]:
        """ Returns a list of length [LODs], where each element contains the total cell capacity in that LOD """
        return [8**lod for lod in range(self.max_level)]

    def name(self) -> str:
        return "Octree"
