# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import kaolin.ops.spc as spc_ops
import wisp.ops.mesh as mesh_ops
from wisp.ops.spc.processing import dilate_points


def pointcloud_to_octree(pointcloud, level, attributes=None, dilate=0):
    """Converts floating point coordinates to an octree.

    Args:
        pointcloud (torch.FloatTensor): 3D coordinates in [-1, 1] of shape [N, 3]
        level (int): Depth of the octreee
        attributes (torch.FloatTensor): Attributes of shape [N, F]. Will be averaged within voxels.
        dilate (int): Dilates the octree if specified.

    Returns:
        (torch.ByteTensor) or (torch.ByteTensor, torch.FloatTensor)
        - octree tensor
        - the averaged attributes
    """
    points = spc_ops.quantize_points(pointcloud.contiguous().cuda(), level)

    for i in range(dilate):
        points = dilate_points(points, level)

    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                      return_inverse=True, return_counts=True)

    morton, keys = torch.sort(spc_ops.points_to_morton(unique.contiguous()).contiguous())

    points = spc_ops.morton_to_points(morton.contiguous())
    octree = spc_ops.unbatched_points_to_octree(points, level, sorted=True)

    if attributes is not None:
        att = torch.zeros_like(unique).float()
        att = att.index_add_(0, unique_keys, attributes) / unique_counts[... ,None].float()
        att = att[keys]
        return octree, att

    return octree


def mesh_to_spc(vertices, faces, level, num_samples=100000000):
    """Construct SPC from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree
        num_samples (int): The number of samples to be generated on the mesh surface.

    Returns:
        (torch.ByteTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor):
        - the octree tensor
        - point hierarchy
        - pyramid
        - prefix
    """
    octree = mesh_to_octree(vertices, faces, level, num_samples)
    points, pyramid, prefix = octree_to_spc(octree)
    return octree, points, pyramid, prefix


def octree_to_spc(octree):
    """Convenience function to generate the usual SPC data.

    Args:
        octree (torch.ByteTensor): SPC octree

    Returns:
        (torch.LongTensor, torch.LongTensor, torch.BoolTensor)
        - point hierarcy
        - pyramid
        - prefix
    """
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc_ops.scan_octrees(octree, lengths)
    points = spc_ops.generate_points(octree, pyramid, prefix)
    pyramid = pyramid[0]
    return points, pyramid, prefix


def mesh_to_octree(vertices, faces, level, num_samples=100000000):
    """Construct an octree from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree
        num_samples (int): The number of samples to be generated on the mesh surface.

    Returns:
        (torch.ByteTensor): the octree tensor
    """
    samples = mesh_ops.sample_surface(vertices.cuda(), faces.cuda(), num_samples)[0]
    # Augment samples... may be a hack that isn't actually needed
    samples = torch.cat([samples,
        samples + (torch.rand_like(samples) * 2.0 - 1.0) * (1.0/(2**(level+1)))], dim=0)
    samples = spc_ops.quantize_points(samples, level)
    octree = spc_ops.unbatched_points_to_octree(samples, level)
    return octree



def mesh_to_trilinear_spc(vertices, faces, level):
    """Builds a trilinear spc from a regular spc.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree

    Returns:
        (torch.ByteTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor,
         torch.ShortTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor)
        - the octree tensor
        - point hierarchy
        - pyramid
        - prefix
        - The dual point_hierarchy.
        - The dual pyramid.
        - The trinkets.
        - The parent pointers.
    """
    """Builds a trilinear spc from a mesh"""
    octree, points, pyramid, prefix = mesh_to_spc(vertices, faces, level)
    points_dual, pyramid_dual, trinkets, parents = build_trilinear_spc(points, pyramid)
    return octree, points, pyramid, prefix, points_dual, pyramid_dual, trinkets, parents
