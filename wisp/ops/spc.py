# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys

import torch
import numpy as np
import pandas as pd

import kaolin.ops.spc as spc_ops

import wisp.ops.mesh as mesh_ops
import wisp.ops.geometric as geo_ops

# A set of useful SPC utility functions. Will probably get folded into Kaolin...

def create_dense_octree(level):
    """Creates a dense SPC model

    Args:
        level (int): The level at which the octree will be initialized to.

    Returns:
        (torch.ByteTensor): the octree tensor
    """
    coords = np.arange(2**level)
    points = np.array(np.meshgrid(coords, coords, coords, indexing='xy'))
    points = points.transpose(3,2,1,0).reshape(-1, 3)
    points = torch.from_numpy(points).short().cuda()
    octree = spc_ops.unbatched_points_to_octree(points, level)
    return octree

def sample_from_depth_intervals(depth_intervals, num_samples):
    """Convert depth intervals to samples.

    SPC raytrace will return a [num_nuggets, 2] array where the first element is the entry depth 
    and the second element is the exit depth. This function will convert them into a
    [num_nuggets, num_samples, 3] array of samples.

    Args:
        depth_intervals (torch.FloatTensor): intervals of shape [num_nuggets, 2]
        num_samples (int): sample size
    
    Returns:
        (torch.FloatTensor): Samples of shape [num_nuggets, num_samples, 3]
    """
    device = depth_intervals.device
    steps = torch.arange(num_samples, device=device)[None].float().repeat([depth_intervals.shape[0], 1]) 
    steps += torch.rand_like(steps)
    steps *= (1.0 / num_samples)
    samples = depth_intervals[...,0:1] + (depth_intervals[...,1:2]-depth_intervals[...,0:1])*steps

    return samples

def expand_pack_boundary(pack_boundary, num_samples):
    """Expands the pack boundaries according to the number of samples.

    Args:
        pack_boundary (torch.BoolTensor): pack boundaries [N]
        num_samples (int): Number of samples

    Returns:
        (torch.BoolTensor): pack boundaries of shape [N*num_samples]
    """
    bigpack_boundary = torch.zeros(pack_boundary.shape[0]*num_samples, device=pack_boundary.device).bool()
    bigpack_boundary[pack_boundary.nonzero().long() * num_samples] = True
    bigpack_boundary = bigpack_boundary.int()
    return bigpack_boundary

def total_variation(pidx, trinkets, features, level):
    """Calculates total variation for the voxels specified by the pidx.

    Args:
        pidx : int tensor of size [N] specifying the point indices to calculate TV on.
        trinkets : the trinkets.
        features : the features for the given level. (assumes the correct level is given)
        level : int specifying the level of spc

    Returns:
        (torch.FloatTensor): Total variation of shape [1]
    """
    resolution = 2**level
    # N, 8, F tensor of features
    feats = features[trinkets.index_select(0, pidx).long()]

    # N, F
    diff_x = ((torch.abs(feats[:, [4,5,6,7]] - feats[:, [0,1,2,3]]) / resolution)**2).sum((1,2))
    diff_y = ((torch.abs(feats[:, [2,3,6,7]] - feats[:, [0,1,4,5]]) / resolution)**2).sum((1,2))
    diff_z = ((torch.abs(feats[:, [1,3,5,7]] - feats[:, [0,2,4,6]]) / resolution)**2).sum((1,2)) 
    return diff_x + diff_y + diff_z

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

def mesh_to_octree(vertices, faces, level):
    """Construct an octree from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree

    Returns:
        (torch.ByteTensor): the octree tensor
    """
    samples = mesh_ops.sample_surface(vertices.cuda(), faces.cuda(), 100000000)[0]
    # Augment samples... may be a hack that isn't actually needed
    samples = torch.cat([samples, 
        samples + (torch.rand_like(samples) * 2.0 - 1.0) * (1.0/(2**(level+1)))], dim=0)
    samples = spc_ops.quantize_points(samples, level)
    octree = spc_ops.unbatched_points_to_octree(samples, level)
    #octree = kaolin._C.ops.conversions.mesh_to_spc(self.V.cuda() * 0.5 + 0.5, self.F.cuda(), self.max_lod)
    return octree

def dilate_points(points, level):
    """Dilates the octree points.

    Args:
        points (torch.ShortTensor): The SPC points from some level
        level (int): The level from which the points come from

    Returns:
        (torch.ShortTensor): The dilated points
    """
    _x = torch.ShortTensor([[1,0,0]]).to(points.device)
    _y = torch.ShortTensor([[0,1,0]]).to(points.device)
    _z = torch.ShortTensor([[0,0,1]]).to(points.device)
    points = torch.cat([
        points + _x, points - _x,
        points + _y, points - _y,
        points + _z, points - _z,
        points + _x + _y, points + _x - _y, points + _x + _z, points + _x - _z,
        points + _y + _x, points + _y - _x, points + _y + _z, points + _y - _z,
        points + _z + _x, points + _z - _x, points + _z + _y, points + _z - _y,
        points + _x + _y + _z, points + _x + _y - _z, 
        points + _x - _y + _z, points + _x - _y - _z,
        points - _x + _y + _z, points - _x + _y - _z, 
        points - _x - _y + _z, points - _x - _y - _z, 
        ], dim=0)
    points = torch.clip(points, 0, 2**level-1)
    
    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0, 
            return_inverse=True, return_counts=True)
    
    morton, keys = torch.sort(spc_ops.points_to_morton(unique.contiguous()).contiguous())
    
    points = spc_ops.morton_to_points(morton.contiguous())
    
    return points

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
        att = att.index_add_(0, unique_keys, attributes) / unique_counts[...,None].float()
        att = att[keys]
        return octree, att

    return octree

def mesh_to_spc(vertices, faces, level):
    """Construct SPC from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree

    Returns:
        (torch.ByteTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor): 
        - the octree tensor
        - point hierarchy
        - pyramid
        - prefix 
    """
    octree = mesh_to_octree(vertices, faces, level)
    points, pyramid, prefix = octree_to_spc(octree)
    return octree, points, pyramid, prefix

def make_trilinear_spc(points, pyramid):
    """Builds a trilinear spc from a regular spc.

    Args:
        points (torch.ShortTensor): The point_hierarchy.
        pyramid (torch.LongTensor): The pyramid tensor.

    Returns:
        (torch.ShortTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor)
        - The dual point_hierarchy.
        - The dual pyramid.
        - The trinkets.
        - The parent pointers.
    """
    points_dual, pyramid_dual = spc_ops.unbatched_make_dual(points, pyramid)
    trinkets, parents = spc_ops.unbatched_make_trinkets(points, pyramid, points_dual, pyramid_dual)
    return points_dual, pyramid_dual, trinkets, parents

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

def compute_sdf_iou(nef, coords, gts, lod_idx=0):
    """Given a sparse SDF neural field, coordinates, and ground truth SDF, will calculate the IOU.

    In the case where the point does not exist in the bounds of the octree, will simply calculate 
    those as intersections.

    Inputs:
        nef (wisp.models.NeuralFields) : The neural field. Assumed to be sparse.
        coords (torch.FloatTensor) : 3D coordinates of shape [N, 3]
        gts (torch.FloatTensor) : Ground truth SDF of shape [N, 1]
        lod_idx : level of detail (if specified)

    Returns:
        (float): The calculated IOU
    """
    if nef.grid is None:
        raise Exception(f"{nef.__class__.__name__} is incompatible with this function.")
    pred = torch.zeros([coords.shape[0], 1])
    pidx = nef.grid.query(pts, lod_idx=lod_idx)
    mask = (pidx != -1)
    
    pred[mask] = nef(coords=pts[mask], pidx=pidx[mask], lod_idx=lod_idx, channels="sdf").cpu()
    pred[~mask] = gts[~mask]
    
    return geo_ops.compute_sdf_iou(pred, gts)
