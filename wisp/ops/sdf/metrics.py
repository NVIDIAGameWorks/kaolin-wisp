# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


def compute_sdf_iou(pred, gts):
    """Compute intersection over union for SDFs.

    Args:
        pred (torch.FloatTensor): Predicted signed distances
        gts (torch.FloatTensor): Groundtruth signed distances

    Returns:
        (float): The IOU score between 0 and 100.
    """
    inside_pred = (pred < 0).byte()
    inside_gts = (gts < 0).byte()

    area_union = torch.sum((inside_pred | inside_gts).float()).item()
    area_intersect = torch.sum((inside_pred & inside_gts).float()).item()

    iou = area_intersect / area_union
    return 100.0 * iou


def compute_sparse_sdf_iou(nef, coords, gts, lod_idx=0):
    """Given a sparse SDF neural field, coordinates, and ground truth SDF, will calculate the narrowband IOU.

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
    query_results = nef.grid.query(gts, lod_idx=lod_idx)
    pidx = query_results.pidx
    mask = (pidx != -1)

    pred[mask] = nef(coords=gts[mask], pidx=pidx[mask], lod_idx=lod_idx, channels="sdf").cpu()
    pred[~mask] = gts[~mask]

    return compute_sdf_iou(pred, gts)
