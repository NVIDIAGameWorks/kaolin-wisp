# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from kaolin import _C
import wisp._C as wisp_C
import kaolin.ops.spc as spc_ops

PRIMES = [1, 2654435761, 805459861]

def hashgrid_naive(coords, resolutions, codebook_bitwidth, lod_idx, codebook, codebook_lod_sizes, codebook_lod_first_idx):
    """
    A naive PyTorch implementation of the hashgrid.
    This code exists here mostly as a reference:
    Do NOT expect a 1-to-1 numerical correspondence to the CUDA accelerated version.
    This code is comparatively very slow. :)

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.FloatTensor): A tensor containing the stacked codebooks, each of shape [codebook_size_lod_idx, feature_dim].
        codebook_lod_sizes (torch.IntTensor): A tensor containig the codebook size at each level of detail.
        codebook_lod_first_idx (torch.IntTensor): A tensor containing the first index of each codebook in the stacked codebook tensor.

    Returns:
        (torch.FloatTensor): Features of shape [batch*num_samples, feature_dim]
    """
    codebook_size = 2**codebook_bitwidth

    feats = []
    for i, res in enumerate(resolutions[:lod_idx+1]):
        # This assumes that the input coordinates are in the range [0, 1].
        tf_coords = torch.clip(((coords + 1.0) / 2.0) * res, 0, res-1-1e-5).reshape(-1, 3)
        cc000 = torch.floor(tf_coords).short()
        cc = spc_ops.points_to_corners(cc000).long()

        num_pts = res**3
        if num_pts > codebook_size:
            cidx = (
                    (cc[...,0] * PRIMES[0]) ^ (cc[...,1] * PRIMES[1]) ^ (cc[...,2] * PRIMES[2])
                ) % codebook_size
        else:
            cidx = cc[...,0] + cc[...,1] * res + cc[...,2] * res * res
        # cidx: B, 8

        fs = codebook[codebook_lod_first_idx[i] : codebook_lod_first_idx[i] + codebook_lod_sizes[i]][cidx.reshape(-1)]  # B*8, F
        fs = fs.reshape(-1, 8, fs.shape[-1])  # B, 8, F

        coeffs = torch.zeros(coords.size(0), 8, device=coords.device, dtype=coords.dtype)  # B, 8
        x = tf_coords - cc000
        _x = 1.0 - x

        # Trilinear interpolation
        coeffs[...,0] = _x[...,0] * _x[...,1] * _x[...,2]
        coeffs[...,1] = _x[...,0] * _x[...,1] * x[...,2]
        coeffs[...,2] = _x[...,0] * x[...,1] * _x[...,2]
        coeffs[...,3] = _x[...,0] * x[...,1] * x[...,2]
        coeffs[...,4] = x[...,0] * _x[...,1] * _x[...,2]
        coeffs[...,5] = x[...,0] * _x[...,1] * x[...,2]
        coeffs[...,6] = x[...,0] * x[...,1] * _x[...,2]
        coeffs[...,7] = x[...,0] * x[...,1] * x[...,2]
        coeffs = coeffs.reshape(-1, 8, 1)  # B, 8, 1

        fs_coeffs = (fs * coeffs).sum(1)  # B, F
        feats.append(fs_coeffs)

    # TODO(ttakikawa): This probably does not return according to the num_samples interface
    return torch.cat(feats, -1)  # B, F*L

class HashGridInterpolate(torch.autograd.Function):
    # TODO(ttakikawa): This class should also support the 2D case... which also means I have to write another kernel!

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, coords, resolutions, codebook_bitwidth, lod_idx, codebook, codebook_first_idx):
        if codebook[0].shape[-1] % 2 == 1:
            raise Exception("The codebook feature dimension needs to be a multiple of 2.")

        assert(coords.shape[-1] in [2, 3])

        if torch.is_autocast_enabled():
            codebook = codebook.half()

        # TODO(ttakikawa): Make the kernel use the LOD
        feats_out = wisp_C.ops.hashgrid_interpolate_cuda(coords.contiguous(), 
                                                         codebook,
                                                         codebook_first_idx,
                                                         resolutions,
                                                         codebook_bitwidth).contiguous()
    
        ctx.save_for_backward(coords, codebook, codebook_first_idx)
        ctx.resolutions = resolutions
        ctx.num_lods = len(resolutions)
        ctx.codebook_size = 2**codebook_bitwidth
        ctx.codebook_bitwidth = codebook_bitwidth
        ctx.feature_dim = codebook.shape[-1]
        return feats_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        coords = ctx.saved_tensors[0]
        codebook = ctx.saved_tensors[1]
        codebook_first_idx = ctx.saved_tensors[2]
        resolutions = ctx.resolutions
        feature_dim = ctx.feature_dim
        codebook_bitwidth = ctx.codebook_bitwidth
        coords_requires_grad = ctx.needs_input_grad[0]

        grad_coords, grad_codebook = wisp_C.ops.hashgrid_interpolate_backward_cuda(
                coords.float().contiguous(), grad_output.contiguous(), codebook,
                codebook_first_idx,
                resolutions,  
                codebook_bitwidth, feature_dim, coords_requires_grad)
        
        if coords_requires_grad:
            return (grad_coords, None, None, None, grad_codebook, None, None)
        else:
            return (None, None, None, None, grad_codebook, None, None) 

def hashgrid(coords, codebook_bitwidth, lod_idx, codebook):
    """A hash-grid query + interpolation function, accelerated with CUDA.

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, 3]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (wisp.models.grids.utils.MultiTable): A class that holds multiresolution tables.

    Returns:
        (torch.FloatTensor): Features of shape [batch, feature_dim]
    """
    batch, dim = coords.shape
    feats = HashGridInterpolate.apply(coords.contiguous(), codebook.resolutions,
                                      codebook_bitwidth, lod_idx, codebook.feats, codebook.begin_idxes)
    feature_dim = codebook.feats.shape[1] * len(codebook.resolutions)
    return feats.reshape(batch, feature_dim)

class GridInterpolate(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(ctx, coords, feats):
        feats_out = wisp_C.ops.grid_interpolate_cuda(coords.float().contiguous(), 
                                                     feats.contiguous()).contiguous()
        ctx.save_for_backward(coords)
        ctx.feature_dim = feats.shape[-1]
        return feats_out
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def backward(ctx, grad_output):
        coords = ctx.saved_tensors[0]
        feature_dim = ctx.feature_dim
        
        grad_feats = wisp_C.ops.grid_interpolate_backward_cuda(
                coords.float().contiguous(), grad_output.contiguous(), feature_dim)
        return (None, grad_feats)
        
def grid_interpolate(coords, feats):
    return GridInterpolate.apply(coords.contiguous(), feats.contiguous())

class HashGridQuery(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, coords, resolutions, codebook_bitwidth, probe_bitwidth, lod_idx, *codebook):
        if codebook[0].shape[-1] % 2 == 1:
            raise Exception("The codebook feature dimension needs to be a multiple of 2.")
        
        # TODO(ttakikawa): Make the kernel use the LOD
        feats_out = wisp_C.ops.hashgrid_query_cuda(coords.float().contiguous(), 
                                                     codebook,
                                                     resolutions,
                                                     codebook_bitwidth,
                                                     probe_bitwidth).contiguous()
    
        ctx.save_for_backward(coords)
        ctx.resolutions = resolutions
        ctx.num_lods = len(resolutions)
        ctx.codebook_shapes = [_c.shape for _c in codebook]
        ctx.codebook_size = 2**codebook_bitwidth
        ctx.codebook_bitwidth = codebook_bitwidth
        ctx.feature_dim = codebook[0].shape[-1]
        ctx.probe_bitwidth = probe_bitwidth
        return feats_out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        coords = ctx.saved_tensors[0]
        resolutions = ctx.resolutions
        codebook_size = ctx.codebook_size
        feature_dim = ctx.feature_dim
        codebook_shapes = ctx.codebook_shapes
        codebook_bitwidth = ctx.codebook_bitwidth
        probe_bitwidth = ctx.probe_bitwidth

        grad_codebook = wisp_C.ops.hashgrid_query_backward_cuda(
                coords.float().contiguous(), grad_output.contiguous(), 
                resolutions, [c_[0] for c_ in codebook_shapes], 
                codebook_bitwidth, feature_dim, probe_bitwidth)
        return (None, None, None, None, None, *grad_codebook)

def hashgrid_query_fwd(coords, resolutions, codebook_bitwidth, lod_idx, codebook, probe_bitwidth=0):
    """Non-differentiable version of hashgrid query. 

    No assumptions on the typing of the codebook.
    """
    batch, dim = coords.shape
    assert(coords.shape[-1] in [2, 3])
    feats_out = wisp_C.ops.hashgrid_query_cuda(coords.float().contiguous(), 
                                               codebook,
                                               resolutions,
                                               codebook_bitwidth,
                                               probe_bitwidth).contiguous()
    feature_dim = codebook[0].shape[1] * len(resolutions)
    return feats_out.reshape(batch, 8, feature_dim*(2**probe_bitwidth))

def hashgrid_query(coords, resolutions, codebook_bitwidth, lod_idx, codebook, probe_bitwidth=0):
    """A hash-grid query, accelerated with CUDA.
    
    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.ModuleList[torch.FloatTensor]): A list of codebooks of shapes [codebook_size, feature_dim].

    Returns:
        (torch.FloatTensor): Features of shape [batch, 8, feature_dim]
    """
    batch, dim = coords.shape
    assert(coords.shape[-1] in [2, 3])
    feats = HashGridQuery.apply(coords.contiguous(), resolutions,
                                codebook_bitwidth, probe_bitwidth, lod_idx, *[_c for _c in codebook])
    feature_dim = codebook[0].shape[1] * len(resolutions)
    return feats.reshape(batch, 8, feature_dim*(2**probe_bitwidth))

