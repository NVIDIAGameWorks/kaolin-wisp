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

# Alternative set of primes
#PRIMES = [2654436881, 5915587277, 1500450271, 3267000013, 5754853343, 
#          4093082899, 9576890767, 3628273133, 2860486313, 5463458053, 
#          3367900313, 5654500741, 5654500763, 5654500771, 5654500783, 
#          5654500801, 5654500811, 5654500861, 5654500879, 5654500889, 
#          5654500897, 5654500927, 5654500961, 5654500981, 5654500993,
#          9999999967, 7654487179, 7654489553, 7654495087, 7654486423,
#          7654488209, 8654487029, 8654489771, 8654494517, 8654495341]

PRIMES = [1, 265443567, 805459861]

def hashgrid_naive(coords, resolutions, codebook_bitwidth, lod_idx, codebook):
    """A naive PyTorch implementation of the hashgrid.

    This code exists here mostly as a reference. Do NOT expect a 1-to-1 numerical correspondence
    to the CUDA accelerated version (this might be a future TODO(ttakikawa) but right now there are no 
    strong motivations to ensure 1-to-1 correspondence).

    This code is also very slow. :) 
    
    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, num_samples, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.ModuleList[torch.FloatTensor]): A list of codebooks of shapes [codebook_size, feature_dim].

    Returns:
        (torch.FloatTensor): Features of shape [batch, num_samples, feature_dim]
    """
    _, feature_dim = codebook[0].shape
    batch, num_samples, _ = coords.shape
    codebook_size = 2**codebook_bitwidth
    feats = []
    for i, res in enumerate(resolutions[:lod_idx+1]):
        tf_coords = torch.clip(((coords + 1.0) / 2.0) * res, 0, res-1-1e-5).reshape(-1, 3)
        cc000 = torch.floor(tf_coords).short()
        cc = spc_ops.points_to_corners(cc000).long()

        num_pts = res**3
        if num_pts > codebook_size:
            cidx = ((cc[...,0] * PRIMES[(i*3+0)%len(PRIMES)]) ^ \
                        (cc[...,1] * PRIMES[(i*3+1)%len(PRIMES)]) ^ \
                        (cc[...,2] * PRIMES[(i*3+2)%len(PRIMES)])) % codebook_size
        else:
            cidx = cc[...,0] + cc[...,1] * res + cc[...,2] * res * res
        fs = codebook[i][cidx]
        
        feats.append(cidx[...,0:1])
        feats.append(cidx[...,0:1])
        
        coeffs = _C.ops.spc.coords_to_trilinear_cuda(tf_coords.contiguous(), cc000.contiguous())[...,None]
        feats.append((fs * coeffs).sum(-2))
    # TODO(ttakikawa): This probably does not return according to the num_samples interface
    return torch.cat(feats, -1)

class HashGridInterpolate(torch.autograd.Function):
    # TODO(ttakikawa): This class should also support the 2D case... which also means I have to write another kernel!

    @staticmethod
    def forward(ctx, coords, resolutions, codebook_bitwidth, lod_idx, *codebook):
        # TODO(ttakikawa): Make the kernel use the LOD
        feats_out = wisp_C.ops.hashgrid_interpolate_cuda(coords.contiguous().float(), 
                                                     codebook,
                                                     resolutions,
                                                     codebook_bitwidth).contiguous()
    
        ctx.save_for_backward(coords)
        ctx.resolutions = resolutions
        ctx.num_lods = len(resolutions)
        ctx.codebook_shapes = [_c.shape for _c in codebook]
        ctx.codebook_size = 2**codebook_bitwidth
        ctx.codebook_bitwidth = codebook_bitwidth
        ctx.feature_dim = codebook[0].shape[-1]
        return feats_out
    
    @staticmethod
    def backward(ctx, grad_output):
        coords = ctx.saved_tensors[0]
        resolutions = ctx.resolutions
        codebook_size = ctx.codebook_size
        feature_dim = ctx.feature_dim
        codebook_shapes = ctx.codebook_shapes
        codebook_bitwidth = ctx.codebook_bitwidth
        
        grad_codebook = wisp_C.ops.hashgrid_interpolate_backward_cuda(
                coords.contiguous(), grad_output.contiguous(), 
                resolutions, [c_[0] for c_ in codebook_shapes], 
                codebook_bitwidth, feature_dim)
        return (None, None, None, None, *grad_codebook)
        
def hashgrid(coords, resolutions, codebook_bitwidth, lod_idx, codebook):
    """The hashgrid function accleerated with CUDA.
    
    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, num_samples, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.ModuleList[torch.FloatTensor]): A list of codebooks of shapes [codebook_size, feature_dim].

    Returns:
        (torch.FloatTensor): Features of shape [batch, num_samples, feature_dim]
    """
    batch, num_samples, dim = coords.shape
    feats = HashGridInterpolate.apply(coords.reshape(-1, dim).contiguous(), resolutions, 
            codebook_bitwidth, lod_idx, *[_c for _c in codebook])
    return feats.reshape(batch, num_samples, codebook[0].shape[1] * len(resolutions))
