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
    """
    -- This function is deprecated and unused --
    This code exists here mostly as a reference:
    Do NOT expect a 1-to-1 numerical correspondence to the CUDA accelerated version.

    A naive PyTorch implementation of the hashgrid.
    This code is therefore very slow. :)
    
    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.ModuleList[torch.FloatTensor]): A list of codebooks of shapes [codebook_size, feature_dim].

    Returns:
        (torch.FloatTensor): Features of shape [batch, feature_dim]
    """
    _, feature_dim = codebook[0].shape
    batch, _ = coords.shape
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
    return torch.cat(feats, -1)

class HashGridInterpolate(torch.autograd.Function):
    # TODO(ttakikawa): This class should also support the 2D case... which also means I have to write another kernel!

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, coords, resolutions, codebook_bitwidth, lod_idx, codebook, codebook_sizes, codebook_first_idx):
        if codebook[0].shape[-1] % 2 == 1:
            raise Exception("The codebook feature dimension needs to be a multiple of 2.")


        # TODO(ttakikawa): Make the kernel use the LOD
        feats_out = wisp_C.ops.hashgrid_interpolate_cuda(coords.float().contiguous(), 
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
        

        grad_codebook = wisp_C.ops.hashgrid_interpolate_backward_cuda(
                coords.float().contiguous(), grad_output.contiguous(), codebook,
                codebook_first_idx,
                resolutions,  
                codebook_bitwidth, feature_dim, ctx.needs_input_grad[0])
        return (None, None, None, None, grad_codebook, None, None)
        
def hashgrid(coords, resolutions, codebook_bitwidth, lod_idx, codebook, codebook_sizes, codebook_first_idx):
    """A hash-grid query + interpolation function, accelerated with CUDA.
    
    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [batch, 3]
        resolutions (torch.LongTensor): the resolution of the grid per level of shape [num_lods]
        codebook_bitwidth (int): The bitwidth of the codebook. The codebook will have 2^bw entries.
        lod_idx (int): The LOD to aggregate to.
        codebook (torch.ModuleList[torch.FloatTensor]): A list of codebooks of shapes [codebook_size, feature_dim].

    Returns:
        (torch.FloatTensor): Features of shape [batch, feature_dim]
    """
    batch, dim = coords.shape
    feats = HashGridInterpolate.apply(coords.contiguous(), resolutions,
                                      codebook_bitwidth, lod_idx, codebook,
                                      codebook_sizes, codebook_first_idx)
    feature_dim = codebook.shape[1] * len(resolutions)
    return feats.reshape(batch, feature_dim)
