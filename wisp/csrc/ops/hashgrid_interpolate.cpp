/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <ATen/ATen.h>
#include <vector>
#include <iostream>

namespace wisp {

void hashgrid_interpolate_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor feats);

void hashgrid_interpolate_backward_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    bool require_grad_coords,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor grad_output,
    at::Tensor grad_codebook,
    at::Tensor grad_coords);

at::Tensor hashgrid_interpolate_cuda(
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor resolution,
    int32_t codebook_bitwidth) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  
    int64_t feature_dim = codebook.size(1);
    int32_t num_lods = resolution.size(0);
    int32_t coord_dim = coords.size(1);
    at::Tensor feats = at::empty({num_coords, codebook.size(1) * int(resolution.size(0))}, codebook.options());

    int32_t codebook_size = pow(2, codebook_bitwidth);

    for (int32_t i=0; i < num_lods; ++i) {
        hashgrid_interpolate_cuda_impl(num_coords, codebook_size, feature_dim, resolution[i], i, num_lods, coord_dim, coords, 
                                       codebook, codebook_first_idx, feats);
    }
    return feats;
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}

std::vector<at::Tensor> hashgrid_interpolate_backward_cuda(
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor resolution,
    int32_t codebook_bitwidth,
    int32_t feature_dim,
    bool require_grad_coords) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  
    int32_t num_lods = resolution.size(0);
    int32_t coord_dim = coords.size(1);

    at::Tensor grad_codebook = at::zeros_like(codebook);
    int32_t codebook_size = pow(2, codebook_bitwidth);

    at::Tensor grad_coords;
    if (require_grad_coords) {
        grad_coords = at::zeros({num_coords, 3}, coords.options());
    } else {
        grad_coords = at::empty({0}, coords.options());
    }

    for (int32_t i=0; i < num_lods; ++i) {
        hashgrid_interpolate_backward_cuda_impl(num_coords, codebook_size, feature_dim, 
                resolution[i], i, num_lods, coord_dim, require_grad_coords,
                coords, codebook, codebook_first_idx, grad_output, grad_codebook, grad_coords);
    }

    return {grad_coords, grad_codebook};
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}


}

