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

void grid_interpolate_cuda_impl(
    int64_t num_coords, 
    int64_t feature_dim,
    at::Tensor coords,
    at::Tensor feats_in,
    at::Tensor feats_out);

void grid_interpolate_backward_cuda_impl(
    int64_t num_coords, 
    int64_t feature_dim,
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor grad_feats);

at::Tensor grid_interpolate_cuda(
    at::Tensor coords,
    at::Tensor features) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  
    int64_t feature_dim = features.size(2);
    at::Tensor features_out = at::empty({num_coords, feature_dim}, features.options());
    grid_interpolate_cuda_impl(num_coords, feature_dim, coords, features, features_out);
    return features_out;
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor grid_interpolate_backward_cuda(
    at::Tensor coords,
    at::Tensor grad_output, // N, F
    int32_t feature_dim) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  

    at::Tensor grad_feats = at::zeros({num_coords, 8, feature_dim}, grad_output.options());

    grid_interpolate_backward_cuda_impl(num_coords, feature_dim, coords, grad_output, grad_feats);
    return grad_feats;
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}


}

