/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace wisp {

at::Tensor grid_interpolate_cuda(
    at::Tensor coords,
    at::Tensor features);

at::Tensor grid_interpolate_backward_cuda(
    at::Tensor coords,
    at::Tensor grad_output, // N, F
    int32_t feature_dim);
}

