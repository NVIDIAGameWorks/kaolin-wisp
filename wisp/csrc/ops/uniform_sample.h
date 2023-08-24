/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

std::vector<at::Tensor> uniform_sample_cuda(
    int         scale,
    at::Tensor  ridx,
    at::Tensor  depth,
    at::Tensor  insum);


//???????????????//
// at::Tensor uniform_sample_backward_cuda(
// )

}

