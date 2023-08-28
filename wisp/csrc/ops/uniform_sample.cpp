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
// #include <iostream>

namespace wisp {

#ifdef WITH_CUDA

std::vector<at::Tensor> uniform_sample_cuda_impl(
    float       scale,
    at::Tensor  ridx,
    at::Tensor  depth,
    at::Tensor  insum);

#endif // WITH_CUDA

std::vector<at::Tensor> uniform_sample_cuda(
    int         scale,
    at::Tensor  ridx,
    at::Tensor  depth,
    at::Tensor  insum) {
#ifdef WITH_CUDA


    return uniform_sample_cuda_impl((float)scale, ridx, depth, insum);


#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}


}

