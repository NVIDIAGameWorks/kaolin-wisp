/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <vector>
#include <ATen/ATen.h>

std::vector<at::Tensor> mesh2sdf_gpu_fast_nopre(
    at::Tensor& points,
    at::Tensor& mesh);

namespace wisp {

std::vector<at::Tensor> mesh_to_sdf_cuda(
    at::Tensor points,
    at::Tensor mesh) {
#ifdef WITH_CUDA
  return mesh2sdf_gpu_fast_nopre(points, mesh);
#else
  AT_ERROR(__func__);
#endif  // WITH_CUDA
}

}

