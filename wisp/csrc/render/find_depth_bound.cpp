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

namespace wisp {

void find_depth_bound_cuda_impl(
    int64_t num_packs, 
    int64_t num_nugs,
    at::Tensor query,
    at::Tensor curr_idxes_in,
    at::Tensor curr_idxes_out,
    at::Tensor depth);

at::Tensor find_depth_bound_cuda(
    at::Tensor query,
    at::Tensor curr_idxes_in,
    at::Tensor depth) {
#ifdef WITH_CUDA
  int64_t num_packs = query.size(0);
  int64_t num_nugs = depth.size(0);
  at::Tensor curr_idxes_out = at::zeros({num_packs}, query.options().dtype(at::kInt)) - 1;
  find_depth_bound_cuda_impl(num_packs, num_nugs, query, curr_idxes_in, curr_idxes_out, depth);
  return curr_idxes_out;
#else
  AT_ERROR(__func__);
#endif  // WITH_CUDA
}


}

