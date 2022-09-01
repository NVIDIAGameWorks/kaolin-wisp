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
typedef unsigned int uint;

__global__ void
find_depth_bound_cuda_kernel(
    int64_t num_packs, 
    int64_t num_nugs, 
    const float* query_depth, 
    const int* curr_idxes_in, 
    int* curr_idxes_out, 
    const float2* depth
){
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs && curr_idxes_in[tidx] > -1) {
    
    uint iidx = curr_idxes_in[tidx]; 
    uint max_iidx = (tidx == num_packs-1) ? num_packs : curr_idxes_in[tidx+1];
    float query = query_depth[tidx];
    
    while (iidx < max_iidx) {
      float entry = depth[iidx].x;
      float exit = depth[iidx].y;
      if (query >= entry && query <= exit) {
        curr_idxes_out[tidx] = iidx;
        return;
      } else if (query < entry) {
        curr_idxes_out[tidx] = iidx;
        return;
      }
      iidx++;
    }
  }
} 

void find_depth_bound_cuda_impl(
    int64_t num_packs, 
    int64_t num_nugs,
    at::Tensor query,
    at::Tensor curr_idxes_in,
    at::Tensor curr_idxes_out,
    at::Tensor depth) {
    find_depth_bound_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
            num_packs, num_nugs, query.data_ptr<float>(),
            curr_idxes_in.data_ptr<int>(), curr_idxes_out.data_ptr<int>(), 
            reinterpret_cast<float2*>(depth.data_ptr<float>()));
}

} // namespace wisp
