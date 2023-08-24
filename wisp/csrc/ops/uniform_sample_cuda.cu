/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace wisp {



__global__ void
uniform_sample_cuda_kernel(
    const int32_t   num_voxels, 
    const float     scale,
    const float     inv_scale,
    const int32_t*  ridx, 
    const float2*   depth, 
    const int32_t*  insum,

    int64_t*        new_ridx,
    float*          depth_samples,
    bool*           boundary) 
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num_voxels) {

    int n = insum[tidx];
    int base = 0;
    if (tidx > 0)
    {
        base = insum[tidx-1];
        n = n - base;
    }

    int32_t ray_idx = ridx[tidx];
    float first_depth = ceil(scale*depth[tidx].x);
    bool bval = tidx == 0 ? true : ridx[tidx] != ridx[tidx-1] ? true : false;

    float f = 0.0f;
    for (int i=0; i<n; i++)
    {
        depth_samples[base+i] = inv_scale*(first_depth+f);
        f += 1.0f;
        new_ridx[base+i] = ray_idx;

        boundary[base+i] = bval;
        bval = false;
    }

  }
}


std::vector<at::Tensor> uniform_sample_cuda_impl(
    float       scale,
    at::Tensor  ridx,
    at::Tensor  depth,
    at::Tensor  insum) 
{
    int32_t num_threads = 512;

    int32_t num_voxels = ridx.size(0);  
    int32_t total_num_samples = 0;

    if (num_voxels > 0)
    {
        int32_t* insum_ptr = insum.data_ptr<int32_t>();
        cudaMemcpy(&total_num_samples, insum_ptr + num_voxels - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    at::Tensor new_ridx = at::zeros({total_num_samples}, ridx.options().dtype(at::kLong));
    at::Tensor depth_samples = at::zeros({total_num_samples, 1}, depth.options());
    at::Tensor boundary = at::zeros({total_num_samples}, ridx.options().dtype(at::kBool));

    uniform_sample_cuda_kernel<<<(num_voxels + num_threads - 1) / num_threads, num_threads>>>(
        num_voxels, 
        scale,
        1.0f/scale,
        ridx.data_ptr<int32_t>(), 
        reinterpret_cast<float2*>(depth.data_ptr<float>()),
        insum.data_ptr<int32_t>(),

        new_ridx.data_ptr<int64_t>(),
        depth_samples.data_ptr<float>(),
        boundary.data_ptr<bool>());

    AT_CUDA_CHECK(cudaGetLastError());

    return {new_ridx, depth_samples, boundary};
}

} // namespace wisp
