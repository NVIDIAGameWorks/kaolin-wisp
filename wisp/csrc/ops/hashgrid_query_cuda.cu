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
#include <c10/cuda/CUDAGuard.h>
#include "hash_utils.cuh"

namespace wisp {
typedef unsigned int uint;

template<typename scalar_t>
__global__ void
hashgrid_query_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t probe_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    scalar_t* __restrict__ feats
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x*gridDim.x;
    uint32_t codebook_mod = codebook_size - probe_size;

    for (uint32_t i=tidx; i<num_coords; i+=stride) { 
        
        uint32_t base_idx = i*8*num_lods*probe_size*feature_dim + lod_idx*probe_size*feature_dim;

        float3 x = make_float3(clamp(resolution * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        
        uint32_t corner_idx[8];
#       pragma unroll
        for (uint32_t j=0; j<8; ++j) {
            int3 corner;
            corner.x = pos.x + ((j & 4) >> 2);
            corner.y = pos.y + ((j & 2) >> 1);
            corner.z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index_3d(corner, resolution, codebook_mod);
        }
        
        
        for (uint32_t j=0; j<feature_dim; ++j) {
            for (uint32_t p=0; p<probe_size; ++p) {
#               pragma unroll
                for (uint32_t k=0; k<8; ++k) {
                    feats[base_idx + k*num_lods*probe_size*feature_dim + p*feature_dim + j] = 
                        codebook[corner_idx[k]*feature_dim+j];
                }
            }
        }
    }
} 

void hashgrid_query_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int32_t probe_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor feats){

    int num_threads = 512;
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, feats.type(), "hashgrid_query_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
        auto stream = at::cuda::getCurrentCUDAStream();
        hashgrid_query_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            codebook_size,
            probe_size,
            feature_dim,
            resolution,
            lod_idx,
            num_lods,
            coords.data_ptr<float>(),
            codebook.data_ptr<scalar_t>(),
            feats.data_ptr<scalar_t>()
        );
    }));
}

template<typename scalar_t>
__global__ void
hashgrid_query_backward_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t probe_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
    scalar_t* __restrict__ grad_codebook // codebook_size, feature_dim
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    uint32_t codebook_mod = codebook_size - probe_size;
    
    for (uint32_t i=tidx; i<num_coords; i+=stride) { 
        
        uint32_t base_idx = i*8*num_lods*probe_size*feature_dim + lod_idx*probe_size*feature_dim;
        
        float3 x = make_float3(clamp(resolution * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        uint32_t corner_idx[8];
#       pragma unroll
        for (uint32_t j=0; j<8; ++j) {
            int3 corner;
            corner.x = pos.x + ((j & 4) >> 2);
            corner.y = pos.y + ((j & 2) >> 1);
            corner.z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index_3d(corner, resolution, codebook_mod);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        if (std::is_same<scalar_t, at::Half>::value) {
            for (uint32_t p=0; p<probe_size; ++p) {
                for (uint32_t j=0; j<feature_dim; j += 2) {
#                   pragma unroll
                    for (int k=0; k<8; ++k) {
                        // grad_output shape: [N, 8, num_lods, feature_dim]
                        uint32_t _idx = base_idx + k*num_lods*probe_size*feature_dim + p*feature_dim + j;
                        __half2 grad = reinterpret_cast<const __half2*>(grad_output)[_idx / 2];
                        atomicAdd((__half2*)(grad_codebook + (corner_idx[k]*feature_dim + j)), grad);
                    }
                }
            }
        } else
#endif
        {
            for (uint32_t p=0; p<probe_size; ++p) {
                for (uint32_t j=0; j<feature_dim; ++j) {
#                   pragma unroll
                    for (uint32_t k=0; k<8; ++k) {
                        uint32_t _idx = base_idx + k*num_lods*probe_size*feature_dim + p*feature_dim + j;
                        float grad = grad_output[_idx];
                        atomicAdd((float*)(grad_codebook + (corner_idx[k]*feature_dim + p*feature_dim + j)), grad);
                    }
                }
            }
        }
    }
}

void hashgrid_query_backward_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int32_t probe_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor grad_codebook){

    int num_threads = 512;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "hashgrid_query_backward_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_codebook));
        auto stream = at::cuda::getCurrentCUDAStream();
        hashgrid_query_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            codebook_size,
            probe_size,
            feature_dim,
            resolution,
            lod_idx,
            num_lods,
            coords.data_ptr<float>(),
            grad_output.data_ptr<scalar_t>(),
            grad_codebook.data_ptr<scalar_t>()
        );
    }));
}

} // namespace wisp
