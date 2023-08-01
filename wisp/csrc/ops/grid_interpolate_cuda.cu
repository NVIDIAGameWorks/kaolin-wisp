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

namespace wisp {

template<typename scalar_t>
__global__ void
grid_interpolate_cuda_kernel(
    const int64_t num_coords,
    const int64_t feature_dim,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ feats_in,
    scalar_t* __restrict__ feats_out
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float3 x_ = make_float3(coords[i*3+0], coords[i*3+1], coords[i*3+2]);
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float c000 = _x.x * _x.y * _x.z;
        float c001 = _x.x * _x.y * x_.z;
        float c010 = _x.x * x_.y * _x.z;
        float c011 = _x.x * x_.y * x_.z;
        float c100 = x_.x * _x.y * _x.z;
        float c101 = x_.x * _x.y * x_.z;
        float c110 = x_.x * x_.y * _x.z;
        float c111 = x_.x * x_.y * x_.z;
        for (int64_t j=0; j<feature_dim; ++j) {
            float feat =
                static_cast<float>(feats_in[(i*8*feature_dim)+(0*feature_dim)+j]) * c000 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(1*feature_dim)+j]) * c001 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(2*feature_dim)+j]) * c010 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(3*feature_dim)+j]) * c011 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(4*feature_dim)+j]) * c100 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(5*feature_dim)+j]) * c101 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(6*feature_dim)+j]) * c110 +
                static_cast<float>(feats_in[(i*8*feature_dim)+(7*feature_dim)+j]) * c111;

            feats_out[i*feature_dim + j] = static_cast<scalar_t>(feat);
        }
    }
} 

void grid_interpolate_cuda_impl(
    int64_t num_coords, 
    int64_t feature_dim,
    at::Tensor coords,
    at::Tensor feats_in,
    at::Tensor feats_out){

    int num_threads = 512;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "grid_interpolate_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_in));
        auto stream = at::cuda::getCurrentCUDAStream();
        grid_interpolate_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            feature_dim,
            coords.data_ptr<float>(),
            feats_in.data_ptr<scalar_t>(),
            feats_out.data_ptr<scalar_t>()
        );
    }));
}

template<typename scalar_t>
__global__ void
grid_interpolate_backward_cuda_kernel(
    const int64_t num_coords,
    const int64_t feature_dim,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
    scalar_t* __restrict__ grad_feats // codebook_size, feature_dim
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float3 x_ = make_float3(coords[i*3+0], coords[i*3+1], coords[i*3+2]);
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float coeffs[8];
        coeffs[0] = _x.x * _x.y * _x.z;
        coeffs[1] = _x.x * _x.y * x_.z;
        coeffs[2] = _x.x * x_.y * _x.z;
        coeffs[3] = _x.x * x_.y * x_.z;
        coeffs[4] = x_.x * _x.y * _x.z;
        coeffs[5] = x_.x * _x.y * x_.z;
        coeffs[6] = x_.x * x_.y * _x.z;
        coeffs[7] = x_.x * x_.y * x_.z;

        // TODO(ttakikawa): This is wrong...
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        if (std::is_same<scalar_t, at::Half>::value) {
            for (uint64_t j=0; j<feature_dim; j += 2) {
#           pragma unroll
                for (int k=0; k<8; ++k) {
                    __half2 grad = reinterpret_cast<const __half2*>(grad_output)[(i*feature_dim + j) / 2];
                    grad = __floats2half2_rn(__half2float(grad.x) * coeffs[k],
                                             __half2float(grad.y) * coeffs[k]);
                    atomicAdd((__half2*)(grad_feats + (i*8*feature_dim + k*feature_dim + j)), grad);
                }
            }
        } else
#endif
        {
            for (uint64_t j=0; j<feature_dim; ++j) {
#           pragma unroll
                for (int k=0; k<8; ++k) {
                    float grad = grad_output[i*feature_dim + j] * coeffs[k];
                    atomicAdd((float*)(grad_feats + (i*8*feature_dim + k*feature_dim + j)), grad);
                }
            }
        }
    }
}

void grid_interpolate_backward_cuda_impl(
    int64_t num_coords, 
    int64_t feature_dim,
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor grad_feats){

    int num_threads = 512;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "grid_interpolate_backward_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_output));
        auto stream = at::cuda::getCurrentCUDAStream();
        grid_interpolate_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            feature_dim,
            coords.data_ptr<float>(),
            grad_output.data_ptr<scalar_t>(),
            grad_feats.data_ptr<scalar_t>()
        );
    }));
}

} // namespace wisp
