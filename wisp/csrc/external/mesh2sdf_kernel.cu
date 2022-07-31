/***
This third-party library is derived from DualSDF, originally shared under the MIT license.
Original Codebase: https://github.com/zekunhao1995/DualSDF

The MIT License

Copyright (c) https://github.com/zekunhao1995/DualSDF

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
***/

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 128 
#define CUDA_NUM_THREADS_AGGR 128 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

#define SPLIT_FACTOR 64

// TODO: Add ATen-equivalent check

//    THCudaCheck(cudaGetLastError());

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const float* a, float3 b)
{ 
    return a[0] * b.x + a[1] * b.y + a[2] * b.z;
}

inline __host__ __device__ float dot(const float* a, const float* b)
{ 
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __device__ float idot2(const float* a)
{ 
    return __frcp_rn(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline __device__ float3 normalize(float3 a)
{ 
    float norm = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    a.x /= norm;
    a.y /= norm;
    a.z /= norm;
    return a;
}

inline __host__ __device__ float sign(float x)
{ 
    return copysignf(1.0, x);
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ float d2axmb(const float* a, float x, const float* b)
{
    float t0 = a[0] * x - b[0];
    float t1 = a[1] * x - b[1];
    float t2 = a[2] * x - b[2];
    return t0*t0 + t1*t1 + t2*t2;
}

inline __device__ __host__ float d2axmb(const float* a, float x, float3 b)
{
    float t0 = a[0] * x - b.x;
    float t1 = a[1] * x - b.y;
    float t2 = a[2] * x - b.z;
    return t0*t0 + t1*t1 + t2*t2;
}

inline __device__ __host__ float d2axmb(float3 a, float x, float3 b)
{
    float t0 = a.x * x - b.x;
    float t1 = a.y * x - b.y;
    float t2 = a.z * x - b.z;
    return t0*t0 + t1*t1 + t2*t2;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ float3 cross(float3 a, const float* b)
{ 
    return make_float3(a.y*b[2] - a.z*b[1], a.z*b[0] - a.x*b[2], a.x*b[1] - a.y*b[0]); 
}

inline __host__ __device__ float3 cross(const float* a, const float* b)
{ 
    return make_float3(a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]); 
}

inline __host__ __device__ void cross(const float* a, const float* b, float* res)
{
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}

#define SUB(dest,v1,v2) \
          dest.x=v1.x-v2.x; \
          dest.y=v1.y-v2.y; \
          dest.z=v1.z-v2.z;

__global__ void kernel_mesh2sdf(
    const int num_points, 
    const float* __restrict__ points,
    const int num_triangles, 
    const float* __restrict__ mesh,
    const float* __restrict__ v10,
    const float* __restrict__ v21,
    const float* __restrict__ v02,
    const float* __restrict__ nor,
    const float* __restrict__ c_v10_nor,
    const float* __restrict__ c_v21_nor,
    const float* __restrict__ c_v02_nor,
    const float* __restrict__ inv_dot2_v10,
    const float* __restrict__ inv_dot2_v21,
    const float* __restrict__ inv_dot2_v02,
    const float* __restrict__ inv_dot2_nor,
    float* __restrict__ output) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_points) {
        return;
    }
    
    const float* point_ptr = points + index * 3;
    float mindistsq = INFINITY;
    //int num_intersect = 0;
    int pos_intersect[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
    int neg_intersect[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
    float stab_dir_table[13][3] = { {1.0f, 0.0f, 0.0f},
                                    {0.0f, 1.0f, 0.0f},
                                    {0.0f, 0.0f, 1.0f},
                                    {0.0f, 1.0f, 1.0f},
                                    {1.0f, 0.0f, 1.0f},
                                    {1.0f, 1.0f, 0.0f},
                                    {0.0f, 1.0f, -1.0f},
                                    {1.0f, 0.0f, -1.0f},
                                    {1.0f, -1.0f, 0.0f},
                                    {1.0f, 1.0f, 1.0f},
                                    {-1.0f, 1.0f, 1.0f},
                                    {1.0f, -1.0f, 1.0f},
                                    {1.0f, 1.0f, -1.0f}};
    
    // Loop over each triangle
    for (int i=0; i<num_triangles; i++) {
        const float* triangle_ptr = mesh + i * 9;
        const float* v10_ptr = v10 + i * 3;
        const float* v21_ptr = v21 + i * 3;
        const float* v02_ptr = v02 + i * 3;
        const float* nor_ptr = nor + i * 3;
        const float* c_v10_nor_ptr = c_v10_nor + i * 3;
        const float* c_v21_nor_ptr = c_v21_nor + i * 3;
        const float* c_v02_nor_ptr = c_v02_nor + i * 3;
        float inv_dot2_v10_val = inv_dot2_v10[i];
        float inv_dot2_v21_val = inv_dot2_v21[i];
        float inv_dot2_v02_val = inv_dot2_v02[i];
        float inv_dot2_nor_val = inv_dot2_nor[i];

        float3 p0, p1, p2;
        p0.x = point_ptr[0] - triangle_ptr[0];
        p0.y = point_ptr[1] - triangle_ptr[1];
        p0.z = point_ptr[2] - triangle_ptr[2];
        p1.x = point_ptr[0] - triangle_ptr[3];
        p1.y = point_ptr[1] - triangle_ptr[4];
        p1.z = point_ptr[2] - triangle_ptr[5];
        p2.x = point_ptr[0] - triangle_ptr[6];
        p2.y = point_ptr[1] - triangle_ptr[7];
        p2.z = point_ptr[2] - triangle_ptr[8];
        
        // if the normal vector is zero, the triangle is degenerative.
        if (nor_ptr[0] != 0.0f || nor_ptr[1] != 0.0f || nor_ptr[2] != 0.0f) {
                
            float s1 = sign(dot(c_v10_nor_ptr, p0));
            float s2 = sign(dot(c_v21_nor_ptr, p1));
            float s3 = sign(dot(c_v02_nor_ptr, p2));
            
            float distsq;
            if ((s1+s2+s3) < 2.0f) {
            //if (0) {
                // Edge dist
                float ed1 = d2axmb(v10_ptr, clamp(dot(v10_ptr,p0)*inv_dot2_v10_val,0.0f,1.0f), p0);
                float ed2 = d2axmb(v21_ptr, clamp(dot(v21_ptr,p1)*inv_dot2_v21_val,0.0f,1.0f), p1);
                float ed3 = d2axmb(v02_ptr, clamp(dot(v02_ptr,p2)*inv_dot2_v02_val,0.0f,1.0f), p2);
                distsq = fminf(ed1, fminf(ed2, ed3));
            } else {
                // Face dist
                distsq = dot(nor_ptr, p0) * dot(nor_ptr, p0) * inv_dot2_nor_val;
            }
            if (distsq < 0.0f) {
                distsq = 0.0f;
            }
            //distsq = fabs(distsq);
            mindistsq = fminf(mindistsq, distsq);
        }
        
        // Calculate inside/outside
        // With triangle-ray intersection
        // center: point_ptr
        // direction: -point_ptr
        // Triangle: triangle_ptr [x y z x y z x y z]
        //float3 tvec;
        float3 edge1, edge2;
        edge1.x = triangle_ptr[3] - triangle_ptr[0];
        edge1.y = triangle_ptr[4] - triangle_ptr[1];
        edge1.z = triangle_ptr[5] - triangle_ptr[2];
        edge2.x = triangle_ptr[6] - triangle_ptr[0];
        edge2.y = triangle_ptr[7] - triangle_ptr[1];
        edge2.z = triangle_ptr[8] - triangle_ptr[2];
        
        float3 dir;
        
        // test 4 directions
        // more directions:
        // (1 0 0), (0 1 0), (0 0 1)
        // (1 1 0), (1 0 1), (0 1 1)
        // (1 -1 0), (1 0 -1), (0 1 -1)
        // (1 1 1), (1 1 -1), (1 -1 1), (-1 1 1)
        for (int i=0; i<13; i++) {
            dir.x = stab_dir_table[i][0];
            dir.y = stab_dir_table[i][1];
            dir.z = stab_dir_table[i][2];
            

            float3 pvec = cross(dir, edge2);
            float det = dot(edge1, pvec);
            //float3 pvec = cross(point_ptr, v02_ptr); // Both have negative direction, result have correct direction
            //float det = dot(v10_ptr, pvec);
            
            
            if (det > -1e-5 && det < 1e-5) // No intersection at all
                continue;
            float inv_det = 1.0f / det;
            
            float3 tvec;
            tvec.x = point_ptr[0] - triangle_ptr[0];
            tvec.y = point_ptr[1] - triangle_ptr[1];
            tvec.z = point_ptr[2] - triangle_ptr[2];
            
            
            // tvec = orig - vert0
            //tvec = p0;
            // Calculate barycentric uvs
            //float u = dot(p0, pvec) * inv_det;
            float u = dot(tvec, pvec) * inv_det;
            if (u < 0.0f || u > 1.0f) { // u out of bound
                continue;
            }
            //float3 qvec = cross(p0, v10_ptr);
            float3 qvec = cross(tvec, edge1);
            
            //float v = - dot(point_ptr, qvec) * inv_det;
            float v = dot(dir, qvec) * inv_det;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            
            // Ray intersects triangle
            //float t = - dot(v02_ptr, qvec) * inv_det;
            float t = dot(edge2, qvec) * inv_det;

            if (t >= 0.0f)
                pos_intersect[i] = 1;
            else
                neg_intersect[i] = 1;
            //if (t >= 0.0f) {
            //    num_intersect++;
            //}
        }
    }
    
    float mindist = sqrtf(mindistsq);
    
    int outside = 0;
    for (int i=0; i<13; i++) {
        if (pos_intersect[i] == 0 || neg_intersect[i] == 0) { // if outside
            outside = 1;
            break;
        }
    }
    if (!outside) {
        mindist = -1e-6f;
    }
    //if (pos_intersect && neg_intersect) {
    //    mindist = -1e-6f;
    //}
    //if (num_intersect % 2 == 1) {
    //    mindist *= -1.0f;
    //    printf("%d\n", num_intersect);
    //}
    //output[index] = sqrtf(mindistsq) * mindistsign;
    output[index] = mindist;
}

__global__ void kernel_mesh2sdf_quad(
    const int num_points, 
    const float* __restrict__ points,
    const int num_triangles, 
    const float* __restrict__ mesh,
    float* __restrict__ output,
    uint8_t* __restrict__ raystab_results) {
    const int split_factor = SPLIT_FACTOR;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int pindex = index % num_points;
    int rindex = index / num_points;
    
    if (rindex >= split_factor) {
        return;
    }

    int tranksize = (num_triangles + split_factor - 1) / split_factor;
    int tstart = tranksize * rindex;
    int tend = min(tstart + tranksize, num_triangles);
    
    // SPLIT_FACTOR, num_points, 13, 2
    uint8_t* raystab_cache_base = raystab_results + rindex * num_points * 13 * 2 + pindex * 13 * 2;
    // SPLIT_FACTOR, num_points
    float* output_base = output + rindex * num_points;
    
    const float* point_ptr = points + pindex * 3;
    float mindistsq = INFINITY;
    //int num_intersect = 0;
    //int pos_intersect[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
    //int neg_intersect[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t* pos_intersect = raystab_cache_base;
    uint8_t* neg_intersect = raystab_cache_base + 13;
    
    float stab_dir_table[13][3] = { {1.0f, 0.0f, 0.0f},
                                    {0.0f, 1.0f, 0.0f},
                                    {0.0f, 0.0f, 1.0f},
                                    {0.0f, 0.707106781f, 0.707106781f},
                                    {0.707106781f, 0.0f, 0.707106781f},
                                    {0.707106781f, 0.707106781f, 0.0f},
                                    {0.0f, 0.707106781f, -0.707106781f},
                                    {0.707106781f, 0.0f, -0.707106781f},
                                    {0.707106781f, -0.707106781f, 0.0f},
                                    {0.577350269f, 0.577350269f, 0.577350269f},
                                    {-0.577350269f, 0.577350269f, 0.577350269f},
                                    {0.577350269f, -0.577350269f, 0.577350269f},
                                    {0.577350269f, 0.577350269f, -0.577350269f}};
    
    // Loop over each triangle
    //for (int i=0; i<num_triangles; i++) {
    for (int i=tstart; i<tend; i++) {
    
        const float* triangle_ptr = mesh + i * 9;
        float v10_ptr[3];
        float v21_ptr[3];
        float v02_ptr[3];
        v10_ptr[0] = triangle_ptr[3] - triangle_ptr[0];
        v10_ptr[1] = triangle_ptr[4] - triangle_ptr[1];
        v10_ptr[2] = triangle_ptr[5] - triangle_ptr[2];
        v21_ptr[0] = triangle_ptr[6] - triangle_ptr[3];
        v21_ptr[1] = triangle_ptr[7] - triangle_ptr[4];
        v21_ptr[2] = triangle_ptr[8] - triangle_ptr[5];
        v02_ptr[0] = triangle_ptr[0] - triangle_ptr[6];
        v02_ptr[1] = triangle_ptr[1] - triangle_ptr[7];
        v02_ptr[2] = triangle_ptr[2] - triangle_ptr[8];
        
        float nor_ptr[3];
        cross(v10_ptr, v02_ptr, nor_ptr);
        
        float c_v10_nor_ptr[3];
        float c_v21_nor_ptr[3];
        float c_v02_nor_ptr[3];
        cross(v10_ptr, nor_ptr, c_v10_nor_ptr);
        cross(v21_ptr, nor_ptr, c_v21_nor_ptr);
        cross(v02_ptr, nor_ptr, c_v02_nor_ptr);
        
        float inv_dot2_v10_val = idot2(v10_ptr);
        float inv_dot2_v21_val = idot2(v21_ptr);
        float inv_dot2_v02_val = idot2(v02_ptr);
        float inv_dot2_nor_val = idot2(nor_ptr);
        
        //const float* v10_ptr = v10 + i * 3;
        //const float* v21_ptr = v21 + i * 3;
        //const float* v02_ptr = v02 + i * 3;
        //const float* nor_ptr = nor + i * 3;
        //const float* c_v10_nor_ptr = c_v10_nor + i * 3;
        //const float* c_v21_nor_ptr = c_v21_nor + i * 3;
        //const float* c_v02_nor_ptr = c_v02_nor + i * 3;
        //float inv_dot2_v10_val = inv_dot2_v10[i];
        //float inv_dot2_v21_val = inv_dot2_v21[i];
        //float inv_dot2_v02_val = inv_dot2_v02[i];
        //float inv_dot2_nor_val = inv_dot2_nor[i];

        float p0[3], p1[3], p2[3];
        p0[0] = point_ptr[0] - triangle_ptr[0];
        p0[1] = point_ptr[1] - triangle_ptr[1];
        p0[2] = point_ptr[2] - triangle_ptr[2];
        p1[0] = point_ptr[0] - triangle_ptr[3];
        p1[1] = point_ptr[1] - triangle_ptr[4];
        p1[2] = point_ptr[2] - triangle_ptr[5];
        p2[0] = point_ptr[0] - triangle_ptr[6];
        p2[1] = point_ptr[1] - triangle_ptr[7];
        p2[2] = point_ptr[2] - triangle_ptr[8];
        /*
        float3 p0, p1, p2;
        p0.x = point_ptr[0] - triangle_ptr[0];
        p0.y = point_ptr[1] - triangle_ptr[1];
        p0.z = point_ptr[2] - triangle_ptr[2];
        p1.x = point_ptr[0] - triangle_ptr[3];
        p1.y = point_ptr[1] - triangle_ptr[4];
        p1.z = point_ptr[2] - triangle_ptr[5];
        p2.x = point_ptr[0] - triangle_ptr[6];
        p2.y = point_ptr[1] - triangle_ptr[7];
        p2.z = point_ptr[2] - triangle_ptr[8];
        */
        // if the normal vector is zero, the triangle is degenerative.
        if (nor_ptr[0] != 0.0f || nor_ptr[1] != 0.0f || nor_ptr[2] != 0.0f) {
                
            float s1 = sign(dot(c_v10_nor_ptr, p0));
            float s2 = sign(dot(c_v21_nor_ptr, p1));
            float s3 = sign(dot(c_v02_nor_ptr, p2));
            
            float distsq;
            if ((s1+s2+s3) < 2.0f) {
                // Edge dist
                float ed1 = d2axmb(v10_ptr, clamp(dot(v10_ptr,p0)*inv_dot2_v10_val,0.0f,1.0f), p0);
                float ed2 = d2axmb(v21_ptr, clamp(dot(v21_ptr,p1)*inv_dot2_v21_val,0.0f,1.0f), p1);
                float ed3 = d2axmb(v02_ptr, clamp(dot(v02_ptr,p2)*inv_dot2_v02_val,0.0f,1.0f), p2);
                distsq = fminf(ed1, fminf(ed2, ed3));
            } else {
                // Face dist
                distsq = dot(nor_ptr, p0) * dot(nor_ptr, p0) * inv_dot2_nor_val;
            }
            if (distsq < 0.0f) {
                distsq = 0.0f;
            }
            //distsq = fabs(distsq);
            mindistsq = fminf(mindistsq, distsq);
        }
        
        // Calculate inside/outside
        // With triangle-ray intersection
        // center: point_ptr
        // direction: -point_ptr
        // Triangle: triangle_ptr [x y z x y z x y z]
        //float3 tvec;
        //float3 edge1, edge2;
        float3 edge2;
        float* edge1 = v10_ptr;
        /*
        edge1.x = triangle_ptr[3] - triangle_ptr[0];
        edge1.y = triangle_ptr[4] - triangle_ptr[1];
        edge1.z = triangle_ptr[5] - triangle_ptr[2];
        edge2.x = triangle_ptr[6] - triangle_ptr[0];
        edge2.y = triangle_ptr[7] - triangle_ptr[1];
        edge2.z = triangle_ptr[8] - triangle_ptr[2];
        */
        //edge1.x = v10_ptr[0];
        //edge1.y = v10_ptr[1];
        //edge1.z = v10_ptr[2];
        edge2.x = -v02_ptr[0];
        edge2.y = -v02_ptr[1];
        edge2.z = -v02_ptr[2];
        
        float3 dir;
        
        // test 4 directions
        // more directions:
        // (1 0 0), (0 1 0), (0 0 1)
        // (1 1 0), (1 0 1), (0 1 1)
        // (1 -1 0), (1 0 -1), (0 1 -1)
        // (1 1 1), (1 1 -1), (1 -1 1), (-1 1 1)
        for (int i=0; i<13; i++) {
            dir.x = stab_dir_table[i][0];
            dir.y = stab_dir_table[i][1];
            dir.z = stab_dir_table[i][2];

            float3 pvec = cross(dir, edge2);
            //pvec = normalize(pvec);
            float det = dot(edge1, pvec);
            //float3 pvec = cross(point_ptr, v02_ptr); // Both have negative direction, result have correct direction
            //float det = dot(v10_ptr, pvec);
            
            
            if (det > -1e-8 && det < 1e-8) // No intersection at all
                continue;
            float inv_det = 1.0f / det;
            
            float3 tvec;
            tvec.x = point_ptr[0] - triangle_ptr[0];
            tvec.y = point_ptr[1] - triangle_ptr[1];
            tvec.z = point_ptr[2] - triangle_ptr[2];
            
            
            // tvec = orig - vert0
            //tvec = p0;
            // Calculate barycentric uvs
            //float u = dot(p0, pvec) * inv_det;
            float u = dot(tvec, pvec) * inv_det;
            if (u < 0.0f || u > 1.0f) { // u out of bound
                continue;
            }
            //float3 qvec = cross(p0, v10_ptr);
            float3 qvec = cross(tvec, edge1);
            
            //float v = - dot(point_ptr, qvec) * inv_det;
            float v = dot(dir, qvec) * inv_det;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            
            // Ray intersects triangle
            //float t = - dot(v02_ptr, qvec) * inv_det;
            float t = dot(edge2, qvec) * inv_det;

            if (t >= 0.0f)
                pos_intersect[i] = 1;
            else
                neg_intersect[i] = 1;
            //if (t >= 0.0f) {
            //    num_intersect++;
            //}
        }
    }
    
    //float mindist = sqrtf(mindistsq);
    //output_base[pindex] = mindist;
    output_base[pindex] = mindistsq;
    
    /*
    int outside = 0;
    for (int i=0; i<13; i++) {
        if (pos_intersect[i] == 0 || neg_intersect[i] == 0) { // if outside
            outside = 1;
            break;
        }
    }
    if (!outside) {
        mindist = -1e-6f;
    }*/
    //if (pos_intersect && neg_intersect) {
    //    mindist = -1e-6f;
    //}
    //if (num_intersect % 2 == 1) {
    //    mindist *= -1.0f;
    //    printf("%d\n", num_intersect);
    //}
    //output[index] = sqrtf(mindistsq) * mindistsign;
    
}

__global__ void kernel_quad_aggr(
    const int num_points, 
    const int num_triangles, 
    const float* __restrict__ output,
    const uint8_t* __restrict__ raystab_results,
    float* __restrict__ final_output) {
    
    const int split_factor = SPLIT_FACTOR;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_points) {
        return;
    }
    
    // for each point in 3D space
    int outside = 0;
    for (int i=0; i<13; i++) {
        uint8_t pos_aggr = 0;
        uint8_t neg_aggr = 0;
        for (int s=0; s<SPLIT_FACTOR; s++) {
            const uint8_t* raystab_cache_base = raystab_results + s * num_points * 13 * 2 + index * 13 * 2;
            const uint8_t* pos_intersect = raystab_cache_base;
            const uint8_t* neg_intersect = raystab_cache_base + 13;
            pos_aggr = pos_aggr | pos_intersect[i];
            neg_aggr = neg_aggr | neg_intersect[i];
        }
        if (pos_aggr == 0 || neg_aggr == 0) { // if outside
            outside = 1;
            break;
        }
    }
    
    float mindist;
    float mindistsq = INFINITY;
    for (int s=0; s<SPLIT_FACTOR; s++) {
        const float* output_base = output + s * num_points;
        mindistsq = fminf(mindistsq, output_base[index]);
    }
    if (mindistsq < 0.0f) {
        mindistsq = 0.0f;
    }
    mindist = sqrtf(mindistsq);
    if (!outside) {
        mindist = -mindist;
    }
    
    /*
    if (outside) {
        float mindistsq = INFINITY;
        for (int s=0; s<SPLIT_FACTOR; s++) {
            const float* output_base = output + s * num_points;
            mindistsq = fminf(mindistsq, output_base[index]);
        }
        mindist = sqrtf(mindistsq);
    } else {
        mindist = -1e-6f;
    }*/
    final_output[index] = mindist;
}

__global__ void kernel_triangle2sdf_forward(
    const int num_points, 
    const float* __restrict__ points,
    const int num_triangles, 
    const float* __restrict__ vertices,
    const float* __restrict__ v10,
    const float* __restrict__ v21,
    const float* __restrict__ v02,
    const float* __restrict__ nor,
    const float* __restrict__ c_v10_nor,
    const float* __restrict__ c_v21_nor,
    const float* __restrict__ c_v02_nor,
    const float* __restrict__ inv_dot2_v10,
    const float* __restrict__ inv_dot2_v21,
    const float* __restrict__ inv_dot2_v02,
    const float* __restrict__ inv_dot2_nor,
    bool* __restrict__ out_isfacedist,
    int* __restrict__ out_nearesttriangle,
    float* __restrict__ output) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_points) {
        return;
    }
    
    const float* point_ptr = points + index * 3;
    float mindistsq = INFINITY;
    int nearest_triangle_id = -1;
    bool is_face_dist = false;
    // Loop over each triangle
    for (int i=0; i<num_triangles; i++) {
        const float* triangle_ptr = vertices + i * 9;
        const float* v10_ptr = v10 + i * 3;
        const float* v21_ptr = v21 + i * 3;
        const float* v02_ptr = v02 + i * 3;
        const float* nor_ptr = nor + i * 3;
        const float* c_v10_nor_ptr = c_v10_nor + i * 3;
        const float* c_v21_nor_ptr = c_v21_nor + i * 3;
        const float* c_v02_nor_ptr = c_v02_nor + i * 3;
        float inv_dot2_v10_val = inv_dot2_v10[i];
        float inv_dot2_v21_val = inv_dot2_v21[i];
        float inv_dot2_v02_val = inv_dot2_v02[i];
        float inv_dot2_nor_val = inv_dot2_nor[i];

        float3 p0, p1, p2;
        p0.x = point_ptr[0] - triangle_ptr[0];
        p0.y = point_ptr[1] - triangle_ptr[1];
        p0.z = point_ptr[2] - triangle_ptr[2];
        p1.x = point_ptr[0] - triangle_ptr[3];
        p1.y = point_ptr[1] - triangle_ptr[4];
        p1.z = point_ptr[2] - triangle_ptr[5];
        p2.x = point_ptr[0] - triangle_ptr[6];
        p2.y = point_ptr[1] - triangle_ptr[7];
        p2.z = point_ptr[2] - triangle_ptr[8];
        
        // if the normal vector is zero, the triangle is degenerative.
        if (nor_ptr[0] != 0.0f || nor_ptr[1] != 0.0f || nor_ptr[2] != 0.0f) {
            
            float s1 = sign(dot(c_v10_nor_ptr, p0));
            float s2 = sign(dot(c_v21_nor_ptr, p1));
            float s3 = sign(dot(c_v02_nor_ptr, p2));
            
            float distsq;
            bool face_dist;
            if ((s1+s2+s3) < 2.0f) {
            //if (0) {
                // Edge dist
                face_dist = false;
                float ed1 = d2axmb(v10_ptr, clamp(dot(v10_ptr,p0)*inv_dot2_v10_val,0.0f,1.0f), p0);
                float ed2 = d2axmb(v21_ptr, clamp(dot(v21_ptr,p1)*inv_dot2_v21_val,0.0f,1.0f), p1);
                float ed3 = d2axmb(v02_ptr, clamp(dot(v02_ptr,p2)*inv_dot2_v02_val,0.0f,1.0f), p2);
                distsq = fminf(ed1, fminf(ed2, ed3));
            } else {
                // Face dist
                face_dist = true;
                distsq = dot(nor_ptr, p0) * dot(nor_ptr, p0) * inv_dot2_nor_val;
            }
            if (distsq < 0.0f) {
                distsq = 0.0f;
            }
            if (distsq < mindistsq) {
                mindistsq = distsq;
                nearest_triangle_id = i;
                is_face_dist = face_dist;
            }
            //distsq = fabs(distsq);
            //mindistsq = fminf(mindistsq, distsq);
        }
    }
    
    float mindist = sqrtf(mindistsq);
    output[index] = mindist;
    out_nearesttriangle[index] = nearest_triangle_id;
    out_isfacedist[index] = is_face_dist;
}


// CUDA kernel for trimming triangles that are inside the meshes.
// Test if one of the vertex of a triangle is outside the shape.
// Failure case: When all of the three vertices are inside the shape but a portion of the face is outside.
__global__ void kernel_trimmesh(
    const int num_triangles, 
    const float* __restrict__ mesh,
    bool* __restrict__ output) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_triangles) {
        return;
    }
    
    // (1 0 0), (0 1 0), (0 0 1)
    // (1 1 0), (1 0 1), (0 1 1)
    // (1 -1 0), (1 0 -1), (0 1 -1)
    // (1 1 1), (1 1 -1), (1 -1 1), (-1 1 1)
    float raystubdirs[13][3] = {{1.0f, 0.0f, 0.0f},
                                {0.0f, 1.0f, 0.0f},
                                {0.0f, 0.0f, 1.0f},
                                {0.0f, 0.707106781f, 0.707106781f},
                                {0.707106781f, 0.0f, 0.707106781f},
                                {0.707106781f, 0.707106781f, 0.0f},
                                {0.0f, 0.707106781f, -0.707106781f},
                                {0.707106781f, 0.0f, -0.707106781f},
                                {0.707106781f, -0.707106781f, 0.0f},
                                {0.577350269f, 0.577350269f, 0.577350269f},
                                {-0.577350269f, 0.577350269f, 0.577350269f},
                                {0.577350269f, -0.577350269f, 0.577350269f},
                                {0.577350269f, 0.577350269f, -0.577350269f}};
    
    const float* vtx_ptr = mesh + index * 9; // x y z
    // for each vertex
    for (int p=6; p>=0; p--) {
        //const float* vtx_ptr = mesh + index * 9 + p * 3; // x y z
        float3 vtx;
        if (p == 6) {           // Center
            vtx.x = (vtx_ptr[0] + vtx_ptr[3] + vtx_ptr[6]) / 3.0f;
            vtx.y = (vtx_ptr[1] + vtx_ptr[4] + vtx_ptr[7]) / 3.0f;
            vtx.z = (vtx_ptr[2] + vtx_ptr[5] + vtx_ptr[8]) / 3.0f;
        } else if (p >= 3) {    // Edges
            int p1 = p % 3;
            int p2 = (p + 1) % 3;
            vtx.x = (vtx_ptr[p1*3] + vtx_ptr[p2*3]) / 2.0f;
            vtx.y = (vtx_ptr[p1*3+1] + vtx_ptr[p2*3+1]) / 2.0f;
            vtx.z = (vtx_ptr[p1*3+2] + vtx_ptr[p2*3+2]) / 2.0f;
        } else {                // Vertices
            int p1 = p % 3;
            vtx.x = vtx_ptr[p1*3];
            vtx.y = vtx_ptr[p1*3+1];
            vtx.z = vtx_ptr[p1*3+2];
        }
        
        int pos_intersect[13] = {};
        int neg_intersect[13] = {};
        // for each triangle
        for (int t=0; t<num_triangles; t++) {
            //if (t == index) {
            //    continue;
            //}
            const float* triangle_ptr = mesh + t * 9;
            
            float3 edge1, edge2;
            edge1.x = triangle_ptr[3] - triangle_ptr[0];
            edge1.y = triangle_ptr[4] - triangle_ptr[1];
            edge1.z = triangle_ptr[5] - triangle_ptr[2];
            edge2.x = triangle_ptr[6] - triangle_ptr[0];
            edge2.y = triangle_ptr[7] - triangle_ptr[1];
            edge2.z = triangle_ptr[8] - triangle_ptr[2];
            
            // test directions:
            float3 dir;
            for (int i=0; i<13; i++) {
                dir.x = raystubdirs[i][0];
                dir.y = raystubdirs[i][1];
                dir.z = raystubdirs[i][2];
                
                float3 pvec = cross(dir, edge2);
                float det = dot(edge1, pvec);
                
                if (det > -1e-8 && det < 1e-8) // No intersection at all
                    continue;
                float inv_det = 1.0f / det;
                
                float3 tvec;
                //tvec.x = vtx_ptr[0] - triangle_ptr[0];
                //tvec.y = vtx_ptr[1] - triangle_ptr[1];
                //tvec.z = vtx_ptr[2] - triangle_ptr[2];
                tvec.x = vtx.x - triangle_ptr[0];
                tvec.y = vtx.y - triangle_ptr[1];
                tvec.z = vtx.z - triangle_ptr[2];
                
                // Calculate barycentric uvs
                float u = dot(tvec, pvec) * inv_det;
                if (u <= 0.0f || u >= 1.0f) { // u out of bound
                    continue;
                }
                float3 qvec = cross(tvec, edge1);
                
                float v = dot(dir, qvec) * inv_det;
                if (v <= 0.0f || u + v >= 1.0f) {
                    continue;
                }
                
                // Ray intersects triangle
                float t = dot(edge2, qvec) * inv_det;

                if (t > 1e-4f)
                    pos_intersect[i] = 1;
                else if (t < -1e-4f)
                    neg_intersect[i] = 1;
            }
            // If there exist one ray that has no or only single sided intersection
            // The point is outside.
        }
        
        // As long as one vertex is outside, the whole triangle is outside.
        for (int i=0; i<13; i++) {
            if (pos_intersect[i] == 0 || neg_intersect[i] == 0) { // if outside
                output[index] = true;
                return;
            }
        }
    }
    
    // 1: fine. 0: trim it!
    output[index] = false;
}


std::vector<torch::Tensor> mesh2sdf_gpu(
    torch::Tensor& points,
    torch::Tensor& mesh,
    torch::Tensor& v10,
    torch::Tensor& v21,
    torch::Tensor& v02,
    torch::Tensor& nor,
    torch::Tensor& c_v10_nor,
    torch::Tensor& c_v21_nor,
    torch::Tensor& c_v02_nor,
    torch::Tensor& inv_dot2_v10,
    torch::Tensor& inv_dot2_v21,
    torch::Tensor& inv_dot2_v02,
    torch::Tensor& inv_dot2_nor ) {
    
    // Step 1: create an empty tensor to store the output.
    int num_points = points.size(0);
    torch::Tensor out_dist = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int num_triangles = mesh.size(0);
    
    //printf("Hello, world!\nEvaluating %d points...\n", num_points);
    
    // Step 2: for each point, find its distance to mesh
    // Assume all the arrays are contiguous.
    // <<<Dg, Db, Ns, S>>>
    //kernel_mesh2sdf<<< (num_points + 256 - 1)/256, 256, 0, at::cuda::getCurrentCUDAStream() >>>(
    kernel_mesh2sdf<<< (num_points + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
          num_points,
          points.data<float>(), 
          num_triangles,
          mesh.data<float>(), 
          v10.data<float>(), 
          v21.data<float>(), 
          v02.data<float>(), 
          nor.data<float>(), 
          c_v10_nor.data<float>(), 
          c_v21_nor.data<float>(), 
          c_v02_nor.data<float>(), 
          inv_dot2_v10.data<float>(), 
          inv_dot2_v21.data<float>(), 
          inv_dot2_v02.data<float>(), 
          inv_dot2_nor.data<float>(),
          out_dist.data<float>());
    
    return {out_dist};
}

std::vector<torch::Tensor> mesh2sdf_gpu_fast_nopre(
    torch::Tensor& points,
    torch::Tensor& mesh ) {
    
    // Step 1: create an empty tensor to store the output.
    int num_points = points.size(0);
    torch::Tensor out_dist = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor out_dist_temp = torch::empty({SPLIT_FACTOR, num_points}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int num_triangles = mesh.size(0);
    
    torch::Tensor raystab_results = torch::zeros({SPLIT_FACTOR, num_points, 13, 2}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    
    // Step 2: for each point, find its distance to mesh
    // Assume all the arrays are contiguous.
    // <<<Dg, Db, Ns, S>>>
    kernel_mesh2sdf_quad<<< (num_points * SPLIT_FACTOR + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
        num_points,
        points.data<float>(), 
        num_triangles,
        mesh.data<float>(), 
        out_dist_temp.data<float>(),
        raystab_results.data<uint8_t>());
        
    kernel_quad_aggr<<<(num_points + CUDA_NUM_THREADS_AGGR - 1)/CUDA_NUM_THREADS_AGGR, CUDA_NUM_THREADS_AGGR, 0, at::cuda::getCurrentCUDAStream() >>>(
        num_points,
        num_triangles,
        out_dist_temp.data<float>(),
        raystab_results.data<uint8_t>(),
        out_dist.data<float>());
    
    return {out_dist};
}


std::vector<torch::Tensor> triangle2sdf_gpu_forward(
    torch::Tensor& points,
    torch::Tensor& triangles,
    torch::Tensor& v10,
    torch::Tensor& v21,
    torch::Tensor& v02,
    torch::Tensor& nor,
    torch::Tensor& c_v10_nor,
    torch::Tensor& c_v21_nor,
    torch::Tensor& c_v02_nor,
    torch::Tensor& inv_dot2_v10,
    torch::Tensor& inv_dot2_v21,
    torch::Tensor& inv_dot2_v02,
    torch::Tensor& inv_dot2_nor ) {
    
    // Step 1: create an empty tensor to store the output.
    int num_points = points.size(0);
    torch::Tensor out_dist = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    torch::Tensor out_isfacedist = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    
    torch::Tensor out_nearesttriangle = torch::empty({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    
    int num_triangles = triangles.size(0);
    
    //printf("Hello, world!\nEvaluating %d points...\n", num_points);
    
    // Step 2: for each point, find its distance to mesh
    // Assume all the arrays are contiguous.
    // <<<Dg, Db, Ns, S>>>
    kernel_triangle2sdf_forward<<< (num_points + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
          num_points,
          points.data<float>(), 
          num_triangles,
          triangles.data<float>(), 
          v10.data<float>(), 
          v21.data<float>(), 
          v02.data<float>(), 
          nor.data<float>(), 
          c_v10_nor.data<float>(), 
          c_v21_nor.data<float>(), 
          c_v02_nor.data<float>(), 
          inv_dot2_v10.data<float>(), 
          inv_dot2_v21.data<float>(), 
          inv_dot2_v02.data<float>(), 
          inv_dot2_nor.data<float>(),
          out_isfacedist.data<bool>(),
          out_nearesttriangle.data<int>(),
          out_dist.data<float>());
          
    return {out_dist, out_isfacedist, out_nearesttriangle};
}


/*
 * Input: mesh: [N 3 3]
 * Output: boolean: [N]
 * For each triangle, for each vertex, do ray-stabbing
 * if all of the vertices of a triangle are inside, indicate the triangle as inside
 */
torch::Tensor trimmesh_gpu( torch::Tensor& mesh ) {
    // Step 1: create an empty tensor to store the output.
    int num_triangles = mesh.size(0);
    //torch::Tensor out_isinside = torch::empty({num_triangles}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    torch::Tensor out_isinside = torch::empty({num_triangles}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    
    // Step 2: for each point, find its distance to mesh
    // Assume all the arrays are contiguous.
    // <<<Dg, Db, Ns, S>>>
    kernel_trimmesh<<< (num_triangles + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
          num_triangles,
          mesh.data<float>(), 
          out_isinside.data<bool>());
          
    return out_isinside;
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mesh2sdf_gpu_pre", &mesh2sdf_gpu, "Mesh2SDF GPU");
    m.def("mesh2sdf_gpu", &mesh2sdf_gpu_fast_nopre, "Mesh2SDF GPU Fast Version No Precomputation");
    m.def("trimmesh_gpu", &trimmesh_gpu, "TrimMesh GPU");
    m.def("p2f_gpu", &triangle2sdf_gpu_forward, "Point to face distance");
}*/
