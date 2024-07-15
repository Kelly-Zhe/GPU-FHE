#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels.h"

__device__ void vec_mul_scalar_mod(uint64_t* vec, uint64_t scalar, uint64_t mod, int N, uint64_t* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        result[idx] = (vec[idx] * scalar) % mod;
    }
}

__device__ void vec_add_int(uint64_t* vec1, uint64_t* vec2, int N, uint64_t* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        result[idx] = vec1[idx] + vec2[idx];
    }
}

__device__ void vec_mod_int(uint64_t* vec, uint64_t mod, int N, uint64_t* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        result[idx] = vec[idx] % mod;
    }
}

__device__ void vec_sub_mod(uint64_t* vec1, uint64_t* vec2, uint64_t mod, int N, uint64_t* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        result[idx] = (vec1[idx] + mod - vec2[idx]) % mod;
    }
}

__global__ void moddown_core_kernel(
        uint64_t* d_intt_a, uint64_t* d_pHatInvModp, uint64_t* d_pHatModq, uint64_t* d_PInvModq,
        uint64_t* d_moduliQ, uint64_t* d_moduliP,
        uint64_t* d_res, int N, int curr_limbs, int K) {

    extern __shared__ uint64_t shared_memory[];
    uint64_t* tmp3 = shared_memory;
    uint64_t* sum = shared_memory + K * N;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int lane = threadIdx.x;

    if (idx < N * K) {
        int k = idx / N;
        int n = idx % N;
        uint64_t* tmpk = d_intt_a + curr_limbs * N;
        vec_mul_scalar_mod(tmpk + k * N, d_pHatInvModp[k], d_moduliP[k], N, tmp3 + k * N);
    }
    __syncthreads();

    if (idx < N * curr_limbs) {
        int i = idx / N;
        int n = idx % N;

        uint64_t* tmpi = d_intt_a;
        sum[n] = 0;
        for (int k = 0; k < K; k++) {
            uint64_t product;
            vec_mul_scalar_mod(tmp3 + k * N, d_pHatModq[k * curr_limbs + i], d_moduliP[k], N, &product);
            vec_add_int(&sum[n], &product, N, &sum[n]);
        }
        vec_mod_int(&sum[n], d_moduliQ[i], N, &sum[n]);
        vec_sub_mod(tmpi + i * N, &sum[n], d_moduliQ[i], N, d_res + i * N);
        vec_mul_scalar_mod(d_res + i * N, d_PInvModq[i], d_moduliQ[i], N, d_res + i * N);
    }
}

void moddown_core_cuda(
        uint64_t* h_intt_a, uint64_t* h_pHatInvModp, uint64_t* h_pHatModq, uint64_t* h_PInvModq,
        uint64_t* h_moduliQ, uint64_t* h_moduliP,
        uint64_t* h_res, int N, int curr_limbs, int K) {

    uint64_t* d_intt_a;
    uint64_t* d_pHatInvModp;
    uint64_t* d_pHatModq;
    uint64_t* d_PInvModq;
    uint64_t* d_moduliQ;
    uint64_t* d_moduliP;
    uint64_t* d_res;

// 分配设备内存
    cudaMalloc((void**)&d_intt_a, (curr_limbs + K) * N * sizeof(uint64_t));
    cudaMalloc((void**)&d_pHatInvModp, K * sizeof(uint64_t));
    cudaMalloc((void**)&d_pHatModq, K * curr_limbs * sizeof(uint64_t));
    cudaMalloc((void**)&d_PInvModq, curr_limbs * sizeof(uint64_t));
    cudaMalloc((void**)&d_moduliQ, curr_limbs * sizeof(uint64_t));
    cudaMalloc((void**)&d_moduliP, K * sizeof(uint64_t));
    cudaMalloc((void**)&d_res, curr_limbs * N * sizeof(uint64_t));

// 将输入数据从主机传输到设备
    cudaMemcpy(d_intt_a, h_intt_a, (curr_limbs + K) * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pHatInvModp, h_pHatInvModp, K * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pHatModq, h_pHatModq, K * curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PInvModq, h_PInvModq, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliQ, h_moduliQ, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliP, h_moduliP, K * sizeof(uint64_t), cudaMemcpyHostToDevice);

// 设置CUDA内核的执行配置
    int blockSize = 256;
    int numBlocks = (N * K + blockSize - 1) / blockSize;
    size_t shared_memory_size = K * N * sizeof(uint64_t) + N * sizeof(uint64_t);

// 启动CUDA内核
    moddown_core_kernel<<<numBlocks, blockSize, shared_memory_size>>>(
            d_intt_a, d_pHatInvModp, d_pHatModq, d_PInvModq,
            d_moduliQ, d_moduliP, d_res, N, curr_limbs, K);

// 将结果从设备传回主机
    cudaMemcpy(h_res, d_res, curr_limbs * N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

// 释放设备内存
    cudaFree(d_intt_a);
    cudaFree(d_pHatInvModp);
    cudaFree(d_pHatModq);
    cudaFree(d_PInvModq);
    cudaFree(d_moduliQ);
    cudaFree(d_moduliP);
    cudaFree(d_res);
}

int main() {
// 示例输入数据和参数
    const int N = 1024;
    const int curr_limbs = 4;
    const int K = 2;

    uint64_t h_intt_a[(curr_limbs + K) * N];
    uint64_t h_pHatInvModp[K];
    uint64_t h_pHatModq[K * curr_limbs];
    uint64_t h_PInvModq[curr_limbs];
    uint64_t h_moduliQ[curr_limbs];
    uint64_t h_moduliP[K];
    uint64_t h_res[curr_limbs * N];

// 初始化输入数据
// ...

// 调用moddown_core函数
    moddown_core_cuda(h_intt_a, h_pHatInvModp, h_pHatModq, h_PInvModq,
                 h_moduliQ, h_moduliP, h_res, N, curr_limbs, K);

// 打印结果或进一步处理
// ...

    return 0;
}