#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
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

// CUDA内核函数，执行向量取模
__device__ void vec_mod_int(uint64_t* vec, uint64_t mod, int N, uint64_t* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        result[idx] = vec[idx] % mod;
    }
}

__global__ void modup_core_kernel(
        uint64_t* d_intt_a, uint64_t* d_d2Tilde,
        uint64_t* d_moduliQ, uint64_t* d_moduliP, uint64_t* d_QHatInvModq, uint64_t* d_QHatModp,
        int curr_limbs, int K, int N) {

    extern __shared__ uint64_t shared_memory[];
    uint64_t* sum = shared_memory;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int j = blockIdx.x;  // 每个block处理一个beta group

    int beta = (int)ceilf((float)curr_limbs / K);
    int in_C_L_index = j * K;
    int in_C_L_len = (j < beta - 1) ? K : (curr_limbs - in_C_L_index);
    int sizeP = curr_limbs - in_C_L_len + K;

    if (j >= beta) return;

    // 初始化sum
    for (int k = 0; k < sizeP * N; k++) {
        sum[k] = 0;
    }
    __syncthreads();

    // 计算tmp和sum
    for (int i = 0; i < in_C_L_len; i++) {
        uint64_t tmp[N];
        vec_mul_scalar_mod(d_intt_a + (in_C_L_index + i) * N, d_QHatInvModq[j * curr_limbs + i], d_moduliQ[in_C_L_index + i], N, tmp);
        for (int k = 0; k < sizeP; k++) {
            uint64_t product[N];
            vec_mul_scalar_mod(tmp, d_QHatModp[curr_limbs * j + i * sizeP + k], d_moduliP[k], N, product);
            vec_add_int(sum + k * N, product, N, sum + k * N);
        }
    }
    __syncthreads();

    // 将sum的结果存储到d_d2Tilde
    int ranges_size = curr_limbs + K - in_C_L_len;
    for (int k = 0; k < ranges_size; k++) {
        int i = (k < in_C_L_index) ? k : (k + in_C_L_len);
        vec_mod_int(sum + k * N, d_moduliP[k], N, d_d2Tilde + j * ranges_size * N + k * N);
    }
}

void modup_core_cuda(
        uint64_t* h_intt_a, uint64_t* h_d2Tilde,
        uint64_t* h_moduliQ, uint64_t* h_moduliP, uint64_t* h_QHatInvModq, uint64_t* h_QHatModp,
        int curr_limbs, int K, int N) {

    uint64_t* d_intt_a;
    uint64_t* d_d2Tilde;
    uint64_t* d_moduliQ;
    uint64_t* d_moduliP;
    uint64_t* d_QHatInvModq;
    uint64_t* d_QHatModp;

    int beta = (int)ceil((float)curr_limbs / K);
    int ranges_size = curr_limbs + K;

    // 分配设备内存
    cudaMalloc((void**)&d_intt_a, curr_limbs * N * sizeof(uint64_t));
    cudaMalloc((void**)&d_d2Tilde, beta * ranges_size * N * sizeof(uint64_t));
    cudaMalloc((void**)&d_moduliQ, curr_limbs * sizeof(uint64_t));
    cudaMalloc((void**)&d_moduliP, K * sizeof(uint64_t));
    cudaMalloc((void**)&d_QHatInvModq, beta * curr_limbs * sizeof(uint64_t));
    cudaMalloc((void**)&d_QHatModp, curr_limbs * beta * ranges_size * sizeof(uint64_t));

    // 将输入数据从主机传输到设备
    cudaMemcpy(d_intt_a, h_intt_a, curr_limbs * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliQ, h_moduliQ, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliP, h_moduliP, K * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QHatInvModq, h_QHatInvModq, beta * curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QHatModp, h_QHatModp, curr_limbs * beta * ranges_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 设置CUDA内核的执行配置
    int blockSize = 256;
    int numBlocks = beta;
    size_t shared_memory_size = ranges_size * N * sizeof(uint64_t);

    // 启动CUDA内核
    modup_core_kernel<<<numBlocks, blockSize, shared_memory_size>>>(
            d_intt_a, d_d2Tilde, d_moduliQ, d_moduliP, d_QHatInvModq, d_QHatModp, curr_limbs, K, N);

    // 将结果从设备传回主机
    cudaMemcpy(h_d2Tilde, d_d2Tilde, beta * ranges_size * N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_intt_a);
    cudaFree(d_d2Tilde);
    cudaFree(d_moduliQ);
    cudaFree(d_moduliP);
    cudaFree(d_QHatInvModq);
    cudaFree(d_QHatModp);
}

int main() {
    // 示例输入数据和参数
    const int N = 1024;
    const int curr_limbs = 4;
    const int K = 2;

    uint64_t h_intt_a[curr_limbs * N];
    uint64_t h_d2Tilde[(int)ceil((float)curr_limbs / K) * (curr_limbs + K) * N];
    uint64_t h_moduliQ[curr_limbs];
    uint64_t h_moduliP[K];
    uint64_t h_QHatInvModq[(int)ceil((float)curr_limbs / K) * curr_limbs];
    uint64_t h_QHatModp[curr_limbs * (int)ceil((float)curr_limbs / K) * (curr_limbs + K)];

    // 初始化输入数据
    // ...

    // 调用modup_core函数
    modup_core_cuda(h_intt_a, h_d2Tilde, h_moduliQ, h_moduliP, h_QHatInvModq, h_QHatModp, curr_limbs, K, N);

    // 打印结果或进一步处理
    // ...

    return 0;
}
