#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using namespace std;


//__global__ void modup_core_kernel(
//        uint64_t *d_intt_a, uint64_t *d_d2Tilde,
//        uint64_t *d_moduliQ, uint64_t *d_moduliP, uint64_t *d_QHatInvModq, uint64_t *d_QHatModp,
//        int curr_limbs, int K, int N) {
//
//    extern __shared__ uint64_t
//    shared_memory[];
//    uint64_t * sum = shared_memory;
//
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    int j = blockIdx.x;  // 每个block处理一个beta group
//
//    int beta = (int) ceilf((float) curr_limbs / K);
//    int in_C_L_index = j * K;
//    int in_C_L_len = (j < beta - 1) ? K : (curr_limbs - in_C_L_index);
//    int sizeP = curr_limbs - in_C_L_len + K;
//
//    if (j >= beta) return;
//
//    // 初始化sum
//    for (int k = 0; k < sizeP * N; k++) {
//        sum[k] = 0;
//    }
//    __syncthreads();
//
//    // 计算tmp和sum
//    for (int i = 0; i < in_C_L_len; i++) {
//        uint64_t tmp[N];
//        vec_mul_scalar_mod(d_intt_a + (in_C_L_index + i) * N, d_QHatInvModq[j * curr_limbs + i],
//                           d_moduliQ[in_C_L_index + i], N, tmp);
//        for (int k = 0; k < sizeP; k++) {
//            uint64_t product[N];
//            vec_mul_scalar_mod(tmp, d_QHatModp[curr_limbs * j + i * sizeP + k], d_moduliP[k], N, product);
//            vec_add_int(sum + k * N, product, N, sum + k * N);
//        }
//    }
//    __syncthreads();
//
//    // 将sum的结果存储到d_d2Tilde
//    int ranges_size = curr_limbs + K - in_C_L_len;
//    for (int k = 0; k < ranges_size; k++) {
//        int i = (k < in_C_L_index) ? k : (k + in_C_L_len);
//        vec_mod_int(sum + k * N, d_moduliP[k], N, d_d2Tilde + j * ranges_size * N + k * N);
//    }
//}

// 计算tmp和sum
//block size = 256, girdsize = in_C_L_len*N/256
__global__ void
modup_kernel(uint64_t *d_sum, uint64_t *d_intt_a, uint64_t in_C_L_index, int curr_limbs, int N, int sizeP,
             uint64_t *d_QHatInvModq, uint64_t *d_moduliQ, uint64_t *d_QHatModp, int beta,
             int in_C_L_len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / N;
    int j = idx % N;
    uint128_t tmp[256] = {0};
    tmp[idx] = mul_mod(d_intt_a + (in_C_L_index + i) * N, d_QHatInvModq[beta * curr_limbs + i],
                       d_moduliQ[in_C_L_index + i]);
    for (int k = 0; k < sizeP; ++k) {
        uint128_t product = mult_64_64_128(tmp[idx], d_QHatModp[(curr_limbs - 1) * beta * in_C_L_len + i * sizeP + k]);
        inplace_add_128_128(product, d_sum[idx]);
    }
}

//blocksize = 256, gridsize = range*N/256
__global__ void cal_d2Tilde(uint64_t *d_d2Tilde, uint64_t *d_sum, uint64_t *d_moduliQP, int N, int in_C_L_index) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / N;
    int j = idx % N;
    d_d2Tilde[i + in_C_L_index][j] = mod_128_64(d_sum[i][j], d_moduliQP[i][j]);
}

void modup_core_kernel(uint64_t *d_intt_a, uint64_t *d_d2Tilde,
                       uint64_t *d_moduliQ, uint64_t *d_moduliP, uint64_t *d_QHatInvModq, uint64_t *d_QHatModp,
                       int curr_limbs, int K, int N, int in_C_L_len, int in_C_L_index, int sizeP, int beta,
                       int range_size) {
    uint64_t *d_sum;
    cudaMalloc((void **) &d_sum, sizeP * N * sizeof(uint64_t));
    int blockSize = 256;
    int numBlocks = in_C_L_len * N / blockSize;
    modup_kernel<<<numBlocks, blockSize>>>(
            d_sum, d_intt_a, in_C_L_index, curr_limbs, N, sizeP,
                    d_QHatInvModq,
                    d_moduliQ,
                    d_QHatModp, beta,
                    in_C_L_len);

    int numBlocks2 = range_size * N / blockSize
    cal_d2Tilde<<<numBlocks, blockSize2>>>(
            d_d2Tilde, d_sum, d_moduliQP, N, in_C_L_index);

}


void modup_core_cuda(
        uint64_t *h_intt_a, uint64_t *h_d2Tilde,
        uint64_t *h_moduliQ, uint64_t *h_moduliP, uint64_t *h_QHatInvModq, uint64_t *h_QHatModp,
        int curr_limbs, int K, int N) {


    int beta = static_cast<int>(ceil(static_cast<double>(curr_limbs) / K));  // total beta groups

    for (int j = 0; j < beta; ++j) {
        int in_C_L_index = j * K;
        int in_C_L_len = (j < (beta - 1)) ? K : (curr_limbs - in_C_L_index);
        int sizeP = curr_limbs - in_C_L_len + K;

        uint64_t a = intt_a + in_C_L_index;

        vector<int> qi(h_moduliQ.begin() + in_C_L_index, h_moduliQ.begin() + in_C_L_index + in_C_L_len);

        vector<int> qi_comple(h_moduliQ.begin(), h_moduliQ.begin() + in_C_L_index);
        qi_comple.insert(qi_comple.end(), h_moduliQ.begin() + in_C_L_index + in_C_L_len, h_moduliQ.end());
        vector<int> moduliQP = qi_comple;
        moduliQP.insert(moduliQP.end(), h_moduliP.begin(), h_moduliP.end());

        assert(moduliQP.size() == sizeP && "moduliQP.size() should equal to sizeP, check again");

//        vector<int> ranges;
//        for (int i = 0; i < in_C_L_index; ++i) {
//            ranges.push_back(i);
//        }
//        for (int i = in_C_L_index + in_C_L_len; i < curr_limbs + K; ++i) {
//            ranges.push_back(i);
//        }
        int ranges_size = curr_limbs + K - in_C_L_len;
        modup_core_kernel(d_intt_a, d_d2Tilde,
                          d_moduliQ, d_moduliP, d_QHatInvModq, d_QHatModp,
                          curr_limbs, K, N, in_C_L_len, in_C_L_index, sizeP, beta, range_size);
    }


    uint64_t *d_intt_a;
    uint64_t *d_d2Tilde;
    uint64_t *d_moduliQ;
    uint64_t *d_moduliP;
    uint64_t *d_QHatInvModq;
    uint64_t *d_QHatModp;

    int beta = (int) ceil((float) curr_limbs / K);
    int ranges_size = curr_limbs + K;

    // 分配设备内存
    cudaMalloc((void **) &d_intt_a, curr_limbs * N * sizeof(uint64_t));
    cudaMalloc((void **) &d_d2Tilde, beta * ranges_size * N * sizeof(uint64_t));
    cudaMalloc((void **) &d_moduliQ, curr_limbs * sizeof(uint64_t));
    cudaMalloc((void **) &d_moduliP, K * sizeof(uint64_t));
    cudaMalloc((void **) &d_QHatInvModq, beta * curr_limbs * sizeof(uint64_t));
    cudaMalloc((void **) &d_QHatModp, curr_limbs * beta * ranges_size * sizeof(uint64_t));

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
    uint64_t h_d2Tilde[(int) ceil((float) curr_limbs / K) * (curr_limbs + K) * N];
    uint64_t h_moduliQ[curr_limbs];
    uint64_t h_moduliP[K];
    uint64_t h_QHatInvModq[(int) ceil((float) curr_limbs / K) * curr_limbs];
    uint64_t h_QHatModp[curr_limbs * (int) ceil((float) curr_limbs / K) * (curr_limbs + K)];

    // 初始化输入数据
    // ...

    // 调用modup_core函数
    modup_core_cuda(h_intt_a, h_d2Tilde, h_moduliQ, h_moduliP, h_QHatInvModq, h_QHatModp, curr_limbs, K, N);

    // 打印结果或进一步处理
    // ...

    return 0;
}
