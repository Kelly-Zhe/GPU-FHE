#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <vector>

using namespace std;

struct uint128_t
{
    uint64_t hi;
    uint64_t lo;

    __device__ uint128_t &operator+=(const uint128_t &op);
};

// https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520
__inline__ __device__ uint128_t mult_64_64_128(const uint64_t op1,
                                               const uint64_t op2)
{
    uint128_t temp;
    asm("{\n\t"
        ".reg .u32 p0l, p0h, p1l, p1h, p2l, p2h, p3l, p3h, r0, r1, r2, r3, "
        "alo, "
        "ahi, blo, bhi;\n\t"
        ".reg .u64 p0, p1, p2, p3;\n\t"
        "mov.b64         {alo,ahi}, %2;\n\t"
        "mov.b64         {blo,bhi}, %3;\n\t"
        "mul.wide.u32    p0, alo, blo;\n\t"
        "mul.wide.u32    p1, alo, bhi;\n\t"
        "mul.wide.u32    p2, ahi, blo;\n\t"
        "mul.wide.u32    p3, ahi, bhi;\n\t"
        "mov.b64         {p0l,p0h}, p0;\n\t"
        "mov.b64         {p1l,p1h}, p1;\n\t"
        "mov.b64         {p2l,p2h}, p2;\n\t"
        "mov.b64         {p3l,p3h}, p3;\n\t"
        "mov.b32         r0, p0l;\n\t"
        "add.cc.u32      r1, p0h, p1l;\n\t"
        "addc.cc.u32     r2, p1h, p2h;\n\t"
        "addc.u32        r3, p3h, 0;\n\t"
        "add.cc.u32      r1, r1, p2l;\n\t"
        "addc.cc.u32     r2, r2, p3l;\n\t"
        "addc.u32        r3, r3, 0;\n\t"
        "mov.b64         %0, {r0,r1};\n\t"
        "mov.b64         %1, {r2,r3};\n\t"
        "}"
        : "=l"(temp.lo), "=l"(temp.hi)
        : "l"(op1), "l"(op2));
    return temp;
}

__inline__ __device__ void mul_mod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m)
{
    unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
    mul %= static_cast<unsigned __int128>(m);
    r = static_cast<uint64_t>(mul);
}

__inline__ __device__ void inplace_add_128_128(const uint128_t op1,
                                               uint128_t &res)
{
    asm("add.cc.u64 %1, %3, %1;\n\t"
        "addc.cc.u64 %0, %2, %0;\n\t"
        : "+l"(res.hi), "+l"(res.lo)
        : "l"(op1.hi), "l"(op1.lo));
}

__inline__ __device__ uint64_t mod_128_64(const uint128_t op1, uint64_t mod)
{
    // return ((op1.hi % mod) << 64 | op1.lo) % mod;
    return op1.lo % mod;
}

// 计算tmp和sum
// block size = 256, girdsize = sizeP*N/256
__global__ void
modup_kernel(uint128_t *d_sum, uint64_t *d_intt_a, int in_C_L_index, int curr_limbs, int N, int sizeP,
             uint64_t *d_QHatInvModq, uint64_t *d_moduliQ, uint64_t *d_QHatModp, int beta, int total_beta,
             int in_C_L_len)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / N;
    int col = idx % N;
    uint64_t tmp[256] = {0};
    for (int i = 0; i < in_C_L_len; i++)
    {
        mul_mod(tmp[col], d_intt_a[(in_C_L_index + i) * N + col], d_QHatInvModq[beta * in_C_L_len * in_C_L_len + (in_C_L_len - 1) * in_C_L_len + i],
                d_moduliQ[in_C_L_index + i]);

        for (int k = 0; k < sizeP; ++k)
        {
            uint128_t product = mult_64_64_128(tmp[col], d_QHatModp[(curr_limbs - 1) * total_beta * in_C_L_len * sizeP + beta * in_C_L_len * sizeP + i * sizeP + k]);
            d_sum[k * N + col] = {0};
            inplace_add_128_128(product, d_sum[k * N + col]);
        }
    }
}

// blocksize = 256, gridsize = ranges_size*N/256
__global__ void cal_d2Tilde(uint64_t *d_d2Tilde, uint128_t *d_sum, uint64_t *d_moduliQP, int N, int in_C_L_index, int beta, int ranges_size, int *ranges)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / N; // range_index
    int j = idx % N; // n_index
    // printf("%lld, %lld, %d\n", d_sum[i * N + j].lo, d_moduliQP[i], beta * ranges[ranges_size - 1] * N + ranges[i] * N + j);
    // printf("%d,%d,\n",beta * ranges[ranges_size - 1] * N + ranges[i] * N + j, N);
    d_d2Tilde[beta * ranges[ranges_size - 1] * N + ranges[i] * N + j] = mod_128_64(d_sum[i * N + j], d_moduliQP[i]);
    if (beta == 0 && ranges[i] == 1)
    {
        printf("%lld,", beta * ranges[ranges_size - 1] * N + ranges[i] * N + j);
    }
}

void modup_core_kernel(uint64_t *d_intt_a, uint64_t *d_d2Tilde,
                       uint64_t *d_moduliQ, uint64_t *d_moduliP, uint64_t *d_moduliQP, uint64_t *d_QHatInvModq, uint64_t *d_QHatModp,
                       int curr_limbs, int K, int N, int in_C_L_len, int in_C_L_index, int sizeP, int beta, int total_beta,
                       int ranges_size, int *ranges)
{
    uint128_t *d_sum;
    int *d_ranges;
    cudaMalloc((void **)&d_sum, sizeP * N * sizeof(uint128_t));
    // cudaMemset(d_sum, 0, sizeP * N * sizeof(uint128_t));
    int blockSize = 256;
    int numBlocks = sizeP * N / blockSize;
    modup_kernel<<<numBlocks, blockSize>>>(
        d_sum, d_intt_a, in_C_L_index, curr_limbs, N, sizeP,
        d_QHatInvModq,
        d_moduliQ,
        d_QHatModp, beta, total_beta,
        in_C_L_len);

    cudaMalloc((void **)&d_ranges, ranges_size * sizeof(int));
    cudaMemcpy(d_ranges, ranges, ranges_size * sizeof(int), cudaMemcpyHostToDevice);
    int numBlocks2 = ranges_size * N / blockSize;
    cal_d2Tilde<<<numBlocks2, blockSize>>>(
        d_d2Tilde, d_sum, d_moduliQP, N, in_C_L_index, beta, ranges_size, d_ranges);
}

void modup_core_cuda(
    uint64_t *h_intt_a, uint64_t *h_d2Tilde,
    uint64_t *h_moduliQ, uint64_t *h_moduliP, uint64_t *h_QHatInvModq, uint64_t *h_QHatModp,
    int curr_limbs, int K, int N)
{

    uint64_t *d_intt_a;
    uint64_t *d_d2Tilde;
    uint64_t *d_moduliQ;
    uint64_t *d_moduliP;
    uint64_t *d_QHatInvModq;
    uint64_t *d_QHatModp;
    uint64_t *d_moduliQP;

    int beta = static_cast<int>(ceil(static_cast<double>(curr_limbs) / K)); // total beta groups

    // 分配设备内存
    cudaMalloc((void **)&d_intt_a, curr_limbs * N * sizeof(uint64_t));
    cudaMalloc((void **)&d_d2Tilde, beta * (curr_limbs + K) * N * sizeof(uint64_t));
    cudaMalloc((void **)&d_moduliQ, curr_limbs * sizeof(uint64_t));
    cudaMalloc((void **)&d_moduliP, K * sizeof(uint64_t));
    cudaMalloc((void **)&d_QHatInvModq, beta * curr_limbs * sizeof(uint64_t));
    cudaMalloc((void **)&d_QHatModp, curr_limbs * beta * (curr_limbs + K) * sizeof(uint64_t));
    cudaMalloc((void **)&d_moduliQP, curr_limbs * sizeof(uint64_t));

    // 将输入数据从主机传输到设备
    cudaMemcpy(d_intt_a, h_intt_a, curr_limbs * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliQ, h_moduliQ, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliP, h_moduliP, K * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QHatInvModq, h_QHatInvModq, beta * curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_QHatModp, h_QHatModp, curr_limbs * beta * (curr_limbs + K) * sizeof(uint64_t), cudaMemcpyHostToDevice);

    for (int j = 0; j < beta; ++j)
    {
        int in_C_L_index = j * K;
        int in_C_L_len = (j < (beta - 1)) ? K : (curr_limbs - in_C_L_index);
        int sizeP = curr_limbs - in_C_L_len + K;
        int ranges_size = (curr_limbs + K) - in_C_L_len;

        // uint64_t a = d_intt_a + in_C_L_index;

        vector<uint64_t> qi(h_moduliQ + in_C_L_index, h_moduliQ + in_C_L_index + in_C_L_len);

        vector<uint64_t> qi_comple(h_moduliQ, h_moduliQ + in_C_L_index);
        qi_comple.insert(qi_comple.end(), h_moduliQ + in_C_L_index + in_C_L_len, h_moduliQ + curr_limbs);
        vector<uint64_t> moduliQP = qi_comple;
        moduliQP.insert(moduliQP.end(), h_moduliP, h_moduliP + K);

        cudaMemcpy(d_moduliQP, moduliQP.data(), curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);

        assert(moduliQP.size() == sizeP && "moduliQP.size() should equal to sizeP, check again");

        vector<int> ranges;
        for (int i = 0; i < in_C_L_index; ++i)
        {
            ranges.push_back(i);
        }
        for (int i = in_C_L_index + in_C_L_len; i < curr_limbs + K; ++i)
        {
            ranges.push_back(i);
        }
        int *ranges_ptr = ranges.data();

        modup_core_kernel(d_intt_a, d_d2Tilde,
                          d_moduliQ, d_moduliP, d_moduliQP, d_QHatInvModq, d_QHatModp,
                          curr_limbs, K, N, in_C_L_len, in_C_L_index, sizeP, j, beta, ranges_size, ranges.data());
    }

    // 将结果从设备传回主机
    cudaMemcpy(h_d2Tilde, d_d2Tilde, beta * (curr_limbs + K) * N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_intt_a);
    cudaFree(d_d2Tilde);
    cudaFree(d_moduliQ);
    cudaFree(d_moduliP);
    cudaFree(d_QHatInvModq);
    cudaFree(d_QHatModp);
}

int main()
{
    // 示例输入数据和参数
    const int N = 256;
    const int curr_limbs = 4;
    const int K = 1;

    uint64_t h_intt_a[curr_limbs * N];
    uint64_t h_d2Tilde[(int)ceil((float)curr_limbs / K) * (curr_limbs + K) * N];
    uint64_t h_moduliQ[curr_limbs] = {9007199254746113, 4503599627366401, 4503599627364353, 4503599627355649};
    uint64_t h_moduliP[K] = {4503599627355137};
    uint64_t h_QHatInvModq[(int)ceil((float)curr_limbs / K) * curr_limbs] = {1, 1, 1, 1};
    uint64_t h_QHatModp[curr_limbs * (int)ceil((float)curr_limbs / K) * (curr_limbs + K)] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // 初始化输入数据
    for (int i = 0; i < curr_limbs * N; i++)
    {
        h_intt_a[i] = 3943650899777888;
    }

    // 调用modup_core函数
    modup_core_cuda(h_intt_a, h_d2Tilde, h_moduliQ, h_moduliP, h_QHatInvModq, h_QHatModp, curr_limbs, K, N);

    return 0;
}
