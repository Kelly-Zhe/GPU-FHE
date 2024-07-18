#include <iostream>
#include <cuda_runtime.h>
#include<cmath>

struct uint128_t {
    uint64_t hi;
    uint64_t lo;
    __device__ uint128_t& operator+=(const uint128_t& op);
};
// https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520
__inline__ __device__ uint128_t mult_64_64_128(const uint64_t op1,
                                               const uint64_t op2) {
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

__inline__ __device__ void inplace_add_128_128(const uint128_t op1,
                                               uint128_t& res) {
    asm("add.cc.u64 %1, %3, %1;\n\t"
        "addc.cc.u64 %0, %2, %0;\n\t"
            : "+l"(res.hi), "+l"(res.lo)
            : "l"(op1.hi), "l"(op1.lo));
}

__inline__ __device__ void mod_128_64(const uint128_t op1, uint64_t mod){
    return ((op1.hi % mod) << 64 | op1.lo) % mod;
}

void mul_mod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
    unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
    mul %= static_cast<unsigned __int128>(m);
    r = static_cast<uint64_t>(mul);
}

__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b,
                                            uint64_t mod) {
    uint64_t res = a + b;
    return res >= mod ? res - mod : res;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b,
                                            uint64_t mod) {
    return a >= b ? a - b : a + mod - b;
}



__global__ void constMulMod_kernel(size_t K, size_t N, uint64_t *in, uint64_t *pHatInvModp, uint64_t *moduliP, uint64_t *out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int k=0;k<K;k++){
        mul_mod(out[k*N+i],in[k*N+i],pHatInvModp[k], moduliP[k]);
    }
}

void constMulMod(size_t K, size_t N, size_t curr_limbs, uint64_t *in, uint64_t *pHatInvModp, uint64_t *moduliP, uint64_t *out){
    const int block_dim = 256;
    const int grid_dim = K * N / block_dim;
    constMulMod_kernel<<<grid_dim, block_dim>>>(K,N,in, pHatInvModp,moduliP,out);
}


__global__ void ModDownKernel(uint64_t *intt_a, uint64_t *tmp3,uint64_t *tmpi,
                              const uint64_t *pHatInvModp, const uint64_t *pHatModq,
                              const uint64_t *PInvModq, const uint64_t *moduliQ,
                              const uint64_t *moduliP,
                              int curr_limbs, int K, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx/N;
    int col = idx%N;

    for (int i = 0; i < curr_limbs; ++i) {
        uint128_t sum[N] = {0};
        for (int k = 0; k < K; ++k) {
            uint128_t product = mult_64_64_128(&tmp3[k * N+col], pHatModq[k * curr_limbs + i]);
            inplace_add_128_128(product, sum[col]);
        }

        res[i * N + col] = mod_128_64(sum[col], moduliQ[i]); //这里其实需要用barret算法
        res[i * N + col] = sub_mod(&tmpi[i * N+col], &res[i * N+col], moduliQ[i]);
        mul_mod(&res[i * N+ col], &res[i * N+col], PInvModq[i], moduliQ[i]);
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
    cudaMalloc(&d_intt_a, N * curr_limbs * sizeof(uint64_t));
    cudaMalloc(&d_pHatInvModp, K * sizeof(uint64_t));
    cudaMalloc(&d_pHatModq, K * curr_limbs * sizeof(uint64_t));
    cudaMalloc(&d_PInvModq, curr_limbs * sizeof(uint64_t));
    cudaMalloc(&d_moduliQ, curr_limbs * sizeof(uint64_t));
    cudaMalloc(&d_moduliP, K * sizeof(uint64_t));

// 将输入数据从主机传输到设备
    cudaMemcpy(d_intt_a, intt_a, N * curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pHatInvModp, pHatInvModp, K * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pHatModq, pHatModq, K * curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PInvModq, PInvModq, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliQ, moduliQ, curr_limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduliP, moduliP, K * sizeof(uint64_t), cudaMemcpyHostToDevice);
// 设置CUDA内核的执行配置
    int blockSize = 256;
    int numBlocks = (N * (K+curr_limbs) + blockSize - 1) / blockSize;
    // size_t shared_memory_size = (3*N*(K+curr_limbs)) * sizeof(uint64_t);

// 启动CUDA内核
    moddown_core_kernel<<<numBlocks, blockSize>>>(
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
    const int N = 256;
    const int curr_limbs = 4;
    const int K = 1;

    uint64_t h_intt_a[(curr_limbs + K) * N];
    uint64_t h_pHatInvModp[K * (curr_limbs+K)] ={1,1,1,1,1};
    uint64_t h_pHatModq[K * (curr_limbs+K)] = {1,1,1,1,0};
    uint64_t h_PInvModq[curr_limbs + K] = {8309272629270236,399822410100,3503288382006841,8796093022179,0};
    uint64_t h_moduliQ[curr_limbs + K]{9007199254746113,4503599627366401,4503599627364353,4503599627355649,0};
    uint64_t h_moduliP[K * (curr_limbs+K)] {4503599627355137,4503599627355137,4503599627355137,4503599627355137,4503599627355137};
    uint64_t *h_res = (uint64_t*)malloc(sizeof(int64_t) * ((curr_limbs+K) * N));
    for(int i =0; i<(curr_limbs+K)*N;i++){
        h_intt_a[i] = 8309272629270236;

        // h_intt_a[i] = (rand()<<30) + (rand()<<15)+rand();
    }

// 初始化输入数据
// ...

// 调用moddown_core函数
    moddown_core_cuda(h_intt_a, h_pHatInvModp, h_pHatModq, h_PInvModq,
                      h_moduliQ, h_moduliP, h_res, N, curr_limbs, K);

// 打印结果或进一步处理
    for(int i = 0;i <curr_limbs * N;i++){
        // std::cout<<i<<":"<<h_res[i] <<", ";
    }

    return 0;
}