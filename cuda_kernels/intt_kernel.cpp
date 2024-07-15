#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <cmath>

__global__ void iNTTKernel(uint64_t* a, int N, uint64_t moduli, uint64_t moduli_Inv, uint64_t* RootScalePowsInv, uint64_t NScaleInvModq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint64_t q = moduli;
    uint64_t qd = q * 2;
    uint64_t qInv = moduli_Inv;
    int t = 1;
    int m = N;

    while (m > 1) {
        int j1 = 0;
        int h = m >> 1;
        for (int i = 0; i < h; ++i) {
            int j2 = j1 + t - 1;
            uint64_t W = RootScalePowsInv[h + i];
            W = W & 0xFFFFFFFFFFFFFFFF;
            for (int j = j1; j <= j2; ++j) {
                uint64_t T = a[j] + qd;
                T = T & 0xFFFFFFFFFFFFFFFF;
                T -= a[j + t];
                a[j] += a[j + t];
                if (a[j] >= qd) {
                    a[j] -= qd;
                }
                uint64_t UU = T * W;
                uint64_t U0 = UU & 0xFFFFFFFFFFFFFFFF;
                uint64_t U1 = UU >> 64;
                U1 = U1 & 0xFFFFFFFFFFFFFFFF;
                uint64_t Q = U0 * qInv;
                Q = Q & 0xFFFFFFFFFFFFFFFF;
                uint64_t Hx = Q * q;
                uint64_t H = Hx >> 64;
                a[j + t] = U1 + q;
                a[j + t] -= H;
            }
            j1 += (t << 1);
        }
        t <<= 1;
        m >>= 1;
    }

    uint64_t NScale = NScaleInvModq;
    NScale = NScale & 0xFFFFFFFFFFFFFFFF;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        uint64_t T = a[i] < q ? a[i] : a[i] - q;
        T = T & 0xFFFFFFFFFFFFFFFF;
        uint64_t U = T * NScale;
        uint64_t U0 = U & 0xFFFFFFFFFFFFFFFF;
        uint64_t U1 = U >> 64;
        uint64_t Q = U0 * qInv;
        Q = Q & 0xFFFFFFFFFFFFFFFF;
        uint64_t Hx = Q * q;
        uint64_t H = Hx >> 64;
        if (U1 < H) {
            a[i] = U1 + q - H;
        } else {
            a[i] = U1 - H;
        }
    }
}

int main() {
    const int N = 1024;  // Size of the input array
    uint64_t moduli = 1000000007;
    uint64_t moduli_Inv = 123456789;  // Placeholder value
    uint64_t NScaleInvModq = 987654321;  // Placeholder value
    uint64_t* RootScalePowsInv = new uint64_t[2 * N];  // Placeholder values

    for (int i = 0; i < 2 * N; ++i) {
        RootScalePowsInv[i] = 1;  // Initialize with placeholder values
    }

    uint64_t* a = new uint64_t[N];
    for (int i = 0; i < N; ++i) {
        a[i] = rand() % moduli;  // Initialize with random values
    }

    uint64_t* d_a;
    uint64_t* d_RootScalePowsInv;
    cudaMalloc(&d_a, N * sizeof(uint64_t));
    cudaMalloc(&d_RootScalePowsInv, 2 * N * sizeof(uint64_t));

    cudaMemcpy(d_a, a, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RootScalePowsInv, RootScalePowsInv, 2 * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    iNTTKernel<<<grid_size, block_size>>>(d_a, N, moduli, moduli_Inv, d_RootScalePowsInv, NScaleInvModq);

    cudaMemcpy(a, d_a, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    delete[] a;
    delete[] RootScalePowsInv;
    cudaFree(d_a);
    cudaFree(d_RootScalePowsInv);

    return 0;
}
