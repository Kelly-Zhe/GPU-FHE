#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <cmath>

__global__ void NTTKernel(uint64_t* a, int N, uint64_t moduli, uint64_t moduli_Inv, uint64_t* RootScalePows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int t = N;
    int logt1 = int(log2((double)N)) + 1;
    uint64_t q = moduli;
    uint64_t qInv = moduli_Inv;
    int m = 1;

    while (m < N) {
        t >>= 1;
        logt1 -= 1;
        int i = idx / t;
        int j = idx % t;
        int j1 = i << logt1;
        int j2 = j1 + t - 1;

        if (idx >= j1 && idx <= j2) {
            uint64_t W = RootScalePows[m + i];
            uint64_t T = a[j + t];
            uint64_t U = T * W;
            uint64_t U0 = U & 0xFFFFFFFFFFFFFFFF;
            uint64_t U1 = U >> 64;
            uint64_t Q = (U0 * qInv) & 0xFFFFFFFFFFFFFFFF;
            uint64_t H = (Q * q) >> 64;

            uint64_t V = (U1 < H) ? U1 + q - H : U1 - H;

            if (a[j] < V) {
                a[j + t] = a[j] + q - V;
            } else {
                a[j + t] = a[j] - V;
            }

            uint64_t tmp = a[j] + V;
            if (tmp > q) {
                a[j] = tmp - q;
            } else {
                a[j] = tmp;
            }
        }

        m <<= 1;
    }
}

int main() {
    const int N = 1024;  // Size of the input array
    uint64_t moduli = 1000000007;
    uint64_t moduli_Inv = 123456789;  // Placeholder value
    uint64_t* RootScalePows = new uint64_t[2 * N];  // Placeholder values

    for (int i = 0; i < 2 * N; ++i) {
        RootScalePows[i] = 1;  // Initialize with placeholder values
    }

    uint64_t* a = new uint64_t[N];
    for (int i = 0; i < N; ++i) {
        a[i] = rand() % moduli;  // Initialize with random values
    }

    uint64_t* d_a;
    uint64_t* d_RootScalePows;
    cudaMalloc(&d_a, N * sizeof(uint64_t));
    cudaMalloc(&d_RootScalePows, 2 * N * sizeof(uint64_t));

    cudaMemcpy(d_a, a, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RootScalePows, RootScalePows, 2 * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    NTTKernel<<<grid_size, block_size>>>(d_a, N, moduli, moduli_Inv, d_RootScalePows);

    cudaMemcpy(a, d_a, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    delete[] a;
    delete[] RootScalePows;
    cudaFree(d_a);
    cudaFree(d_RootScalePows);

    return 0;
}
