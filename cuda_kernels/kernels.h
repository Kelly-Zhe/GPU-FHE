//
// Created by wongw on 2024/7/15.
//

#ifndef GPU_FHE_KERNELS_H
#define GPU_FHE_KERNELS_H
#include <cstdint>

void modup_core_cuda(
        uint64_t* h_intt_a, uint64_t* h_d2Tilde,
        uint64_t* h_moduliQ, uint64_t* h_moduliP, uint64_t* h_QHatInvModq, uint64_t* h_QHatModp,
        int curr_limbs, int K, int N);


void moddown_core_cuda(
        uint64_t* h_intt_a, uint64_t* h_pHatInvModp, uint64_t* h_pHatModq, uint64_t* h_PInvModq,
        uint64_t* h_moduliQ, uint64_t* h_moduliP,
        uint64_t* h_res, int N, int curr_limbs, int K) ;

#endif //GPU_FHE_KERNELS_H
