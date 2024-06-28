import arithmetic
import numpy as np
import math

def Decompose(axax, curr_limbs, K, N):
    beta = math.ceil(curr_limbs / K)  # total beta groups
    d2Tilde = np.zeros((beta, curr_limbs + K, N), dtype=np.uint64)

    for j in range(beta):
        start = K * j
        end = min(start + K, curr_limbs)
        d2Tilde[j, start:end, :] = axax[start:end, :]

    return d2Tilde

def InnerProduct(d2Tilde, key,
                 moduliQ, moduliP,
                 curr_limbs, K, N):

    sumMult = np.zeros((2, curr_limbs + K, N), dtype=np.uint64)
    product = np.zeros((2, curr_limbs + K, N), dtype=np.uint64)
    moduliQP = np.concatenate((moduliQ, moduliP))
    beta = math.ceil(curr_limbs / K)

    for k in range(2): # deal with 2 dim of a ct
        for j in range(beta):
            for i in range(len(moduliQP)):
                product[k][i] = arithmetic.vec_mul_mod(d2Tilde[j][i], key[k][j][i], moduliQP[i])
                sumMult[k][i] = arithmetic.vec_add_mod(sumMult[k][i], product[k][i], moduliQP[i])

    return sumMult


def ModUp(a, d2Tilde,
          moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
          moduliP, pInvVec, pRootScalePows, QHatModp,
          curr_limbs, K, N):

    intt_a = np.zeros((curr_limbs, N), np.uint64)
    beta = math.ceil(curr_limbs / K)

    for i in range(curr_limbs):
        intt_a[i] = arithmetic.iNTT(a[i], N, moduliQ[i], qInvVec[i], qRootScalePowsInv[i], NScaleInvModq[i])


    arithmetic.ModUp_Core(intt_a, d2Tilde,
               moduliQ, moduliP, QHatInvModq, QHatModp,
               curr_limbs, K, N)

    moduliQP = np.concatenate((moduliQ, moduliP))
    qpInv = np.concatenate((qInvVec, pInvVec))
    qpRSP = np.concatenate((qRootScalePows, pRootScalePows))
    for j in range(beta):
        in_C_L_index = j * K
        in_C_L_len = K if j < (beta - 1) else (curr_limbs - in_C_L_index)

        ranges = list(range(0, in_C_L_index)) + list(range(in_C_L_index + in_C_L_len, curr_limbs + K))
        for i in ranges:
            d2Tilde[j][i] = arithmetic.NTT(d2Tilde[j][i], N, moduliQP[i], qpInv[i], qpRSP[i])

    return d2Tilde


def ModDown(a,
            moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
            moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
            curr_limbs, K, N):

    intt_a = np.zeros((curr_limbs+K, N), np.uint64)

    moduliQP = np.concatenate((moduliQ, moduliP))
    qpInv = np.concatenate((qInvVec, pInvVec))
    qpRSPInv = np.concatenate((qRootScalePowsInv, pRootScalePowsInv))
    qpNScaleInvMod = np.concatenate((NScaleInvModq, NScaleInvModp))
    for i in range(curr_limbs+K):
        intt_a[i] = arithmetic.iNTT(a[i], N, moduliQP[i], qpInv[i], qpRSPInv[i], qpNScaleInvMod[i])


    res = arithmetic.ModDown_Core(intt_a,
                 pHatInvModp, pHatModq, PInvModq,
                 moduliQ, moduliP,
                 N, curr_limbs, K,)

    for i in range(curr_limbs):
        res[i] = arithmetic.NTT(res[i], N, moduliQ[i], qInvVec[i], qRootScalePows[i])

    return res
