import math
from context import Context
import numpy as np
import arithmetic
import KeySwitch
import data.params_N256_L4_P1 as N256L4P1
import data.params_N256_L4_P2 as N256L4P2

def test_KS2():
    # test case 1
    axax = N256L4P1.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 1
    moduliQ = N256L4P1.moduliQ4_N256
    moduliP = N256L4P1.moduliP1_N256
    rootsQ = N256L4P1.rootsQ4_N256
    rootsP = N256L4P1.rootsP1_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P1.swk_bxL4P1
    for i in range(dnum):
        swk[0, i] = N256L4P1.swk_axL4P1
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)


    qInvVec = np.array(cryptoContext.qInvVec, dtype=np.uint64)
    pInvVec = np.array(cryptoContext.pInvVec, dtype=np.uint64)
    qRootScalePows = np.array(cryptoContext.qRootScalePows, dtype=np.uint64)
    pRootScalePows = np.array(cryptoContext.pRootScalePows, dtype=np.uint64)
    qRootScalePowsInv = np.array(cryptoContext.qRootScalePowsInv, dtype=np.uint64)
    pRootScalePowsInv = np.array(cryptoContext.pRootScalePowsInv, dtype=np.uint64)
    NScaleInvModq = np.array(cryptoContext.NScaleInvModq, dtype=np.uint64)
    NScaleInvModp = np.array(cryptoContext.NScaleInvModp, dtype=np.uint64)
    QHatInvModq = np.array(cryptoContext.PartQlHatInvModq, dtype = np.uint64)
    QHatModp    = np.array(cryptoContext.PartQlHatModp, dtype = np.uint64)
    pHatInvModp = np.array(cryptoContext.pHatInvModp, dtype = np.uint64)
    pHatModq    = np.array(cryptoContext.pHatModq, dtype = np.uint64)
    PInvModq    = np.array(cryptoContext.PInvModq, dtype = np.uint64)

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                               QHatInvModq, moduliP, pInvVec, pRootScalePows, QHatModp, L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)

    res_0 = KeySwitch.ModDown(sumMult[0], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    res_1 = KeySwitch.ModDown(sumMult[1], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N256L4P1.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 1: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P1.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")


    #test case 2
    axax = N256L4P2.axax_
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    dnum = int(L / K)
    swk = np.zeros((2, dnum, L + K, N), dtype=np.uint64)
    swk[1] = N256L4P2.swk_bxL4P2
    for i in range(dnum):
        swk[0, i] = N256L4P2.swk_axL4P2
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)

    qInvVec = np.array(cryptoContext.qInvVec, dtype=np.uint64)
    pInvVec = np.array(cryptoContext.pInvVec, dtype=np.uint64)
    qRootScalePows = np.array(cryptoContext.qRootScalePows, dtype=np.uint64)
    pRootScalePows = np.array(cryptoContext.pRootScalePows, dtype=np.uint64)
    qRootScalePowsInv = np.array(cryptoContext.qRootScalePowsInv, dtype=np.uint64) #??
    pRootScalePowsInv = np.array(cryptoContext.pRootScalePowsInv, dtype=np.uint64)
    NScaleInvModq = np.array(cryptoContext.NScaleInvModq, dtype=np.uint64)
    NScaleInvModp = np.array(cryptoContext.NScaleInvModp, dtype=np.uint64)
    QHatInvModq = np.array(cryptoContext.PartQlHatInvModq, dtype = np.uint64)
    QHatModp    = np.array(cryptoContext.PartQlHatModp, dtype = np.uint64)
    pHatInvModp = np.array(cryptoContext.pHatInvModp, dtype = np.uint64)
    pHatModq    = np.array(cryptoContext.pHatModq, dtype = np.uint64)
    PInvModq    = np.array(cryptoContext.PInvModq, dtype = np.uint64)

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res_0 = KeySwitch.ModDown(sumMult[0],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    res_1 = KeySwitch.ModDown(sumMult[1],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res_0.shape)
    compare = np.array_equal(res_0, golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res_1.shape)
    compare = np.array_equal(res_1, golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")
