import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
from . import homo_ops
from . import arithmetic
from . import KeySwitch
from .data import params_N256_L4_P1 as N256L4P1
from .data import params_N256_L4_P2 as N256L4P2
from .data import params_N64 as N64
Tensor = torch.Tensor


def compare_and_print(res, golden, test_name):
    compare = np.array_equal(res, golden)
    # compare = res == golden_answer
    print(f"\ntest {test_name}: \nresult: ")
    print(compare)

def test_KS_ModDown2():
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

    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                               QHatInvModq, moduliP, pInvVec, pRootScalePows, QHatModp, L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)

    res_0 = KeySwitch.ModDown_method2(sumMult[0], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                      pHatModq,
                               PInvModq, moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp, L, K, N)

    res_1 = KeySwitch.ModDown_method2(sumMult[1], moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                      pHatModq,
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

    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res_0 = KeySwitch.ModDown_method2(sumMult[0],
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq, PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    res_1 = KeySwitch.ModDown_method2(sumMult[1],
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

def test_KS_components():
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

    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

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

    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

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

def test_KS_ModDown_ct():
    #test case N256_L4_P2
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

    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq

    d2Tilde = KeySwitch.Decompose(axax, L, K, N)
    d2Tilde = KeySwitch.ModUp(axax, d2Tilde,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, QHatInvModq,
                               moduliP, pInvVec, pRootScalePows, QHatModp,
                               L, K, N)
    sumMult = KeySwitch.InnerProduct(d2Tilde, swk,
                                      moduliQ, moduliP,
                                      L, K, N)
    res = KeySwitch.ModDown_ct(sumMult,
                               moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq, pHatModq,PInvModq,
                               moduliP, pInvVec, pRootScalePowsInv, NScaleInvModp, pHatInvModp,
                               L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_KS_ct():
    #test case N256_L4_P2
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
                            moduliQ, moduliP, rootsQ, rootsP, swk)

    mult_swk = cryptoContext.mult_swk
    qInvVec = cryptoContext.qInvVec
    pInvVec = cryptoContext.pInvVec
    qRootScalePows = cryptoContext.qRootScalePows
    pRootScalePows = cryptoContext.pRootScalePows
    qRootScalePowsInv = cryptoContext.qRootScalePowsInv
    pRootScalePowsInv = cryptoContext.pRootScalePowsInv
    NScaleInvModq = cryptoContext.NScaleInvModq
    NScaleInvModp = cryptoContext.NScaleInvModp
    QHatInvModq = cryptoContext.PartQlHatInvModq
    QHatModp = cryptoContext.PartQlHatModp
    pHatInvModp = cryptoContext.pHatInvModp
    pHatModq = cryptoContext.pHatModq
    PInvModq = cryptoContext.PInvModq


    res = KeySwitch.KeySwitch_core(axax, mult_swk, moduliQ, qInvVec, qRootScalePows, qRootScalePowsInv, NScaleInvModq,
                                   QHatInvModq, pHatModq, PInvModq, moduliP, pInvVec, pRootScalePows, pRootScalePowsInv,
                                   QHatModp, NScaleInvModp, pHatInvModp, L, K, N)

    golden_answer = N256L4P2.res[0]
    golden_answer = golden_answer.reshape(res[0].shape)
    compare = np.array_equal(res[0], golden_answer)
    # compare = res == golden_answer
    print("\n\ntest 2: \n\nres_ax result: ")
    print(compare)
    print("\n")

    golden_answer = N256L4P2.res[1]
    golden_answer = golden_answer.reshape(res[1].shape)
    compare = np.array_equal(res[1], golden_answer)
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_SwitchModulus():
    print("------------------")
    print("test_SwitchModulus")
    print("------------------")
    # N=64
    ax = N64.ct0_before_SwitchModulus
    bx = N64.ct1_before_SwitchModulus
    golden_ax = N64.ct0_after_SwitchModulus
    golden_bx = N64.ct1_after_SwitchModulus
    res_ax = arithmetic.vec_switch_modulus(ax, 36028797018918401, 1152921504606844417)
    res_bx = arithmetic.vec_switch_modulus(bx, 36028797018918401, 1152921504606844417)
    compare_and_print(res_ax, golden_ax, "res_ax")
    compare_and_print(res_bx, golden_bx, "res_bx")

def test_homo_add():
    print("-------------")
    print("test_homo_add")
    print("-------------")
    L=2
    K=1
    a = torch.tensor([[[1, 2, 3, 4], [4, 5, 6, 4]],[[7, 8, 9, 4], [0, 2, 0, 4]]], dtype=torch.uint64, device='cuda')
    b = torch.tensor([[[4, 5, 6, 4], [4, 5, 6, 4]],[[4, 5, 6, 4], [4, 5, 6, 4]]], dtype=torch.uint64, device='cuda')
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    in0 = Ciphertext(a, L)
    in1 = Ciphertext(b, L)
    cryptoContext = Context(2, 53, 52, 52, L, K)
    cryptoContext.moduliQ = np.array([10,11], dtype=np.uint64)
    ct_ = homo_ops.homo_add(in0, in1, cryptoContext)
    golden_answer = torch.tensor([[[ 5,  7,  9,  8], [ 8, 10,  1,  8]], [[ 1,  3,  5,  8], [ 4,  7, 6, 8]]], dtype = torch.uint64)
    ct_.cv = torch.from_numpy(ct_.cv)
    correct = torch.equal(golden_answer, ct_.cv)
    # print(ct_.cv)
    print("test_homo_add")
    print(correct)
    print("\n")


def test_HMultCore():
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
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
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    q_mu = cryptoContext.q_mu
    res = homo_ops.homo_mult_core(ct, ct, moduliQ, q_mu)

    golden_answer = np.array(N256L4P2.cipher1_after_multCore, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res[:2].shape)

    compare = np.array_equal(res[0], golden_answer[0])
    # compare = res == golden_answer
    print("test_HMultCore")
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res[2], N256L4P2.axax_)
    # compare = res == golden_answer
    print("\nres_cx result: ")
    print(compare)
    print("\n")

def test_HMult():

    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
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
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    res = homo_ops.homo_mult(ct, ct, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_rescale_ct():
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)
    ct = Ciphertext(cv, L)
    res = homo_ops.rescale_ct(ct, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_rescale, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)
    compare = np.array_equal(res.cv[0], golden_answer[0])
    print("\n\ntest 2: \n\n res_ax result: ")
    print(compare)
    print("\n")

    compare = np.array_equal(res.cv[1], golden_answer[1])
    print("\nres_bx result: ")
    print(compare)
    print("\n")

def test_HMult_and_rescale_1():
    print("------------------------")
    print("test_HMult_and_rescale_1")
    print("------------------------")
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
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
                            moduliQ, moduliP, rootsQ, rootsP, swk)
    ct = Ciphertext(cv, L)
    mult = homo_ops.homo_mult(ct, ct, cryptoContext)
    res = homo_ops.rescale_ct(mult, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_rescale, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)

    compare = np.array_equal(res.cv[0], golden_answer[0])
    # compare = res == golden_answer
    print("test 2: \n\nres_ax result: ")
    print(compare)

    compare = np.array_equal(res.cv[1], golden_answer[1])
    # compare = res == golden_answer
    print("res_bx result: ")
    print(compare)
    print("\n")

def test_ModReduce_ct():
    print("------------------")
    print("test_ModReduce_ct")
    print("------------------")
    #test case: N256_L4_P2
    logN = 8
    N = 2**logN
    L = 4
    K = 2
    cv = np.array(N256L4P2.cipher1_after_mult, dtype=np.uint64)
    cv = cv.reshape(2, L, N)
    moduliQ = N256L4P2.moduliQ4_N256
    moduliP = N256L4P2.moduliP2_N256
    rootsQ = N256L4P2.rootsQ4_N256
    rootsP = N256L4P2.rootsP2_N256
    cryptoContext = Context(logN, 53, 52, 52, L, K,
                            moduliQ, moduliP, rootsQ, rootsP)
    ct = Ciphertext(cv, L)
    res = homo_ops.ModReduce_ct(ct, 1, cryptoContext)

    golden_answer = np.array(N256L4P2.cipher1_after_mult_ModReduce, dtype=np.uint64)
    golden_answer = golden_answer.reshape(res.cv.shape)
    compare = np.array_equal(res.cv[0], golden_answer[0])
    print("test 2: \n\nres_ax result: ")
    print(compare)

    compare = np.array_equal(res.cv[1], golden_answer[1])
    print("res_bx result: ")
    print(compare)