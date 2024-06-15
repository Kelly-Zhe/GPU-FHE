import math
import numpy as np
import arithmetic
import data.params_N1024_L4_P1 as N1024

def test_ntt():
    # test 1
    # a = np.zeros(1024,dtype = np.uint64)
    # a[0]=1

    # test 2
    a = np.arange(1024, dtype='uint64')

    copy_a = a
    N=1024
    moduli = N1024.moduliQ4_N1024[0]
    moduli_Inv = N1024.qInvVec4_N1024[0]
    RootScalePows = N1024.qRootScalePows_N1024[0]

    moduli_double = N1024.qdVec4_N1024[0]
    RootScalePowsInv = N1024.qRootScalePowsInv_N1024[0]
    NScaleInvModq = N1024.NScaleInvModq4_N1024[0]

    arithmetic.NTT(a, N, moduli, moduli_Inv, RootScalePows)
    arithmetic.iNTT(a, N, moduli, moduli_double, moduli_Inv, RootScalePowsInv, NScaleInvModq)

    # print(a)

    compare = np.array_equal(a, copy_a)
    print(compare)