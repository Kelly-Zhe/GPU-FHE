import numpy as np
import math

np.seterr(over='ignore')


def vec_add(a, b, MOD):
    assert a.shape == b.shape
    return (a + b) % MOD


def vec_sub(a, b, MOD):
    assert a.shape == b.shape
    return (a - b) % MOD


def vec_mul(a, b, MOD):
    assert a.shape == b.shape
    return (a * b) % MOD


def NTT(a, N, moduli, moduli_Inv, RootScalePows):
    t = N
    logt1 = int(math.log2(N)) + 1
    q = int(moduli)  # np
    qInv = int(moduli_Inv)  # np
    m = 1
    while m < N:
        t >>= 1
        logt1 -= 1
        for i in range(m):
            j1 = i << logt1
            j2 = j1 + t - 1
            W = RootScalePows[m + i]  # np
            for j in range(j1, j2 + 1):
                T = a[j + t]  # np
                U = int(T) * int(W)  # np
                U0 = U & 0xFFFFFFFFFFFFFFFF
                U0 = np.uint64(U0)
                U1 = U >> 64
                U1 = np.uint64(U1)
                Q = int(U0) * qInv
                Q = Q & 0xFFFFFFFFFFFFFFFF
                Hx = int(Q) * q
                H = Hx >> 64
                H = np.uint64(H)
                if U1 < H:
                    V = int(U1) + q - int(H)
                    V = np.uint64(V)
                else:
                    V = int(U1) - int(H)
                    V = np.uint64(V)
                if a[j] < V:
                    tmp = int(a[j]) + q - int(V)
                    a[j + t] = np.uint64(tmp)
                else:
                    a[j + t] = np.uint64(int(a[j]) - int(V))
                tmp = int(a[j])+int(V)
                if tmp > q:
                    a[j] = np.uint64(tmp-q)
                else:
                    a[j] = np.uint64(tmp)
        m = m << 1
    return a


def iNTT(a, N, moduli, moduli_double, moduli_Inv, RootScalePowsInv, NScaleInvModq):
    q = moduli
    qd = moduli_double
    qInv = moduli_Inv
    t = 1
    m = N
    while m > 1:
        j1 = 0
        h = m >> 1
        for i in range(h):
            j2 = j1 + t - 1
            W = int(RootScalePowsInv[h + i])
            W = W & 0xFFFFFFFFFFFFFFFF
            for j in range(j1, j2 + 1):
                T = a[j] + qd
                T = T & 0xFFFFFFFFFFFFFFFF
                T -= a[j + t]
                a[j] += a[j + t]
                if a[j] >= qd:
                    a[j] -= qd
                UU = int(T) * int(W)
                U0 = UU & 0xFFFFFFFFFFFFFFFF
                U0 = np.uint64(U0)
                U1 = UU >> 64
                U1 = np.uint64(U1)
                U1 = U1 & 0xFFFFFFFFFFFFFFFF
                Q = int(U0) * int(qInv)
                Q = Q & 0xFFFFFFFFFFFFFFFF
                Hx = int(Q) * int(q)
                H = Hx >> 64
                H = np.uint64(H)
                a[j + t] = U1 + q
                a[j + t] -= H
            j1 += (t << 1)
        t <<= 1
        m >>= 1

    NScale = NScaleInvModq
    NScale = NScale & 0xFFFFFFFFFFFFFFFF
    for i in range(N):
        T = a[i] if a[i] < q else a[i] - q
        T = T & 0xFFFFFFFFFFFFFFFF
        U = int(T) * int(NScale)
        U0 = U & 0xFFFFFFFFFFFFFFFF
        U0 = np.uint64(U0)
        U1 = U >> 64
        U1 = np.uint64(U1)
        Q = U0 * qInv
        Q = Q & 0xFFFFFFFFFFFFFFFF
        Hx = int(Q) * int(q)
        H = Hx >> 64
        H = np.uint64(H)
        if U1 < H:
            a[i] = U1 + q - H
        else:
            a[i] = U1 - H
    return a
