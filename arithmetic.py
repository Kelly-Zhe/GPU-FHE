import numpy as np

def vec_add(a, b, MOD):
    assert a.shape == b.shape
    return (a + b) % MOD

def vec_sub(a, b, MOD):
    assert a.shape == b.shape
    return (a - b) % MOD

def vec_mul(a, b, MOD):
    assert a.shape == b.shape
    return (a * b) % MOD

def NTT(x):
    return x

def iNTT(x):
    return x

