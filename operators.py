import numpy as np

def add(a, b, MOD):
    assert a.shape == b.shape
    return (a + b) % MOD

def sub(a, b, MOD):
    assert a.shape == b.shape
    return (a - b) % MOD

def mul(a, b, MOD):
    assert a.shape == b.shape
    return (a * b) % MOD

def NTT(x):
    return x

def iNTT(x):
    return x

