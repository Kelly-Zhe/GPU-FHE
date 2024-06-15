import numpy as np
import math
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

def reverse_bits(num, msb):
    msbb = (msb >> 3) + (1 if msb & 0x7 else 0)

    def reverse_byte(byte):
        return int('{:08b}'.format(byte)[::-1], 2)

    shift_trick = {
        0: 0,
        1: 7,
        2: 6,
        3: 5,
        4: 4,
        5: 3,
        6: 2,
        7: 1
    }

    if msbb == 1:
        return (reverse_byte(num & 0xff) >> shift_trick[msb & 0x7])
    elif msbb == 2:
        return (reverse_byte(num & 0xff) << 8 | reverse_byte((num >> 8) & 0xff)) >> shift_trick[msb & 0x7]
    elif msbb == 3:
        return (reverse_byte(num & 0xff) << 16 | reverse_byte((num >> 8) & 0xff) << 8 |reverse_byte((num >> 16) & 0xff)) >> shift_trick[msb & 0x7]
    elif msbb == 4:
        return (reverse_byte(num & 0xff) << 24 | reverse_byte((num >> 8) & 0xff) << 16 |reverse_byte((num >> 16) & 0xff) << 8 | reverse_byte((num >> 24) & 0xff)) >> shift_trick[msb & 0x7]
    else:
        return -1
        # Handle the case for msbb values greater than 4 if necessary.


def precompute_auto_map(n, k, precomp):
    m = n << 1  # cyclOrder
    logm = int(math.log2(m))
    logn = int(math.log2(n))
    for j in range(n):
        j_tmp = ((j << 1) + 1)
        idx = ((j_tmp * k) - (((j_tmp * k) >> logm) << logm)) >> 1
        jrev = reverse_bits(j, logn)
        idxrev = reverse_bits(idx, logn)
        precomp[jrev] = idxrev

def test_reverse_bits():
    # Example usage:
    num = 0b1111111000111001
    msb = 16
    print( format(num, '16b') )
    print(format(reverse_bits(num, msb), '16b'))

    num = 0b1010101010101010
    print( format(num, '16b') )
    print(format(reverse_bits(num, msb), '16b'))

def test_precompute_auto_map():
    n = 512
    k = 25
    precomp = np.zeros(n, dtype = 'uint64')
    precompute_auto_map(n, k, precomp)

    golden_answer = np.array([
        96,97,98,99,100,101,102,103,105,104,107,106,109,108,111,110,115,114,112,113,119,118,116,117,122,123,121,120,126,127,125,124,86,87,85,84,81,80,83,82,95,94,92,93,88,89,90,91,76,77,78,79,74,75,73,72,67,66,64,65,71,70,68,69,28,29,30,31,26,27,25,24,19,18,16,17,23,22,20,21,0,1,2,3,4,5,6,7,9,8,11,10,13,12,15,14,57,56,59,58,61,60,63,62,52,53,54,55,50,51,49,48,38,39,37,36,33,32,35,34,47,46,44,45,40,41,42,43,248,249,250,251,252,253,254,255,245,244,247,246,243,242,240,241,231,230,228,229,224,225,226,227,238,239,237,236,233,232,235,234,193,192,195,194,197,196,199,198,200,201,202,203,204,205,206,207,210,211,209,208,214,215,213,212,219,218,216,217,223,222,220,221,138,139,137,136,142,143,141,140,135,134,132,133,128,129,130,131,152,153,154,155,156,157,158,159,149,148,151,150,147,146,144,145,173,172,175,174,171,170,168,169,162,163,161,160,166,167,165,164,190,191,189,188,185,184,187,186,176,177,178,179,180,181,182,183,328,329,330,331,332,333,334,335,325,324,327,326,323,322,320,321,347,346,344,345,351,350,348,349,342,343,341,340,337,336,339,338,366,367,365,364,361,360,363,362,352,353,354,355,356,357,358,359,380,381,382,383,378,379,377,376,371,370,368,369,375,374,372,373,290,291,289,288,294,295,293,292,299,298,296,297,303,302,300,301,304,305,306,307,308,309,310,311,313,312,315,314,317,316,319,318,277,276,279,278,275,274,272,273,284,285,286,287,282,283,281,280,270,271,269,268,265,264,267,266,256,257,258,259,260,261,262,263,468,469,470,471,466,467,465,464,477,476,479,478,475,474,472,473,463,462,460,461,456,457,458,459,449,448,451,450,453,452,455,454,497,496,499,498,501,500,503,502,504,505,506,507,508,509,510,511,490,491,489,488,494,495,493,492,487,486,484,485,480,481,482,483,442,443,441,440,446,447,445,444,439,438,436,437,432,433,434,435,420,421,422,423,418,419,417,416,429,428,431,430,427,426,424,425,387,386,384,385,391,390,388,389,394,395,393,392,398,399,397,396,401,400,403,402,405,404,407,406,408,409,410,411,412,413,414,415
    ], dtype = 'uint64')
    
    compare = np.array_equal(precomp, golden_answer)
    print(compare)

