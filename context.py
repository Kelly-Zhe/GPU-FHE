from math import log, pi, cos, sin
import number_theory as nbtheory
import math
import random

class Context:
    #     QHatInvModq = N256L4P1.PartQlHatInvModq_L4P1N256
    #     QHatModp = N256L4P1.PartQlHatModp_L4P1N256
    #     pHatInvModp = N256L4P1.pHatInvModpP1_N256
    #     pHatModq = N256L4P1.pHatModq_N256
    #     PInvModq = N256L4P1.PInvModqP1_N256

    def __init__(self, logN, logq0, logqi, logp, L, K,
                 moduliQ = None, moduliP = None, rootsQ= None, rootsP=None,
                 h=64, sigma=32,):

        self.logN = logN
        self.logqi = logqi
        self.L = L
        self.K = K
        self.dnum = int(L / K)
        self.h = h
        self.sigma = sigma
        self.N = 1 << logN
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << logqi

        self.qVec = [0] * L
        self.qrVec = [0] * L
        self.qTwok = [0] * L
        self.qkVec = [0] * L
        self.qdVec = [0] * L
        self.qInvVec = [0] * L
        self.qRoots = [0] * L
        self.qRootsInv = [0] * L
        self.qRootPows = [[] for _ in range(L)]
        self.qRootScalePows = [[] for _ in range(L)]
        self.qRootScalePowsOverq = [[] for _ in range(L)]
        self.qRootScalePowsInv = [[] for _ in range(L)]
        self.qRootPowsInv = [[] for _ in range(L)]
        self.NInvModq = [0] * L
        self.NScaleInvModq = [0] * L
        bnd = 1
        cnt = 1
        if moduliQ is None and rootsQ is None:
            while True:
                prime = (1 << logq0) + bnd * self.M + 1
                if nbtheory.is_prime(prime):
                    self.qVec[0] = prime
                    break
                bnd += 1
            # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.qVec[i])
            self.qRoots[0] = nbtheory.root_of_unity(order=self.M, modulus=self.qVec[0])
            # print("qVec[0]", self.qVec[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << logqi) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.qVec[cnt] = prime1
                    cnt += 1
                prime2 = (1 << logqi) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.qVec[cnt] = prime2
                    # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.qVec[i])
                    self.qRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.qVec[cnt - 1])
                    cnt += 1
                bnd += 1

            if logqi - logN - 1 - math.ceil(math.log2(bnd)) < 10:
                print("ERROR: too small number of precision")
                print("TRY to use larger logqi or smaller depth")
        else:
            if moduliQ is None:
                print("moduliQ needs to be set!")
                return
            elif rootsQ is None:
                print("rootsQ needs to be set!")
                return
            for i in range(L):
                self.qVec[i] = moduliQ[i]
                self.qRoots[i] = rootsQ[i]

        for i in range(L):
            # print(i)
            self.qTwok[i] = 2 * (int(math.log2(self.qVec[i])) + 1)
            self.qrVec[i] = (1 << self.qTwok[i]) // self.qVec[i]
            self.qkVec[i] = ((nbtheory.mod_inv(1 << 62, self.qVec[i]) << 62) - 1) // self.qVec[i]
            self.qRootsInv[i] = nbtheory.mod_inv(self.qRoots[i], int(self.qVec[i]))
            self.NInvModq[i] = nbtheory.mod_inv(self.N, int(self.qVec[i]))
            self.NScaleInvModq[i] = self.mulMod(int(self.NInvModq[i]), int(1 << 32), int(self.qVec[i]))
            self.NScaleInvModq[i] = self.mulMod(int(self.NScaleInvModq[i]), int(1 << 32), int(self.qVec[i]))
            self.qInvVec[i] = self.inv(self.qVec[i])
            self.qRootPows[i] = [0] * self.N
            self.qRootPowsInv[i] = [0] * self.N
            self.qRootScalePows[i] = [0] * self.N
            self.qRootScalePowsOverq[i] = [0] * self.N
            self.qRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.qRootPows[i][jprime] = int(power)
                # tmp = (power << 64)
                tmp = (int(power) << 64)
                self.qRootScalePowsOverq[i][jprime] = int(tmp // int(self.qVec[i]))
                self.qRootScalePows[i][jprime] = int(self.mulMod(int(self.qRootPows[i][jprime]), int(1 << 32), int(self.qVec[i])))
                self.qRootScalePows[i][jprime] = int(self.mulMod(int(self.qRootScalePows[i][jprime]), int(1 << 32), int(self.qVec[i])))
                self.qRootPowsInv[i][jprime] = int(powerInv)
                self.qRootScalePowsInv[i][jprime] = int(self.mulMod(int(self.qRootPowsInv[i][jprime]), int(1 << 32), int(self.qVec[i])))
                self.qRootScalePowsInv[i][jprime] = int(self.mulMod(int(self.qRootScalePowsInv[i][jprime]), int(1 << 32), int(self.qVec[i])))
                if j < self.N - 1:
                    power = self.mulMod(int(power), int(self.qRoots[i]), int(self.qVec[i]))
                    powerInv = self.mulMod(powerInv, int(self.qRootsInv[i]), int(self.qVec[i]))

        self.pVec = [0] * self.K
        self.prVec = [0] * self.K
        self.pTwok = [0] * self.K
        self.pkVec = [0] * self.K
        self.pdVec = [0] * self.K
        self.pInvVec = [0] * self.K
        self.pRoots = [0] * self.K
        self.pRootsInv = [0] * self.K
        self.pRootPows = [[] for _ in range(self.K)]
        self.pRootPowsInv = [[] for _ in range(self.K)]
        self.pRootScalePows = [[] for _ in range(self.K)]
        self.pRootScalePowsOverp = [[] for _ in range(self.K)]
        self.pRootScalePowsInv = [[] for _ in range(self.K)]
        self.NInvModp = [0] * self.K
        self.NScaleInvModp = [0] * self.K

        if moduliP is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << logp) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.pVec[cnt] = prime1
                    self.pRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.pVec[cnt])
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << logp) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.pVec[cnt] = prime2
                    self.pRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.pVec[cnt])
                    cnt += 1
                bnd += 1

        else:
            if moduliP is None:
                print("moduliP needs to be set")
                return
            elif rootsP is None:
                print("rootsP needs to be set")
                return
            for i in range(K):
                self.pVec[i] = moduliP[i]
                self.pRoots[i] = rootsP[i]

        for i in range(K):
            # print(i)
            self.pTwok[i] = 2 * (int(math.log2(self.pVec[i])) + 1)
            self.prVec[i] = (1 << self.pTwok[i]) // self.pVec[i]
            self.pkVec[i] = ((nbtheory.mod_inv(1 << 62, self.pVec[i]) << 62) - 1) // self.pVec[i]
            self.pRootsInv[i] = nbtheory.mod_inv(self.pRoots[i], int(self.pVec[i]))
            self.NInvModp[i] = nbtheory.mod_inv(self.N, int(self.pVec[i]))
            self.NScaleInvModp[i] = self.mulMod(int(self.NInvModp[i]), int(1 << 32), int(self.pVec[i]))
            self.NScaleInvModp[i] = self.mulMod(int(self.NScaleInvModp[i]), int(1 << 32), int(self.pVec[i]))
            self.pInvVec[i] = self.inv(self.pVec[i])
            self.pRootPows[i] = [0] * self.N
            self.pRootPowsInv[i] = [0] * self.N
            self.pRootScalePows[i] = [0] * self.N
            self.pRootScalePowsOverp[i] = [0] * self.N
            self.pRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.pRootPows[i][jprime] = int(power)
                tmp = (int(power) << 64)
                self.pRootScalePowsOverp[i][jprime] = tmp // self.pVec[i]
                self.pRootScalePows[i][jprime] = self.mulMod(self.pRootPows[i][jprime], int(1 << 32), int(self.pVec[i]))
                self.pRootScalePows[i][jprime] = self.mulMod(self.pRootScalePows[i][jprime], int(1 << 32), int(self.pVec[i]))
                self.pRootPowsInv[i][jprime] = powerInv
                self.pRootScalePowsInv[i][jprime] = self.mulMod(self.pRootPowsInv[i][jprime], int(1 << 32), int(self.pVec[i]))
                self.pRootScalePowsInv[i][jprime] = self.mulMod(self.pRootScalePowsInv[i][jprime], int(1 << 32), int(self.pVec[i]))
                if j < self.N - 1:
                    power = self.mulMod(power, int(self.pRoots[i]), int(self.pVec[i]))
                    powerInv = self.mulMod(powerInv, int(self.pRootsInv[i]), int(self.pVec[i]))

        moduliPartQ = [0] * self.dnum
        for j in range(self.dnum):
            moduliPartQ[j]= int(1)
            for i in range(K*j, K*(j+1)):
                if i<L:
                    moduliPartQ[j] *= int(self.qVec[i])

        self.PartQlHatInvModq = [[[0 for _ in range(K)] for _ in range(K)] for _ in range(self.dnum)]
        for k in range(self.dnum):
            sizePartQk = (L - (k * K)) if (k == self.dnum - 1) else K
            modulusPartQ = moduliPartQ[k]
            for l in range(sizePartQk):
                if l > 0:
                    modulusPartQ = int(int(modulusPartQ) // int(self.qVec[(k + 1) * K - l]))
                for i in range(sizePartQk - l):
                    moduli = int(self.qVec[k * K + i])
                    QHat = modulusPartQ // moduli
                    QHatInvModqi = int(self.invMod(QHat, moduli))
                    self.PartQlHatInvModq[k][sizePartQk - l - 1][i] = QHatInvModqi

        # 初始化 PartQlHatModp
        self.PartQlHatModp = [[[[0 for _ in range(self.dnum * K)] for _ in range(K)] for _ in range(self.dnum)] for _ in range(L)]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            for k in range(beta):
                partQ_size = (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                digitSize = K
                modulusPartQ = int(moduliPartQ[k])

                if k == beta - 1:
                    digitSize = l + 1 - k * K
                    for idx in range(digitSize, partQ_size):
                        modulusPartQ //= int(self.qVec[K * k + idx])

                for i in range(digitSize):
                    partQHat = modulusPartQ // int(self.qVec[K * k + i])

                    start_idx = k * K
                    end_idx = start_idx + digitSize
                    complBasis_vec = (
                            self.qVec[:start_idx] + self.qVec[end_idx:l + 1] + self.pVec
                    )

                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp[l][k][i][j] = QHatModpj

        self.pHatModp = [0] * K  # 初始化 pHatModp 列表
        self.pHatInvModp = [0] * K  # 初始化 pHatInvModp 列表
        # 计算 pHatModp
        for k in range(K):
            self.pHatModp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, K)):
                temp = int(self.pVec[j] % self.pVec[k])
                self.pHatModp[k] = (self.pHatModp[k] * temp) % int(self.pVec[k])

        # 计算 pHatInvModp # [k] qhat_k^-1 mod q_k
        for k in range(K):
            self.pHatInvModp[k] = int(self.invMod(int(self.pHatModp[k]), self.pVec[k]))

        # 初始化 pHatModq
        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.pVec[s]) % int(self.qVec[i])
                    self.pHatModq[k][i] = self.mulMod(int(self.pHatModq[k][i]), temp, int(self.qVec[i]))

        self.PModq = [0] * L  # 初始化 PModq

        # 计算 PModq
        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.pVec[k] % self.qVec[i]
                self.PModq[i] = self.mulMod(int(self.PModq[i]), int(temp), int(self.qVec[i]))

        self.PInvModq = [0] * L  # 初始化 PInvModq
        # 计算 PInvModq
        for i in range(L):
            self.PInvModq[i] = self.invMod(int(self.PModq[i]), int(self.qVec[i]))


    def negate(self, r, a):
        r = -a

    def addMod(self, r, a, b, m):
        r = (a + b) % m

    def subMod(self, r, a, b, m):
        r = b % m
        r = (a + m - r) % m

    def mulMod(self, a, b, m):
        mul = (a % m) * (b % m)
        mul %= m
        return int(mul)

    def mulModBarrett(self, r, a, b, p, pr, twok):
        mul = (a % p) * (b % p)
        self.modBarrett(r, mul, p, pr, twok)

    def modBarrett(self, r, a, m, mr, twok):
        tmp = (a * mr) >> twok
        tmp *= m
        tmp = a - tmp
        r = tmp
        if r < m:
            return
        else:
            r -= m
            return

    def invMod(self, x, m):
        temp = int(x) % int(m)
        if self.gcd(temp, m) != 1:
            raise ValueError("Inverse doesn't exist!!!")
        else:
            return self.powMod(temp, int(m - 2), int(m))

    def powMod(self, x, y, modulus):
        res = 1
        while y > 0:
            if y & 1:
                res = self.mulMod(res, x, modulus)
            y = y >> 1
            x = self.mulMod(x, x, modulus)
        return res

    def inv(self, x):
        UINT64_MAX = 0xffffffffffffffff
        return pow(int(x), UINT64_MAX, UINT64_MAX + 1)

    def pow(self, x, y):
        res = 1
        while y > 0:
            if y & 1:
                res *= x
            y = y >> 1
            x *= x
        return res

    def bitReverse(self, x):
        x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1))
        x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2))
        x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4))
        x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8))
        return ((x >> 16) | (x << 16))

    def gcd(self, a, b):
        if a == 0:
            return b
        return self.gcd(int(b) % int(a), int(a))

    def findPrimeFactors(self, s, number):
        while number % 2 == 0:
            s.add(2)
            number //= 2
        for i in range(3, int(math.sqrt(number)) + 1):
            while number % i == 0:
                s.add(i)
                number //= i
        if number > 2:
            s.add(number)

    def findPrimitiveRoot(self, modulus):
        s = set()
        phi = modulus - 1
        self.findPrimeFactors(s, phi)
        for r in range(2, phi + 1):
            flag = False
            for prime in s:
                if self.powMod(r, phi // prime, modulus) == 1:
                    flag = True
                    break
            if not flag:
                return r
        return -1

    def findMthRootOfUnity(self, M, mod):
        res = self.findPrimitiveRoot(mod)
        if (mod - 1) % M == 0:
            factor = (mod - 1) // M
            res = self.powMod(res, factor, mod)
            return res
        else:
            return -1

    # Miller-Rabin Prime Test #
    def primeTest(self, p):
        if p < 2:
            return False
        if p != 2 and p % 2 == 0:
            return False
        s = p - 1
        while s % 2 == 0:
            s //= 2
        for _ in range(200):
            temp1 = random.getrandbits(64)
            temp1 = (temp1 << 32) | random.getrandbits(32)
            temp1 = temp1 % (p - 1) + 1
            temp2 = s
            mod = self.powMod(temp1, temp2, p)
            while temp2 != p - 1 and mod != 1 and mod != p - 1:
                mod = self.mulMod(mod, mod, p)
                temp2 *= 2
            if mod != p - 1 and temp2 % 2 == 0:
                return False
        return True
    def method(self): # function to initialize variables
        pass