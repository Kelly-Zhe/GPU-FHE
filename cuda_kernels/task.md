一：

目标：将arithmetic.py 中的ModUP_Core  和 ModDown_Core代码的cuda算子实现，就是保证用python中目前的接口能调用两个core函数的cuda版本，并且通过目前python中已有的一组测试。

目前主要关注GPU-FHE中的yhh-dev分支，该分支下，test.py 有一些测试代码，可以作为示例代码辅助理解。

目前首先需要将arithmetic.py 中的ModUP_Core  和 ModDown_Core代码的cuda算子实现。（这两个core函数是论文中的basisconv的在具体场景中的特殊形式，这两个函数被KeySwitch.py中的ModUp和ModDown调用，一般论文中我们谈论ModUp和ModDown。）

实现这两个函数的cuda代码可以参考 ：

1. Over100x(仓库地址见论文) 
2. https://github.com/encryptorion-lab/phantom-fhe （论文见仓库readme）
3. 3. 其它您能找到的任何fhe+gpu的代码....都可以的

---

二：

同上的，把over100x的ntt和intt算子也清洗出来并通过目前yhh-dev中的测试。

ntt和intt的测试代码及相关数据在test_arithmetic.py中。