# Revisiting Prefetch of Query Processing on GPU

## Progress

| | Naive | GP \[1\]| SPP \[1\]| AMAC \[2\]| IMV \[3\]|
|:---:|:---:|:---:| :---:| :---:| :---:|
| Hash Join Probe | ✅ | 🔨 | 🔨 | ✅ | ✅ |
| Hash Join Build | ✅ |  |  |  |  |
| BTree Lookup    | ✅ | 🔨 | 🔨 | ✅ | 🔨 |
| BTree Insert    | ✅ | |  |  |  |

## Workload Description

### Hash Join

* Adapted from Timo Kersten's \[4\]\[5\] implementation on CPU.
* Non-partitioned and non-unique hash join with early materialization.
* Staticaly pre-allocated hash table entries.
* Uniform distribution (`--gtest_filter="unique.*"`) and skew distribution (`--gtest_filter="skew.*"`)
* Row-format input tables.

### BTree

* Adapted from Vicktor Leis's \[6\]\[7\] implementation on CPU.
* Concurrent insertion with Optimistic Lock Coupling.
* Allocate nodes with a centralized memory pool.

## Reference

\[1\] S. Chen, A. Ailamaki, P. B. Gibbons and T. C. Mowry. 2004. Improving hash join performance through prefetching. ICDE. \
\[2\] Onur Kocberber, Babak Falsafi, and Boris Grot. 2015. Asynchronous memory access chaining. VLDB. \
\[3\] Zhuhe Fang, Beilei Zheng, and Chuliang Weng. 2019. Interleaved multi-vectorizing. VLDB. \
\[4\] Timo Kersten, Viktor Leis, Alfons Kemper, Thomas Neumann, Andrew Pavlo, and Peter Boncz. 2018. Everything you always wanted to know about compiled and vectorized queries but were afraid to ask. VLDB. \
\[5\] <https://github.com/TimoKersten/db-engine-paradigms> \
\[6\] Leis, Viktor, Michael Haubenschild and Thomas Neumann. 2019. Optimistic Lock Coupling: A Scalable and Efficient General-Purpose Synchronization Method. IEEE Data Eng. Bull. \
\[7\] <https://github.com/wangziqi2016/index-microbench/tree/master/BTreeOLC> \
