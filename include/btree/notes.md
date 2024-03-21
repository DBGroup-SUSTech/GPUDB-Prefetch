

datanum default 512 mtuples

A: Prefetch Distance

- prefetch distance 1,2,...,6

B: Effect of Node Fanout
- 8 开始 3(spp)/9(others) 个点, delta = 4

C: Index Performance Comparison
- QPS vs datanum
  - datanum: 2^2, 2^3, ..., 2^10 MB
- bandwidth ... method
  - datanum: 256MB, 512MB, 1G

保留 72 & 128

D: Effect of GPU Resource Allocation (Index)
- datanum = 512MB
- QPS vs Threads Per Block
  - Blocks Per Grid = 
  - Threads Per Block: 
- QPS vs Blocks Per Grid

Effect of Optimizations (Index)?

---

(A) PDIST (delete)



(B) Figure 13: Fanout

8 ~ 40 (+4)

(C) Figure 17(b): GS=72, BS = 64~512 (+64)

(D) Figure 17(a): BS=128, GS = 36 ~ 288 (+36)

(E) Figure 12(a) Performance Evaluation

(Z) Figure 14: Effect of Caching States in Shared Memory on Index Searching

TODO: draw a red line (naive)

(F) Figure 15: The Trade-off Between Divergence and MLP
LanesPerWarp (LPW):  4 ~ 32 (+4), spp 4 ~ 8



