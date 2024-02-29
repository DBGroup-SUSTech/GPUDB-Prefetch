// #pragma once
// #include <assert.h>
// #include <cuda_pipeline_primitives.h>

// #include "btree/common.cuh"
// #include "btree/insert.cuh"

// /// @note

// namespace btree {
// namespace gp_sep {

// struct prefetch_sep_t {
//   int8_t pending = 0;
//   __device__ __forceinline__ void commit(void *__restrict__ dst_shared,
//                                          const void *__restrict__ src_global,
//                                          int bytes) {
//     auto dst = reinterpret_cast<char *>(dst_shared);
//     auto src = reinterpret_cast<const char *>(src_global);

//     for (int i = 0; i < (bytes >> 3); i ++) {
//       __pipeline_memcpy_async(dst, src, 8);
//       dst += 8, src += 8;
//     }
//     bytes &= 7;
//     if (bytes > 0) {
//       assert(bytes == 4);
//       __pipeline_memcpy_async(dst, src, 4);
//     }
//     __pipeline_commit();
//     assert(pending < INT8_MAX);
//     ++pending;
//   }

//   __device__ __forceinline__ void wait() {
//     // assert(pending);
//     __pipeline_wait_prior(pending - 1);
//     --pending;
//   }
// };

// constexpr int ELS_PER_THREAD = 8;
// constexpr int THREADS_PER_BLOCK = 8;
// // (gp-sep) blocksize 64, N = 1e7: gp 39ms gp-sep 70ms
// // (gp-sep) blocksize 32, N = 1e7: gp 39ms gp-sep 76ms
// // (gp-sep) blocksize 16, N = 1e7: gp 39ms gp-sep 71ms
// // (gp-sep) blocksize 8, N = 1e7: gp 39ms gp-sep 65ms
// // (gp-sep) blocksize 4, N = 1e7: gp 39ms gp-sep 81ms


// #define VSMEM3(v, index) v[index * blockDim.x + threadIdx.x]

// enum class state_t : int8_t {
//   WAIT_TYPE = 0,     // type prefetch start
//   WAIT_KEYS = 1,     // keys[] prefetch start, wait for keys[]
//   WAIT_CH = 2,       // ch[?] prefetch start, wait for ch[?]
//   FINISH = 3         // done
// };
// // KEY -> Prefetch Key, lower_bound -> CH
// __shared__ state_t st[ELS_PER_THREAD * THREADS_PER_BLOCK];
// __shared__ const Node* ch[ELS_PER_THREAD * THREADS_PER_BLOCK];
// // prefetch buffer for exactly one child

// struct node_pack {
//   Node::Type type;
//   int n_key;
//   int64_t keys[InnerNode::MAX_ENTRIES];
//   __device__ __forceinline__ int lower_bound(int64_t key) const {
//     int lower = 0;
//     int upper = n_key;
//     while (lower < upper) {
//       int mid = (upper - lower) / 2 + lower;
//       int cmp = util::scalar_cmp(key, keys[mid]);
//       if (cmp < 0) {
//         upper = mid;
//       } else if (cmp > 0) {
//         lower = mid + 1;
//       } else {
//         return mid;
//       }
//     }
//     return lower;
//   }
// };

// __shared__ node_pack v[ELS_PER_THREAD * THREADS_PER_BLOCK];
// // prefetch buffer for keys

// __device__ void prefetch_type(prefetch_sep_t &pref,
//                               node_pack *pack,
//                               const Node *node) {
//   pref.commit(&pack->type, &node->type, sizeof(Node::type));
// }

// __device__ void prefetch_keys(prefetch_sep_t &pref,
//                               node_pack *pack,
//                               const Node *node) {
//   if (pack->type == Node::Type::INNER) {
//     auto nd = static_cast<const InnerNode *>(node);
//     pref.commit(&pack->n_key, &nd->n_key, sizeof(pack->n_key));
//     pref.commit(pack->keys, nd->keys, sizeof(pack->keys));
//   } else {
//     auto nd = static_cast<const LeafNode *>(node);
//     pref.commit(&pack->n_key, &nd->n_key, sizeof(pack->n_key));
//     pref.commit(pack->keys, nd->keys, sizeof(pack->keys));
//   }
  
// }
// __device__ void prefetch_ch(prefetch_sep_t &pref,
//                             const Node* *ch,
//                             const Node *node,
//                             int pos) {
//   auto nd = static_cast<const InnerNode *>(node);
//   pref.commit(ch, &nd->children[pos], sizeof(Node *));
// }
// __device__ void prefetch_wait(prefetch_sep_t &pref, state_t st) {
//   if (st == state_t::WAIT_TYPE) {
//     pref.wait();
//   } else if (st == state_t::WAIT_KEYS) {
//     pref.wait();
//     pref.wait();
//   } else if (st == state_t::WAIT_CH) {
//     pref.wait();
//   } else {
//     assert(false);
//   }
// }

// __global__ void gets_parallel(int64_t *keys, int n, int64_t *values,
//                               const Node *const *root_p) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= n)
//     return ;
//   int stride = blockDim.x * gridDim.x;
//   int G = (n - 1 - tid) / stride + 1;
//   assert(G <= ELS_PER_THREAD);
//   assert(blockDim.x == THREADS_PER_BLOCK);
//   static_assert(sizeof(InnerNode) == sizeof(LeafNode));


//   prefetch_sep_t pref{};
//   int64_t key[ELS_PER_THREAD];
//   for (int k = 0; k < G; k ++) {
//     key[k] = keys[tid + k * stride];
//     VSMEM3(st, k) = state_t::WAIT_TYPE;
//     VSMEM3(ch, k) = *root_p;
//     prefetch_type(pref, &VSMEM3(v, k), static_cast<const Node*>(*root_p));
//   }

//   int finished = 0;
//   while (finished < G) {
//     for (int k = 0, i = tid; k < G; k ++, i += stride) {
//       if (VSMEM3(st, k) == state_t::WAIT_TYPE) {
//         prefetch_wait(pref, state_t::WAIT_TYPE);
//         prefetch_keys(pref, &VSMEM3(v, k), VSMEM3(ch, k));
//         VSMEM3(st, k) = state_t::WAIT_KEYS;
//       } else if (VSMEM3(st, k) == state_t::WAIT_KEYS) {
//         prefetch_wait(pref, state_t::WAIT_KEYS);
//         auto pack = VSMEM3(v, k);
//         int pos = pack.lower_bound(key[k]);
//         if (pack.type == Node::Type::INNER) {
//           prefetch_ch(pref, &VSMEM3(ch, k), VSMEM3(ch, k), pos);
//           VSMEM3(st, k) = state_t::WAIT_CH;
//         } else {
//           auto &value = values[i];
//           if (pos < pack.n_key && key[k] == pack.keys[pos]) {
//             value = static_cast<const LeafNode *>(VSMEM3(ch, k))->values[pos];
//           } else {
//             value = -1;
//           }
//           finished ++;
//           VSMEM3(st, k) = state_t::FINISH;
//         }
//       } else if (VSMEM3(st, k) == state_t::WAIT_CH) {
//         prefetch_wait(pref, state_t::WAIT_CH);
//         prefetch_type(pref, &VSMEM3(v, k), VSMEM3(ch, k));
//         VSMEM3(st, k) = state_t::WAIT_TYPE;
//       } // else: Finish
//     }
//   }
// }

// void index(int64_t *keys, int64_t *values, int32_t n, Config cfg) {
//   CHKERR(cudaDeviceReset());
//   BTree tree;

//   // input
//   int64_t *d_keys = nullptr, *d_values = nullptr;
//   CHKERR(cutil::DeviceAlloc(d_keys, n));
//   CHKERR(cutil::DeviceAlloc(d_values, n));
//   CHKERR(cutil::CpyHostToDevice(d_keys, keys, n));
//   CHKERR(cutil::CpyHostToDevice(d_values, values, n));

//   // output
//   int64_t *d_outs = nullptr;
//   CHKERR(cutil::DeviceAlloc(d_outs, n));
//   CHKERR(cutil::DeviceSet(d_outs, 0, n));

//   // run
//   cudaEvent_t start_build, end_build, start_probe, end_probe;
//   CHKERR(cudaEventCreate(&start_build));
//   CHKERR(cudaEventCreate(&end_build));
//   CHKERR(cudaEventCreate(&start_probe));
//   CHKERR(cudaEventCreate(&end_probe));

//   cudaStream_t stream;
//   cudaGraph_t graph;
//   cudaGraphExec_t instance;

//   CHKERR(cudaStreamCreate(&stream));
//   CHKERR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
//   fmt::print(
//       "Build: {} blocks * {} threads"
//       "Probe: {} blocks * {} threads\n",
//       cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
//       cfg.probe_blocksize);

//   {
//     CHKERR(
//         cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
//     puts_olc<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
//         d_keys, d_values, n, tree.d_root_p_, tree.allocator_);
//     CHKERR(
//         cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
//   }

//   {
//     CHKERR(
//         cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
//     gets_parallel<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
//         d_keys, n, d_outs, tree.d_root_p_);
//     CHKERR(
//         cudaEventRecordWithFlags(end_probe, stream, cudaEventRecordExternal));
//   }

//   CHKERR(cudaStreamEndCapture(stream, &graph));
//   CHKERR(cudaGraphInstantiate(&instance, graph));
//   CHKERR(cudaGraphLaunch(instance, stream));
//   CHKERR(cudaStreamSynchronize(stream));

//   float ms_build, ms_probe;
//   CHKERR(cudaEventElapsedTime(&ms_build, start_build, end_build));
//   CHKERR(cudaEventElapsedTime(&ms_probe, start_probe, end_probe));

//   fmt::print(
//       "BTreeOLC Naive (bucket size = 1)\n"
//       "[build(R), {} ms, {} tps (S)]\n"
//       "[probe(S), {} ms, {} tps (R)]\n",
//       ms_build, n * 1.0 / ms_build * 1000, ms_probe, n * 1.0 / ms_probe * 1000);

//   // check output
//   int64_t *outs = new int64_t[n];
//   CHKERR(cutil::CpyDeviceToHost(outs, d_outs, n));
//   for (int i = 0; i < n; ++i) {
//     if (outs[i] != values[i]) {
//       printf("%lld vs %lld\n", outs[i], values[i]);
//     }
//     assert(outs[i] == values[i]);
//   }
//   delete[] outs;
// }
// }  // namespace naive
// }  // namespace btree