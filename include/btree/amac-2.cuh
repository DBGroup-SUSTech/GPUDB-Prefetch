#pragma once
#include <assert.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note
/// B-Tree, fanout = 16, leaf node size = 16
/// int32_t keys, int32_t values
/// insert with OLC
/// search with

namespace btree {
namespace amac {

// TODO: compare 2 methods:
// 1. ad hoc request and prefetch root node
// 2. statically cache root node in shared memory
// 3. save states of 1 in local memory
// 4. only use few lanes in a warp
struct ConfigAMAC : public Config {
  int method = 4;  // ad hoc request and prefetch root node
};

// for prefetch  ---------------------------------------------------------
constexpr int PDIST = 5;
constexpr int THREADS_PER_BLOCK = 64;
#define VSMEM(index) v[index * blockDim.x + threadIdx.x]

enum class state_t : int8_t {
  INIT = 0,          // prefetch root nodej j
  SEARCH_FIRST = 3,  // the first search after init
  SEARCH = 1,        // search in a node, prefetch child
  DONE = 2,          // fetch next
};

/*
struct fsm_shared_t {
  int32_t key[THREADS_PER_BLOCK];
  // const Node *next[THREADS_PER_BLOCK];
  int i[THREADS_PER_BLOCK];
  state_t state[THREADS_PER_BLOCK];
};

struct fsm_t {
  int32_t key;
  // const Node *next[THREADS_PER_BLOCK];
  int i;
  state_t state;
};

*/

// for method 4 ------------------------------
constexpr int PDIST_4 = COMMON_PDIST; // 8
constexpr int THREADS_PER_BLOCK_4 = MACRO_BLOCKSIZE;  // method 4
constexpr int LANES_PER_WARP = 8;  // or 8
constexpr int LANES_PER_BLOCK = THREADS_PER_BLOCK_4 / 32 * LANES_PER_WARP;
constexpr int WARPS_PER_BLOCK = LANES_PER_BLOCK / LANES_PER_WARP;
#define VSMEM_4(index) v[index * LANES_PER_BLOCK + lid]

// struct fsm_shared_4_t {
//   int32_t key[LANES_PER_BLOCK];
//   // const Node *next[THREADS_PER_BLOCK];
//   int i[LANES_PER_BLOCK];
//   state_t state[LANES_PER_BLOCK];
};

struct fsm_shared {
  int32_t key;
  // const Node *next[THREADS_PER_BLOCK];
  int i;
  state_t state;
};
// ------------------------------------------

// __launch_bounds__(128, 1)  //
    __global__
    void gets_4(int32_t *keys, int n, int32_t *values, const NodePtr *root_p,
                DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int warpid = threadIdx.x / 32;
  int warplane = threadIdx.x % 32;
  if (warplane >= LANES_PER_WARP) return;
  int lid = warpid * LANES_PER_WARP + warplane;  // lane id

  int stride = gridDim.x * LANES_PER_BLOCK;
  int i = blockIdx.x * LANES_PER_BLOCK + lid;

  assert(blockDim.x == THREADS_PER_BLOCK_4);
  static_assert((LANES_PER_BLOCK / LANES_PER_WARP) * 32 == THREADS_PER_BLOCK_4);
  static_assert(sizeof(InnerNode) == sizeof(LeafNode));

  __shared__ InnerNode v[PDIST_4 * LANES_PER_BLOCK];  // prefetch buffer
  fsm_shared fsm[PDIST_4];

  for (int k = 0; k < PDIST_4; ++k) fsm[k].state = state_t::INIT;

  prefetch_node_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST_4) {
    k = ((k == PDIST_4) ? 0 : k);

    switch (fsm[k].state) {
      case state_t::INIT: {
        if (i < n) {
          fsm[k].state = state_t::SEARCH;
          fsm[k].key = keys[i];

          fsm[k].i = i;
          i += stride;
          pref.commit(&VSMEM_4(k), (*root_p).to_ptr<InnerNode>(node_allocator));

        } else {
          fsm[k].state = state_t::DONE;
          ++all_done;
        }
        break;
      }

      case state_t::SEARCH: {
        pref.wait();
        auto node = static_cast<Node *>(&VSMEM_4(k));

        if (node->type == Node::Type::INNER) {  // Inner Node
          auto inner = static_cast<const InnerNode *>(node);
          int pos = inner->lower_bound(fsm[k].key);

          pref.commit(&VSMEM_4(k),
                      inner->children[pos].to_ptr<InnerNode>(node_allocator));
          fsm[k].state = state_t::SEARCH;

        } else {  // Leaf Node
          auto leaf = static_cast<const LeafNode *>(node);
          auto key = fsm[k].key;
          auto &value = values[fsm[k].i];

          // TODO: output, prefetch this random write
          int pos = leaf->lower_bound(key);
          if (pos < leaf->n_key && key == leaf->keys[pos]) {
            value = leaf->values[pos];
          } else {
            value = -1;
          }

          fsm[k].state = state_t::INIT;
          --k;
        }
        break;
      }

      default: {
        break;
      }
    }
    ++k;
  }
}

/*
__launch_bounds__(32, 1)  //
    __global__
    void gets_1(int32_t *keys, int n, int32_t *values, const NodePtr *root_p,
                DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;
  assert(blockDim.x == THREADS_PER_BLOCK);

  static_assert(sizeof(InnerNode) == sizeof(LeafNode));
  __shared__ InnerNode v[PDIST * THREADS_PER_BLOCK];  // prefetch buffer
  __shared__ fsm_shared_t fsm[PDIST];

  for (int k = 0; k < PDIST; ++k) fsm[k].state[threadIdx.x] = state_t::INIT;

  prefetch_node_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    // printf("tid=%d, state= %d\n", tid, fsm[k].state[threadIdx.x]);
    switch (fsm[k].state[threadIdx.x]) {
      case state_t::INIT: {
        if (i < n) {
          fsm[k].state[threadIdx.x] = state_t::SEARCH;
          fsm[k].key[threadIdx.x] = keys[i];
          // printf("tid=%d, address=%p\n", tid, &fsm[k].i[threadIdx.x]);
          fsm[k].i[threadIdx.x] = i;
          i += stride;
          pref.commit(&VSMEM(k), (*root_p).to_ptr<InnerNode>(node_allocator));

        } else {
          fsm[k].state[threadIdx.x] = state_t::DONE;
          ++all_done;
        }
        break;
      }

      case state_t::SEARCH: {
        pref.wait();
        auto node = static_cast<Node *>(&VSMEM(k));

        if (node->type == Node::Type::INNER) {  // Inner Node
          auto inner = static_cast<const InnerNode *>(node);
          int pos = inner->lower_bound(fsm[k].key[threadIdx.x]);

          pref.commit(&VSMEM(k),
                      inner->children[pos].to_ptr<InnerNode>(node_allocator));
          fsm[k].state[threadIdx.x] = state_t::SEARCH;

        } else {  // Leaf Node
          auto leaf = static_cast<const LeafNode *>(node);
          auto key = fsm[k].key[threadIdx.x];
          auto &value = values[fsm[k].i[threadIdx.x]];

          // TODO: output, prefetch this random write
          int pos = leaf->lower_bound(key);
          if (pos < leaf->n_key && key == leaf->keys[pos]) {
            value = leaf->values[pos];
          } else {
            value = -1;
          }

          fsm[k].state[threadIdx.x] = state_t::INIT;
          --k;
        }
        break;
      }

      default: {
        break;
      }
    }
    ++k;
  }
}
*/

// __launch_bounds__(32, 1)  //
//     __global__ void gets_2(int32_t *keys, int n, int32_t *values,
//                            const Node *const *root_p) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   int i = tid;
//   assert(blockDim.x == THREADS_PER_BLOCK);

//   static_assert(sizeof(InnerNode) == sizeof(LeafNode));
//   __shared__ InnerNode v[PDIST * THREADS_PER_BLOCK];  // prefetch buffer
//   __shared__ fsm_shared_t fsm[PDIST];
//   __shared__ InnerNode root;

//   if (threadIdx.x == 0) root = *static_cast<const InnerNode *>(*root_p);
//   __syncthreads();

//   for (int k = 0; k < PDIST; ++k) fsm[k].state[threadIdx.x] = state_t::INIT;

//   prefetch_node_t pref{};
//   int all_done = 0, k = 0;

//   while (all_done < PDIST) {
//     k = ((k == PDIST) ? 0 : k);
//     // printf("tid=%d, state= %d\n", tid, fsm[k].state[threadIdx.x]);
//     switch (fsm[k].state[threadIdx.x]) {
//       case state_t::INIT: {
//         if (i < n) {
//           fsm[k].state[threadIdx.x] = state_t::SEARCH_FIRST;
//           fsm[k].key[threadIdx.x] = keys[i];
//           // printf("tid=%d, address=%p\n", tid, &fsm[k].i[threadIdx.x]);
//           fsm[k].i[threadIdx.x] = i;
//           i += stride;
//           // pref.commit(&VSMEM(k), *root_p);
//           VSMEM(k) = root;

//         } else {
//           fsm[k].state[threadIdx.x] = state_t::DONE;
//           ++all_done;
//         }
//         break;
//       }

//       case state_t::SEARCH_FIRST:
//       case state_t::SEARCH: {
//         if (fsm[k].state[threadIdx.x] == state_t::SEARCH) pref.wait();
//         auto node = static_cast<Node *>(&VSMEM(k));

//         if (node->type == Node::Type::INNER) {  // Inner Node
//           auto inner = static_cast<const InnerNode *>(node);
//           int pos = inner->lower_bound(fsm[k].key[threadIdx.x]);

//           pref.commit(&VSMEM(k), inner->children[pos]);
//           fsm[k].state[threadIdx.x] = state_t::SEARCH;

//         } else {  // Leaf Node
//           auto leaf = static_cast<const LeafNode *>(node);
//           auto key = fsm[k].key[threadIdx.x];
//           auto &value = values[fsm[k].i[threadIdx.x]];

//           // TODO: output, prefetch this random write
//           int pos = leaf->lower_bound(key);
//           if (pos < leaf->n_key && key == leaf->keys[pos]) {
//             value = leaf->values[pos];
//           } else {
//             value = -1;
//           }

//           fsm[k].state[threadIdx.x] = state_t::INIT;
//           --k;
//         }
//         break;
//       }

//       default: {
//         break;
//       }
//     }
//     ++k;
//   }
// }

// __launch_bounds__(32, 1)  //
//     __global__ void gets_3(int32_t *keys, int n, int32_t *values,
//                            const Node *const *root_p) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   int i = tid;
//   assert(blockDim.x == THREADS_PER_BLOCK);

//   static_assert(sizeof(InnerNode) == sizeof(LeafNode));
//   __shared__ InnerNode v[PDIST * THREADS_PER_BLOCK];  // prefetch buffer
//   fsm_t fsm[PDIST];

//   for (int k = 0; k < PDIST; ++k) fsm[k].state = state_t::INIT;

//   prefetch_node_t pref{};
//   int all_done = 0, k = 0;

//   while (all_done < PDIST) {
//     k = ((k == PDIST) ? 0 : k);
//     // printf("tid=%d, state= %d\n", tid, fsm[k].state[threadIdx.x]);
//     switch (fsm[k].state) {
//       case state_t::INIT: {
//         if (i < n) {
//           fsm[k].state = state_t::SEARCH;
//           fsm[k].key = keys[i];
//           // printf("tid=%d, address=%p\n", tid, &fsm[k].i[threadIdx.x]);
//           fsm[k].i = i;
//           i += stride;
//           pref.commit(&VSMEM(k), *root_p);

//         } else {
//           fsm[k].state = state_t::DONE;
//           ++all_done;
//         }
//         break;
//       }

//       case state_t::SEARCH: {
//         pref.wait();
//         auto node = static_cast<Node *>(&VSMEM(k));

//         if (node->type == Node::Type::INNER) {  // Inner Node
//           auto inner = static_cast<const InnerNode *>(node);
//           int pos = inner->lower_bound(fsm[k].key);

//           pref.commit(&VSMEM(k), inner->children[pos]);
//           fsm[k].state = state_t::SEARCH;

//         } else {  // Leaf Node
//           auto leaf = static_cast<const LeafNode *>(node);
//           auto key = fsm[k].key;
//           auto &value = values[fsm[k].i];

//           // TODO: output, prefetch this random write
//           int pos = leaf->lower_bound(key);
//           if (pos < leaf->n_key && key == leaf->keys[pos]) {
//             value = leaf->values[pos];
//           } else {
//             value = -1;
//           }

//           fsm[k].state = state_t::INIT;
//           --k;
//         }
//         break;
//       }

//       default: {
//         break;
//       }
//     }
//     ++k;
//   }
// }

void index(int32_t *keys, int32_t *values, int32_t n, ConfigAMAC cfg) {
  CHKERR(cudaDeviceReset());
  BTree tree;

  // input
  int32_t *d_keys = nullptr, *d_values = nullptr;
  CHKERR(cutil::DeviceAlloc(d_keys, n));
  CHKERR(cutil::DeviceAlloc(d_values, n));
  CHKERR(cutil::CpyHostToDevice(d_keys, keys, n));
  CHKERR(cutil::CpyHostToDevice(d_values, values, n));

  // output
  int32_t *d_outs = nullptr;
  CHKERR(cutil::DeviceAlloc(d_outs, n));
  CHKERR(cutil::DeviceSet(d_outs, 0, n));

  // run
  cudaEvent_t start_build, end_build, start_probe, end_probe;
  CHKERR(cudaEventCreate(&start_build));
  CHKERR(cudaEventCreate(&end_build));
  CHKERR(cudaEventCreate(&start_probe));
  CHKERR(cudaEventCreate(&end_probe));

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  CHKERR(cudaStreamCreate(&stream));
  CHKERR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  fmt::print(
      "Build: {} blocks * {} threads"
      "Probe: {} blocks * {} threads\n",
      cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
      cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    puts_olc<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_keys, d_values, n, tree.d_root_p_, tree.allocator_);
    CHKERR(
        cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
  }

  {
    CHKERR(
        cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
    if (cfg.method == 1) {
      // const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(InnerNode);
      // fmt::print("smem_size = {}\n", smeme_size);
      // gets_1<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
          // d_keys, n, d_outs, tree.d_root_p_, tree.allocator_);
      assert(0);
    } else if (cfg.method == 2) {
      // gets_2<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
      //     d_keys, n, d_outs, tree.d_root_p_);
      assert(0);
    } else if (cfg.method == 3) {
      // gets_3<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
      //     d_keys, n, d_outs, tree.d_root_p_);
      assert(0);
    } else if (cfg.method == 4) {
      // printf("config 4\n");
      gets_4<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
          d_keys, n, d_outs, tree.d_root_p_, tree.allocator_);
    } else {
      assert(0);
    }
  }
  CHKERR(cudaEventRecordWithFlags(end_probe, stream, cudaEventRecordExternal));

  CHKERR(cudaStreamEndCapture(stream, &graph));
  CHKERR(cudaGraphInstantiate(&instance, graph));
  CHKERR(cudaGraphLaunch(instance, stream));

  CHKERR(cudaStreamSynchronize(stream));
  float ms_build, ms_probe;
  CHKERR(cudaEventElapsedTime(&ms_build, start_build, end_build));
  CHKERR(cudaEventElapsedTime(&ms_probe, start_probe, end_probe));

  fmt::print(
      "BTreeOLC Naive (bucket size = 1)\n"
      "[build(R), {} ms, {} tps (S)]\n"
      "[probe(S), {} ms, {} tps (R)]\n",
      ms_build, n * 1.0 / ms_build * 1000, ms_probe, n * 1.0 / ms_probe * 1000);

  // check output
  int32_t *outs = new int32_t[n];
  CHKERR(cutil::CpyDeviceToHost(outs, d_outs, n));
  for (int i = 0; i < n; ++i) {
    // fmt::print("{}-{}\n", outs[i], values[i]);
    assert(outs[i] == values[i]);
  }
  delete[] outs;
}
}  // namespace amac
}  // namespace btree