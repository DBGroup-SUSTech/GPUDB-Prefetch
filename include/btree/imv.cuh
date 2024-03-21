#pragma once
#include <assert.h>
#include <fmt/format.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

namespace btree {
namespace imv {
constexpr int PDIST_4 = COMMON_PDIST;
constexpr int THREADS_PER_BLOCK_4 = MACRO_BLOCKSIZE;  // method 4
constexpr int LANES_PER_WARP = COMMON_LPW;  // or 8
constexpr int LANES_PER_BLOCK = THREADS_PER_BLOCK_4 / 32 * LANES_PER_WARP;
constexpr int WARPS_PER_BLOCK = LANES_PER_BLOCK / LANES_PER_WARP;
constexpr unsigned MASK_ALL_LANES = 0xFFFFFFFF;
#define VSMEM_4(index) v[index * LANES_PER_BLOCK + lid]

enum class state_t : int8_t {
  INIT = 0,    // prefetch root node
  SEARCH = 1,  // search in a node, prefetch child
  DONE = 2,    // fetch next
};

struct fsm_shared_4_t {
  int32_t key[LANES_PER_BLOCK];
  // const Node *next[THREADS_PER_BLOCK];
  int i[LANES_PER_BLOCK];
  state_t state[LANES_PER_BLOCK];
  bool active[LANES_PER_BLOCK];
};

// __launch_bounds__(128, 1)  //
    __global__ void gets_imv(int32_t *keys, int n, int32_t *values,
                             const NodePtr *const root_p,
                             DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int warpid = threadIdx.x / 32;
  int warplane = threadIdx.x % 32;
  bool enable = (warplane < LANES_PER_WARP);
  int ENABLE_MASK = __ballot_sync(MASK_ALL_LANES, enable);
  if (warplane >= LANES_PER_WARP) return;

  // printf("tid=%d enabled=%d\n", threadIdx.x, enable);
  int lid = warpid * LANES_PER_WARP + warplane;  // lane id

  int stride = gridDim.x * LANES_PER_BLOCK;
  int i = blockIdx.x * LANES_PER_BLOCK + lid;

  assert(blockDim.x == THREADS_PER_BLOCK_4);
  static_assert((LANES_PER_BLOCK / LANES_PER_WARP) * 32 == THREADS_PER_BLOCK_4);
  static_assert(sizeof(InnerNode) == sizeof(LeafNode));

  __shared__ InnerNode v[PDIST_4 * LANES_PER_BLOCK];  // prefetch buffer
  __shared__ fsm_shared_4_t fsm[PDIST_4];

  if (enable) {
    for (int k = 0; k < PDIST_4; ++k) {
      fsm[k].state[lid] = state_t::INIT;
      fsm[k].active[lid] = false;
    }
  }

  prefetch_node_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST_4) {
    k = ((k == PDIST_4) ? 0 : k);

    switch (fsm[k].state[lid]) {
      case state_t::INIT: {
        bool active = (i < n);
        // if (threadIdx.x == 16) printf("i=%d, active=%d\n", i, active);
        int active_mask = __ballot_sync(ENABLE_MASK, active);

        if (active_mask) {
          if (active) {
            fsm[k].key[lid] = keys[i];

            fsm[k].i[lid] = i;
            i += stride;
            pref.commit(&VSMEM_4(k),
                        (*root_p).to_ptr<InnerNode>(node_allocator));
          }

          // __syncwarp();

          fsm[k].state[lid] = state_t::SEARCH;
          fsm[k].active[lid] = active;
        } else {
          fsm[k].state[lid] = state_t::DONE;
          ++all_done;
        }
        break;
      }

      case state_t::SEARCH: {
        bool active = fsm[k].active[lid];
        bool inner;
        Node *node = nullptr;

        if (active) {
          pref.wait();
          node = static_cast<Node *>(&VSMEM_4(k));
          inner = (node->type == Node::Type::INNER);
        }

        int inner_mask = __ballot_sync(ENABLE_MASK, inner);
        if (inner_mask) {
          if (active) {
            // assert goes to same branch
            // assert(node->type == Node::Type::INNER);
            auto inner = static_cast<const InnerNode *>(node);
            int pos = inner->lower_bound(fsm[k].key[lid]);
            pref.commit(&VSMEM_4(k),
                        inner->children[pos].to_ptr<InnerNode>(node_allocator));
          }

          // __syncwarp();

          fsm[k].state[lid] = state_t::SEARCH;
        } else {
          if (active) {
            // assert(node->type == Node::Type::LEAF);
            auto leaf = static_cast<const LeafNode *>(node);
            auto key = fsm[k].key[lid];
            auto &value = values[fsm[k].i[lid]];

            int pos = leaf->lower_bound(key);
            if (pos < leaf->n_key && key == leaf->keys[pos]) {
              value = leaf->values[pos];
            } else {
              value = -1;
            }
          }

          // __syncwarp();
          fsm[k].state[lid] = state_t::INIT;
          fsm[k].active[lid] = false;
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

void index(int32_t *keys, int32_t *values, int32_t n, Config cfg) {
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

    gets_imv<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_keys, n, d_outs, tree.d_root_p_, tree.allocator_);
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

}  // namespace imv
}  // namespace btree


//------------------------------------------------------

/*

#pragma once
#include <assert.h>
#include <fmt/format.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

namespace btree {
namespace imv {
constexpr int PDIST_4 = COMMON_PDIST;
constexpr int THREADS_PER_BLOCK_4 = MACRO_BLOCKSIZE;  // method 4
constexpr int LANES_PER_WARP = 8;  // or 8
constexpr int LANES_PER_BLOCK = THREADS_PER_BLOCK_4 / 32 * LANES_PER_WARP;
constexpr int WARPS_PER_BLOCK = LANES_PER_BLOCK / LANES_PER_WARP;
constexpr unsigned MASK_ALL_LANES = 0xFFFFFFFF;
#define VSMEM_4(index) v[index * LANES_PER_BLOCK + lid]

enum class state_t : int8_t {
  INIT = 0,    // prefetch root node
  SEARCH = 1,  // search in a node, prefetch child
  DONE = 2,    // fetch next
};

// struct fsm_shared_4_t {
//   int32_t key[LANES_PER_BLOCK];
//   // const Node *next[THREADS_PER_BLOCK];
//   int i[LANES_PER_BLOCK];
//   state_t state[LANES_PER_BLOCK];
//   bool active[LANES_PER_BLOCK];
// };

struct fsm_shared {
  int32_t key;
  // const Node *next[THREADS_PER_BLOCK];
  int i;
  state_t state;
  bool active;
};

// __launch_bounds__(128, 1)  //
    __global__ void gets_imv(int32_t *keys, int n, int32_t *values,
                             const NodePtr *const root_p,
                             DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int warpid = threadIdx.x / 32;
  int warplane = threadIdx.x % 32;
  bool enable = (warplane < LANES_PER_WARP);
  int ENABLE_MASK = __ballot_sync(MASK_ALL_LANES, enable);
  if (warplane >= LANES_PER_WARP) return;

  // printf("tid=%d enabled=%d\n", threadIdx.x, enable);
  int lid = warpid * LANES_PER_WARP + warplane;  // lane id

  int stride = gridDim.x * LANES_PER_BLOCK;
  int i = blockIdx.x * LANES_PER_BLOCK + lid;

  assert(blockDim.x == THREADS_PER_BLOCK_4);
  static_assert((LANES_PER_BLOCK / LANES_PER_WARP) * 32 == THREADS_PER_BLOCK_4);
  static_assert(sizeof(InnerNode) == sizeof(LeafNode));

  __shared__ InnerNode v[PDIST_4 * LANES_PER_BLOCK];  // prefetch buffer
  fsm_shared fsm[PDIST_4];
  
  if (enable) {
    for (int k = 0; k < PDIST_4; ++k) {
      fsm[k].state = state_t::INIT;
      fsm[k].active = false;
    }
  }

  prefetch_node_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST_4) {
    k = ((k == PDIST_4) ? 0 : k);

    switch (fsm[k].state) {
      case state_t::INIT: {
        bool active = (i < n);
        // if (threadIdx.x == 16) printf("i=%d, active=%d\n", i, active);
        int active_mask = __ballot_sync(ENABLE_MASK, active);

        if (active_mask) {
          if (active) {
            fsm[k].key = keys[i];

            fsm[k].i = i;
            i += stride;
            pref.commit(&VSMEM_4(k),
                        (*root_p).to_ptr<InnerNode>(node_allocator));
          }

          // __syncwarp();

          fsm[k].state = state_t::SEARCH;
          fsm[k].active = active;
        } else {
          fsm[k].state = state_t::DONE;
          ++all_done;
        }
        break;
      }

      case state_t::SEARCH: {
        bool active = fsm[k].active;
        bool inner;
        Node *node = nullptr;

        if (active) {
          pref.wait();
          node = static_cast<Node *>(&VSMEM_4(k));
          inner = (node->type == Node::Type::INNER);
        }

        int inner_mask = __ballot_sync(ENABLE_MASK, inner);
        if (inner_mask) {
          if (active) {
            // assert goes to same branch
            // assert(node->type == Node::Type::INNER);
            auto inner = static_cast<const InnerNode *>(node);
            int pos = inner->lower_bound(fsm[k].key);
            pref.commit(&VSMEM_4(k),
                        inner->children[pos].to_ptr<InnerNode>(node_allocator));
          }

          // __syncwarp();

          fsm[k].state = state_t::SEARCH;
        } else {
          if (active) {
            // assert(node->type == Node::Type::LEAF);
            auto leaf = static_cast<const LeafNode *>(node);
            auto key = fsm[k].key;
            auto &value = values[fsm[k].i];

            int pos = leaf->lower_bound(key);
            if (pos < leaf->n_key && key == leaf->keys[pos]) {
              value = leaf->values[pos];
            } else {
              value = -1;
            }
          }

          // __syncwarp();
          fsm[k].state = state_t::INIT;
          fsm[k].active = false;
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

void index(int32_t *keys, int32_t *values, int32_t n, Config cfg) {
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

    gets_imv<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_keys, n, d_outs, tree.d_root_p_, tree.allocator_);
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

}  // namespace imv
}  // namespace btree

*/