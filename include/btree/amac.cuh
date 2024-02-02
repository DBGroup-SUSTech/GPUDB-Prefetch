#pragma once
#include <assert.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note
/// B-Tree, fanout = 16, leaf node size = 16
/// int64_t keys, int64_t values
/// insert with OLC
/// search with

namespace btree {
namespace amac {

// TODO: compare 2 methods:
// 1. ad hoc request and prefetch root node
// 2. statically cache root node in shared memory
struct ConfigAMAC : public Config {
  int method = 1;  // ad hoc request and prefetch root node
};

// for prefetch  ---------------------------------------------------------
constexpr int PDIST = 5;
constexpr int THREADS_PER_BLOCK = 64;
#define VSMEM(index) v[index * blockDim.x + threadIdx.x]

enum class state_t : int8_t {
  INIT = 0,    // prefetch root node
  SEARCH = 1,  // search in a node, prefetch child
  DONE = 2,    // fetch next
};

struct fsm_shared_t {
  int64_t key[THREADS_PER_BLOCK];
  // const Node *next[THREADS_PER_BLOCK];
  int i[THREADS_PER_BLOCK];
  state_t state[THREADS_PER_BLOCK];
};

__launch_bounds__(64, 1)  //
    __global__ void gets_1(int64_t *keys, int n, int64_t *values,
                           const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;
  assert(blockDim.x == THREADS_PER_BLOCK);

  static_assert(sizeof(InnerNode) == sizeof(LeafNode));
  extern __shared__ InnerNode v[];  // prefetch buffer

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
          pref.commit(&VSMEM(k), *root_p);

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

          pref.commit(&VSMEM(k), inner->children[pos]);
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

void index(int64_t *keys, int64_t *values, int32_t n, ConfigAMAC cfg) {
  CHKERR(cudaDeviceReset());
  BTree tree;

  // input
  int64_t *d_keys = nullptr, *d_values = nullptr;
  CHKERR(cutil::DeviceAlloc(d_keys, n));
  CHKERR(cutil::DeviceAlloc(d_values, n));
  CHKERR(cutil::CpyHostToDevice(d_keys, keys, n));
  CHKERR(cutil::CpyHostToDevice(d_values, values, n));

  // output
  int64_t *d_outs = nullptr;
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
      const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(InnerNode);
      fmt::print("smem_size = {}\n", smeme_size);
      gets_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size, stream>>>(
          d_keys, n, d_outs, tree.d_root_p_);
    } else {
      assert(0);
    }
    CHKERR(
        cudaEventRecordWithFlags(end_probe, stream, cudaEventRecordExternal));
  }

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
  int64_t *outs = new int64_t[n];
  CHKERR(cutil::CpyDeviceToHost(outs, d_outs, n));
  for (int i = 0; i < n; ++i) {
    assert(outs[i] == values[i]);
  }
  delete[] outs;
}
}  // namespace amac
}  // namespace btree