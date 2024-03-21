#pragma once
#include <assert.h>
#include <cuda_pipeline_primitives.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note

namespace btree {
namespace spp {

#define VSMEM2(v, index) v[(index) * LANES_PER_BLOCK + lid]


// fanout = 20, code path = 8, alloc_cap = 1<<34
// fanout = 18, code path = 9 alloc_cap = 1<<34
// fanout = 16, code path = 9, alloc_cap = 1<<34
// fanout = 14, code path = 10, alloc_cap = 1<<34
// fanout = 12, code path = 12, alloc_cap = 1<<34
// fanout = 10, code path = 12, alloc_cap = 1<<34
// fanout = 8, code path = 14, alloc_cap = 1<<34

#define SPP(x) (x >= 16 ? (x != 20 ? 9 : 8) : (x >= 14 ? 10 : (x >= 10 ? 12 : 14)))

constexpr int STAGE = SPP(MACRO_MAX_ENTRIES);  // Max code path length, assume <= 10
constexpr int LANES_PER_BLOCK = 32; // 32, fanout >= 16 => 16
constexpr int LANES_PER_WARP = 8;

__shared__ InnerNode v[STAGE * LANES_PER_BLOCK];  // prefetch buffer
__shared__ bool f[STAGE * LANES_PER_BLOCK];  // probe completed (true) or not (false)
__shared__ int32_t key[STAGE * LANES_PER_BLOCK];

__global__ void gets_parallel(int32_t *keys, int n, int32_t *values,
                              const NodePtr *root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {

  int warpId = threadIdx.x / 32;
  int warpLane = threadIdx.x % 32;
  if (warpLane >= LANES_PER_WARP)
    return ;
  
  static_assert(sizeof(InnerNode) == sizeof(LeafNode));
  static_assert(LANES_PER_BLOCK % LANES_PER_WARP == 0);

  int lid = warpId * LANES_PER_WARP + warpLane;  // lane id in the block
  int tid = blockIdx.x * LANES_PER_BLOCK + lid; // the thread id
  if (tid >= n)
    return ;

  int stride = gridDim.x * LANES_PER_BLOCK;
  int G = (n - 1 - tid) / stride + 1;

  prefetch_node_t pref{};

  pref.commit(&VSMEM2(v, 0), (*root_p).to_ptr<InnerNode>(node_allocator));
  pref.wait();

  InnerNode root = VSMEM2(v, 0);
  // iteration 1: code 0 for i = 0
  // iteration 2: code 0 for i = 1, code 1 for i = 0
  // ...
  int finished = 0;
  for (int i = 0; ; i ++) {
    if (finished == G) break;
    for (int j = i >= G ? G - 1 : i; j >= 0; j--) {
      int jmod = j % STAGE;
      if (i < G && j == i) {
        VSMEM2(v, jmod) = root;
        VSMEM2(key, jmod) = keys[tid + j * stride];
        VSMEM2(f, jmod) = false;
      } else {
        if (VSMEM2(f, jmod)) break;
        pref.wait();
      }
      auto node = static_cast<const Node *>(&VSMEM2(v, jmod));
      if (node->type == Node::Type::INNER) {
        auto inner = static_cast<const InnerNode *>(node);
        int pos = inner->lower_bound(VSMEM2(key, jmod));
        pref.commit(&VSMEM2(v, jmod),
                    inner->children[pos].to_ptr<InnerNode>(node_allocator));
      } else {
        auto leaf = static_cast<const LeafNode *>(node);
        auto &value = values[tid + stride * j];
        int pos = leaf->lower_bound(VSMEM2(key, jmod));
        if (pos < leaf->n_key && VSMEM2(key, jmod) == leaf->keys[pos]) {
          value = leaf->values[pos];
        } else {
          value = -1;
        }
        finished++;
        VSMEM2(f, jmod) = true;
      }
    }
  }
}

#undef VSMEM2

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
    gets_parallel<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_keys, n, d_outs, tree.d_root_p_, tree.allocator_);
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
  int32_t *outs = new int32_t[n];
  CHKERR(cutil::CpyDeviceToHost(outs, d_outs, n));
  for (int i = 0; i < n; ++i) {
    assert(outs[i] == values[i]);
  }
  delete[] outs;
}
}  // namespace spp
}  // namespace btree