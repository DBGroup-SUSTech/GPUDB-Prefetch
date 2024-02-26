#pragma once
#include <assert.h>
#include <cuda_pipeline_primitives.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note

namespace btree {
namespace gp {

constexpr int PDIST = 8;
constexpr int LANES_PER_BLOCK = 32;
constexpr int LANES_PER_WARP = 8;

// constexpr int WARPS_PER_BLOCK = LANES_PER_BLOCK / LANES_PER_WARP;


#define VSMEM(v, index) v[(index) * LANES_PER_BLOCK + lid]

__shared__ InnerNode v[PDIST * LANES_PER_BLOCK];  // prefetch buffer
__shared__ int32_t key[PDIST * LANES_PER_BLOCK];

__global__ void gets_parallel(int32_t *keys, int n, int32_t *values,
                              const NodePtr *root_p, DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  
  int warpId = threadIdx.x / 32;
  int warpLane = threadIdx.x % 32;
  if (warpLane >= LANES_PER_WARP)
    return ;
  
  int lid = warpId * LANES_PER_WARP + warpLane;  // lane id in the block
  int tid = blockIdx.x * LANES_PER_BLOCK + lid; // the thread id
  if (tid >= n)
    return ;

  int stride = gridDim.x * LANES_PER_BLOCK;

  static_assert(sizeof(InnerNode) == sizeof(LeafNode));
  static_assert(LANES_PER_BLOCK % LANES_PER_WARP == 0);

  prefetch_node_t pref{};
  for (int st = tid; st < n; st += stride * PDIST) { // group start position
    pref.commit(&VSMEM(v, 0), (*root_p).to_ptr<InnerNode>(node_allocator));
    pref.wait();
    
    int G = min( (n - 1 - st) / stride + 1, PDIST );

    for (int k = 0; k < G; k ++) {
      if(k)
        VSMEM(v, k) = VSMEM(v, 0);
      VSMEM(key, k) = keys[st + k * stride];
    }

    for (bool notFirst = false; ; notFirst = true) {
      bool endFlag = false;
      for (int k = 0, i = st; k < G; k ++, i += stride) {
        if (notFirst)
          pref.wait();

        auto node = static_cast<const Node *> (&VSMEM(v, k));
        if (node->type == Node::Type::INNER) {
          auto inner = static_cast<const InnerNode *>(node);
          int pos = inner->lower_bound(VSMEM(key, k));
          pref.commit(&VSMEM(v, k), inner->children[pos].to_ptr<InnerNode>(node_allocator));
        } else {
          endFlag = true;
          auto leaf = static_cast<const LeafNode *>(node);
          auto &value = values[i];

          int pos = leaf->lower_bound(VSMEM(key, k));
          if (pos < leaf->n_key && VSMEM(key, k) == leaf->keys[pos]) {
            value = leaf->values[pos];
          } else {
            value = -1;
          }
        } 
      }
      if (endFlag)
        break;
    }
  }
}

void index(int32_t *keys, int32_t *values, int32_t n, Config cfg) {
  CHKERR(cudaDeviceReset());
  BTree tree;

  printf("size: %d\n", sizeof(int));

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
}  // namespace naive
}  // namespace btree