#pragma once
#include <assert.h>
#include <cuda_pipeline_primitives.h>

#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note

namespace btree {
namespace spp {

constexpr int STAGE = 32; // Max stages
constexpr int ELS_PER_THREAD = 8;
constexpr int THREADS_PER_BLOCK = 8;
#define VSMEM2(v, index) v[(index) * blockDim.x + threadIdx.x]
#define POS(i) (tid + (i) * stride)

__shared__ InnerNode v[ELS_PER_THREAD * THREADS_PER_BLOCK];  // prefetch buffer
__shared__ bool f[ELS_PER_THREAD * THREADS_PER_BLOCK]; // probe completed (true) or not (false)
__shared__ int64_t key[ELS_PER_THREAD * THREADS_PER_BLOCK];

__global__ void gets_parallel(int64_t *keys, int n, int64_t *values,
                              const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= n)
    return ;
  int stride = blockDim.x * gridDim.x;
  int G = (n-1-tid) / stride + 1;

  assert(G <= ELS_PER_THREAD);
  assert(blockDim.x == THREADS_PER_BLOCK);
  static_assert(sizeof(InnerNode) == sizeof(LeafNode));

  prefetch_node_t pref{};
  
  for (int k = 0; k < G; k ++) {
    VSMEM2(key, k) = keys[POS(k)];
  }
  
  pref.commit(&VSMEM2(v, 0), *root_p);
  pref.wait();

  for (int i = 0; i < G; i ++) {
    VSMEM2(f, i) = false;
    if (i)
      VSMEM2(v, i) = VSMEM2(v, 0);
  }
  // iteration 1: code 0 for i = 0
  // iteration 2: code 0 for i = 1, code 1 for i = 0
  // ...
  int finished = 0;
  for (int i = 0; i < G + STAGE; i ++) {
    if (finished == G)
      break ;
    for (int j = i >= G ? G - 1 : i; j >= 0; j --) {
      if ( VSMEM2(f, j) )
        break ;
      if (!(i < G && j == i))
        pref.wait();
      auto node = static_cast<const Node *> (&VSMEM2(v, j));
      if (node->type == Node::Type::INNER) {
        auto inner = static_cast<const InnerNode *>(node);
        int pos = inner->lower_bound(VSMEM2(key,j));
        pref.commit(&VSMEM2(v, j), inner->children[pos]);
      } else {
        auto leaf = static_cast<const LeafNode *>(node);
        auto &value = values[POS(j)];
        int pos = leaf->lower_bound(VSMEM2(key,j));
        if (pos < leaf->n_key && VSMEM2(key,j) == leaf->keys[pos]) {
          value = leaf->values[pos];
        } else {
          value = -1;
        }
        finished ++;
        VSMEM2(f, j) = true;
      }
    }
  }
}

#undef VSMEM2
#undef POS

void index(int64_t *keys, int64_t *values, int32_t n, Config cfg) {
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
    gets_parallel<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_keys, n, d_outs, tree.d_root_p_);
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
}  // namespace naive
}  // namespace btree