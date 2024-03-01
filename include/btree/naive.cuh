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
namespace naive {
constexpr int LANES_PER_WARP = 8;  // or 8

__device__ __forceinline__ void get(
    int32_t key, int32_t &value, const NodePtr root_ptr,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  NodePtr node_ptr = root_ptr;
  const Node *node = node_ptr.to_ptr<Node>(node_allocator);
  while (node->type == Node::Type::INNER) {
    // printf("node == inner\n");
    const InnerNode *inner = static_cast<const InnerNode *>(node);
    node_ptr = inner->children[inner->lower_bound(key)];
    node = node_ptr.to_ptr<Node>(node_allocator);
  }
  const LeafNode *leaf = static_cast<const LeafNode *>(node);
  int pos = leaf->lower_bound(key);
  // printf("key = %d, leaf = %d\n", key, leaf->keys[pos]);
  if (pos < leaf->n_key && key == leaf->keys[pos]) {
    value = leaf->values[pos];
  } else {
    value = -1;
  }
  return;
}

__global__ void gets_parallel(int32_t *keys, int n, int32_t *values,
                              const NodePtr *root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  // int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warpid = threadIdx.x / 32;
  int warplane = threadIdx.x % 32;
  if (warplane >= LANES_PER_WARP) return;

  int lid = warpid * LANES_PER_WARP + warplane;  // lane id
  int lanes_per_block = blockDim.x / 32 * LANES_PER_WARP;
  int stride = lanes_per_block * gridDim.x;

  for (int i = lanes_per_block * blockIdx.x + lid; i < n; i += stride) {
    get(keys[i], values[i], *root_p, node_allocator);
  }
}

__device__ __forceinline__ void get_cache(
    int32_t key, int32_t &value, const Node *cached_root,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  NodePtr node_ptr;
  const Node *node = cached_root;
  while (node->type == Node::Type::INNER) {
    // printf("node == inner\n");
    const InnerNode *inner = static_cast<const InnerNode *>(node);
    node_ptr = inner->children[inner->lower_bound(key)];
    node = node_ptr.to_ptr<Node>(node_allocator);
  }
  const LeafNode *leaf = static_cast<const LeafNode *>(node);
  int pos = leaf->lower_bound(key);
  // printf("key = %d, leaf = %d\n", key, leaf->keys[pos]);
  if (pos < leaf->n_key && key == leaf->keys[pos]) {
    value = leaf->values[pos];
  } else {
    value = -1;
  }
  return;
}
__global__ void gets_parallel_cache(
    int32_t *keys, int n, int32_t *values, const NodePtr *root_p,
    DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int warpid = threadIdx.x / 32;
  int warplane = threadIdx.x % 32;
  if (warplane >= LANES_PER_WARP) return;

  int lid = warpid * LANES_PER_WARP + warplane;  // lane id
  int lanes_per_block = blockDim.x / 32 * LANES_PER_WARP;
  int stride = lanes_per_block * gridDim.x;

  __shared__ InnerNode root;
  if (lanes_per_block * blockIdx.x + lid < n) {  // active block
    if (threadIdx.x == 0) {
      memcpy(&root, (*root_p).to_ptr<Node>(node_allocator), sizeof(InnerNode));
    }
    __syncthreads();
  }

  for (int i = lanes_per_block * blockIdx.x + lid; i < n; i += stride) {
    get_cache(keys[i], values[i], &root, node_allocator);
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