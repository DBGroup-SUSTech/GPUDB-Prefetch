#pragma once
#include "btree/common.cuh"
#include "btree/insert.cuh"

/// @note
/// B-Tree, fanout = 16, leaf node size = 16
/// int32_t keys, int32_t values
/// insert with OLC
/// search with

namespace btree {
namespace naive {
__device__ __forceinline__ void get(int key, int &value, const Node *root) {
  const Node *node = root;
  while (node->type == Node::Type::INNER) {
    const InnerNode *inner = static_cast<const InnerNode *>(node);
    node = inner->children[inner->lower_bound(key)];
  }
  const LeafNode *leaf = static_cast<const LeafNode *>(node);
  int pos = leaf->lower_bound(key);
  if (pos < leaf->n_key && key == leaf->keys[pos]) {
    value = leaf->values[pos];
  } else {
    value = -1;
  }
  return;
}

__global__ void gets_parallel(int *keys, int n, int *values, int *values_sizes,
                              const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  get(keys[tid], values[tid], *root_p);
}

void index(int32_t *key, int32_t *value, int32_t n, int32_t *) {
  // TODO
}
}  // namespace naive
}  // namespace btree