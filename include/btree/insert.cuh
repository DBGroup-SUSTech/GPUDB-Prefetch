#pragma once
#include "btree/common.cuh"

namespace btree {

__device__ __forceinline__ void put_olc(
    int key, int value, Node **root_p,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
restart:
  bool need_restart = false;
  Node *node = gutil::atomic_load(root_p);

  gutil::ull_t v = node->read_lock_or_restart(need_restart);
  if (need_restart || (node != gutil::atomic_load(root_p))) goto restart;

  InnerNode *parent = nullptr;
  gutil::ull_t parent_v;  // empty

  while (node->type == Node::Type::INNER) {
    InnerNode *inner = static_cast<InnerNode *>(node);
    // split preemptively if full
    if (inner->is_full()) {
      // lock parent and current
      if (parent) {
        parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
        if (need_restart) goto restart;
      }
      node->upgrade_to_write_lock_or_restart(v, need_restart);
      if (need_restart) {
        if (parent) parent->write_unlock();
        goto restart;
      }
      if (!parent && (node != gutil::atomic_load(root_p))) {
        node->write_unlock();
        goto restart;
      }

      // split
      int sep;
      InnerNode *new_inner = inner->split_alloc(sep, node_allocator);
      if (parent) {
        parent->insert(sep, new_inner);
      } else {
        // make_root
        InnerNode *new_root = node_allocator.malloc<InnerNode>();
        new_root->type = Node::Type::INNER;

        new_root->n_key = 1;
        new_root->keys[0] = sep;
        new_root->children[0] = inner;
        new_root->children[1] = new_inner;
        // root = new_root;
        gutil::atomic_store(root_p, static_cast<Node *>(new_root));
      }

      // unlock and restart
      node->write_unlock();
      if (parent) parent->write_unlock();
      goto restart;  // TODO: keep going instead of restart
    }

    // unlock parent
    if (parent) {
      parent->read_unlock_or_restart(parent_v, need_restart);
      if (need_restart) goto restart;
    }

    parent = inner;
    parent_v = v;

    // printf("inner.children %p\n", inner->children);
    int pos = inner->lower_bound(key);
    node = inner->children[pos];
    /// @note check_or_restart()'s usage:
    ///   1. The correctness can be achieved without check_or_restart()
    ///   2. check_or_restart() avoids reading illigal memory. The program
    ///      cannot pass sanitizer without check_or_restart()
    inner->check_or_restart(v, need_restart);
    if (need_restart) goto restart;

    v = node->read_lock_or_restart(need_restart);
    if (need_restart) goto restart;
  }

  // split leaf if full
  LeafNode *leaf = static_cast<LeafNode *>(node);

  if (leaf->is_full()) {
    // lock parent and current
    if (parent) {
      parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
      if (need_restart) goto restart;
    }
    node->upgrade_to_write_lock_or_restart(v, need_restart);
    if (need_restart) {
      if (parent) parent->write_unlock();
      goto restart;
    }
    // TODO: why check this
    if (!parent && (node != gutil::atomic_load(root_p))) {
      node->write_unlock();
      goto restart;
    }

    // split
    int sep;
    LeafNode *new_leaf = leaf->split_alloc(sep, node_allocator);
    if (parent) {
      parent->insert(sep, new_leaf);
    } else {
      // make root
      InnerNode *new_root = node_allocator.malloc<InnerNode>();
      new_root->type = Node::Type::INNER;

      new_root->n_key = 1;
      new_root->keys[0] = sep;
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      // root = new_root;
      gutil::atomic_store(root_p, static_cast<Node *>(new_root));
    }

    // unlock and restart
    node->write_unlock();
    if (parent) parent->write_unlock();
    goto restart;  // TODO: keep going instead of restart
  } else {
    // lock leaf node, release parent
    node->upgrade_to_write_lock_or_restart(v, need_restart);
    if (need_restart) goto restart;
    if (parent) {
      parent->read_unlock_or_restart(parent_v, need_restart);
      if (need_restart) {
        node->write_unlock();
        goto restart;
      }
    }

    // insert
    leaf->insert(key, value);
    node->write_unlock();
  }
}

__global__ void puts_olc(int *keys, int *values, int n, Node **root_p,
                         DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  // int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  // if (wid >= n) {
  //   return;
  // }
  // int lid_w = threadIdx.x % 32;
  // if (lid_w > 0) {
  //   return;
  // }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    put_olc(keys[i], values[i], root_p, node_allocator);
  }
}

}  // namespace btree