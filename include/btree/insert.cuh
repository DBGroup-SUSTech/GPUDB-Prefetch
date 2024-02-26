#pragma once
#include "btree/common.cuh"

namespace btree {

__device__ __forceinline__ NodePtr atomic_load(const NodePtr *addr) {
  const volatile NodePtr *vaddr = addr;  // volatile to bypass cache
  __threadfence();  // for seq_cst loads. Remove for acquire semantics.
  const NodePtr value(vaddr->offset_);
  // fence to ensure that dependent reads are correctly ordered
  __threadfence();
  return value;
}

// addr must be aligned properly.
__device__ __forceinline__ void atomic_store(NodePtr *addr, NodePtr value) {
  volatile NodePtr *vaddr = addr;  // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other
  // threads
  __threadfence();
  vaddr->offset_ = value.offset_;
}

__device__ __forceinline__ void put_olc(
    int32_t key, int32_t value, NodePtr *root_p,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
restart:
  bool need_restart = false;
  NodePtr node_ptr = atomic_load(root_p);
  // Node *node = atomic_load(root_p);
  Node *node = node_ptr.to_ptr<Node>(node_allocator);

  gutil::ull_t v = node->read_lock_or_restart(need_restart);
  if (need_restart || (node_ptr != atomic_load(root_p))) goto restart;

  InnerNode *parent = nullptr;
  gutil::ull_t parent_v;  // empty

  while (node->type == Node::Type::INNER) {
    InnerNode *inner = static_cast<InnerNode *>(node);
    NodePtr inner_ptr = node_ptr;
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
      if (!parent && (node_ptr != atomic_load(root_p))) {
        node->write_unlock();
        goto restart;
      }

      // split
      int32_t sep;
      NodePtr new_inner_ptr = inner->split_alloc(sep, node_allocator);
      InnerNode *new_inner = new_inner_ptr.to_ptr<InnerNode>(node_allocator);
      if (parent) {
        parent->insert(sep, new_inner_ptr);
      } else {
        // make_root
        NodePtr new_root_ptr = node_allocator.malloc_obj<InnerNode>();
        InnerNode *new_root = new_root_ptr.to_ptr<InnerNode>(node_allocator);
        // InnerNode *new_root = node_allocator.malloc<InnerNode>();
        new_root->type = Node::Type::INNER;

        new_root->n_key = 1;
        new_root->keys[0] = sep;
        new_root->children[0] = inner_ptr;
        new_root->children[1] = new_inner_ptr;
        // root = new_root;
        atomic_store(root_p, new_root_ptr);
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
    node_ptr = inner->children[pos];
    node = node_ptr.to_ptr<Node>(node_allocator);
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
  NodePtr leaf_ptr = node_ptr;

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
    if (!parent && (node_ptr != atomic_load(root_p))) {
      node->write_unlock();
      goto restart;
    }

    // split
    int32_t sep;
    NodePtr new_leaf_ptr = leaf->split_alloc(sep, node_allocator);
    LeafNode *new_leaf = new_leaf_ptr.to_ptr<LeafNode>(node_allocator);
    if (parent) {
      parent->insert(sep, new_leaf_ptr);
    } else {
      // make root
      NodePtr new_root_ptr = node_allocator.malloc_obj<InnerNode>();
      InnerNode *new_root = new_root_ptr.to_ptr<InnerNode>(node_allocator);
      new_root->type = Node::Type::INNER;

      new_root->n_key = 1;
      new_root->keys[0] = sep;
      new_root->children[0] = leaf_ptr;
      new_root->children[1] = new_leaf_ptr;
      // root = new_root;
      atomic_store(root_p, new_root_ptr);
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

__global__ void puts_olc(int32_t *keys, int32_t *values, int n, NodePtr *root_p,
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