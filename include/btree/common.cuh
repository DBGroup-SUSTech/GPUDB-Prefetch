#pragma once
#include <assert.h>

#include "util/allocator.cuh"
#include "util/util.cuh"

namespace btree {
struct Node {
  enum class Type : int { NONE = 0, INNER, LEAF };
  gutil::ull_t version_lock_obsolete;
  Type type;

  // optimistic lock
  __device__ __forceinline__ gutil::ull_t read_lock_or_restart(
      bool &need_restart) const {
    if (((uint64_t)&version_lock_obsolete) % 8 != 0) {
      // TODO
      printf("misaligned version of node %p\n", this);
      assert(0);
    }
    gutil::ull_t version;
    version = gutil::atomic_load(&version_lock_obsolete);
    if (gutil::is_locked(version) || gutil::is_obsolete(version)) {
      need_restart = true;
    }
    return version;
  }

  __device__ __forceinline__ void read_unlock_or_restart(
      gutil::ull_t start_read, bool &need_restart) const {
    need_restart = (start_read != gutil::atomic_load(&version_lock_obsolete));
  }

  __device__ __forceinline__ gutil::ull_t check_or_restart(
      gutil::ull_t start_read, bool &need_restart) const {
    read_unlock_or_restart(start_read, need_restart);
  }

  __device__ __forceinline__ void upgrade_to_write_lock_or_restart(
      gutil::ull_t &version, bool &need_restart) {
    if (version == atomicCAS(&version_lock_obsolete, version, version + 0b10)) {
      version = version + 0b10;
    } else {
      need_restart = true;
    }
  }

  __device__ __forceinline__ void write_unlock() {
    atomicAdd(&version_lock_obsolete, 0b10);
  }

  __device__ __forceinline__ void write_unlock_obsolete() {
    atomicAdd(&version_lock_obsolete, 0b11);
  }
};

struct InnerNode : public Node {
  static const int MAX_ENTRIES = 16;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;

  Node *children[MAX_ENTRIES];
  int keys[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES - 1;
  }

  // return the position to insert, equal to binary search
  __device__ __forceinline__ int lower_bound(int key) const {
    int lower = 0;
    int upper = n_key;
    while (lower < upper) {
      int mid = (upper - lower) / 2 + lower;
      int cmp = util::scalar_cmp(key, keys[mid]);
      if (cmp < 0) {
        upper = mid;
      } else if (cmp > 0) {
        lower = mid + 1;
      } else {
        return mid;
      }
    }
    return lower;
  }

  __device__ __forceinline__ InnerNode *split_alloc(
      int &sep, DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    InnerNode *new_inner = node_allocator.malloc<InnerNode>();
    new_inner->type = Node::Type::INNER;

    new_inner->n_key = n_key / 2;
    this->n_key = n_key - new_inner->n_key - 1;
    sep = keys[n_key];
    // copy n keys and n + 1 childs
    memcpy(new_inner->keys, keys + n_key + 1, sizeof(int) * new_inner->n_key);
    memcpy(new_inner->children, children + n_key + 1,
           sizeof(Node *) * (new_inner->n_key + 1));
    return new_inner;
  }

  __device__ __forceinline__ void insert(int key, Node *child) {
    assert(n_key < MAX_ENTRIES - 1);
    int pos = lower_bound(key);
    util::memmove_forward(keys + pos + 1, keys + pos,
                          sizeof(int) * (n_key - pos));
    util::memmove_forward(children + pos + 1, children + pos,
                          sizeof(Node *) * (n_key + 1 - pos));
    assert(pos < MAX_ENTRIES);
    keys[pos] = key;
    children[pos] = child;

    // printf("children %p\n", children);
    assert(pos < MAX_ENTRIES && pos >= 0);
    assert(((uint64_t)children) % 8 == 0);
    auto tmp = children[pos];
    children[pos] = children[pos + 1];
    children[pos + 1] = tmp;

    n_key++;
  }
};

struct LeafNode : public Node {
  static const int MAX_ENTRIES = 16;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;
  int keys[MAX_ENTRIES];
  int values[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES;
  }

  __device__ __forceinline__ int lower_bound(int key) const {
    int lower = 0;
    int upper = n_key;
    while (lower < upper) {
      // printf("lower=%d, upper=%d, n_key=%d\n", lower, upper, n_key);
      int mid = (upper - lower) / 2 + lower;
      // int cmp = util::bytes_cmp(key, key_size, keys[mid], keys_sizes[mid]);
      int cmp = util::scalar_cmp(key, keys[mid]);
      if (cmp < 0) {
        upper = mid;
      } else if (cmp > 0) {
        lower = mid + 1;
      } else {
        return mid;
      }
    }
    return lower;
  }

  __device__ __forceinline__ LeafNode *split_alloc(
      int &sep, DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    LeafNode *new_leaf = node_allocator.malloc<LeafNode>();
    new_leaf->type = Node::Type::LEAF;
    new_leaf->n_key = n_key / 2;
    n_key = n_key - new_leaf->n_key;

    sep = keys[n_key - 1];

    memcpy(new_leaf->keys, keys + n_key, sizeof(int) * new_leaf->n_key);
    memcpy(new_leaf->values, values + n_key, sizeof(int) * new_leaf->n_key);

    return new_leaf;
  }

  __device__ __forceinline__ void insert(int key, int value) {
    if (n_key) {
      // printf("n_key = %d\n", n_key);
      int pos = lower_bound(key);
      // printf("pos = %d\n", pos);
      if (pos < n_key && key == keys[pos]) {
        // replace
        values[pos] = value;
        return;
      }
      util::memmove_forward(keys + pos + 1, keys + pos,
                            sizeof(int) * (n_key - pos));
      util::memmove_forward(values + pos + 1, values + pos,
                            sizeof(int) * (n_key - pos));
      keys[pos] = key;
      values[pos] = value;
    } else {
      keys[0] = key;
      values[0] = value;
    }
    n_key++;
  }
};

__global__ void allocate_root(Node **root_p) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  LeafNode *root = new LeafNode{};
  root->type = Node::Type::LEAF;
  *root_p = root;
}

class BTree {
 public:
  Node **d_root_p_;
  DynamicAllocator<ALLOC_CAPACITY> allocator_;

 public:
  BTree() {
    CHKERR(cutil::DeviceAlloc(d_root_p_, 1));
    allocate_root<<<1, 1>>>(d_root_p_);
    CHKERR(cudaDeviceSynchronize());
  }
};

struct Config {
  int build_gridsize = -1;
  int build_blocksize = -1;
  int probe_gridsize = -1;
  int probe_blocksize = -1;
};

}  // namespace btree