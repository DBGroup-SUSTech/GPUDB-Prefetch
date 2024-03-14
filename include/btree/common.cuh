#pragma once
#include <assert.h>
#include <cuda_pipeline.h>

#include "util/allocator.cuh"
#include "util/util.cuh"

namespace btree {

struct Node;
struct InnerNode;
struct LeafNode;

struct NodePtr {
  int offset_;
  __device__ __forceinline__ NodePtr(int32_t offset) : offset_(offset) {}
  __device__ __forceinline__ NodePtr(const NodePtr &ptr) : offset_(ptr.offset_) {}

  __device__ __forceinline__ NodePtr(){};

  template <typename T>
  __device__ __forceinline__ T *to_ptr(
      DynamicAllocator<ALLOC_CAPACITY> &node_allocator) const;

  __device__ __forceinline__ bool operator==(NodePtr ptr) {
    return ptr.offset_ == offset_;
  }
  __device__ __forceinline__ bool operator!=(NodePtr ptr) {
    return ptr.offset_ != offset_;
  }
  __device__ __forceinline__ NodePtr &operator=(const NodePtr &ptr) {
    offset_ = ptr.offset_;
  }
};

struct Node {
  enum class Type : int { NONE = 0, INNER, LEAF };
  gutil::ull_t version_lock_obsolete;
  Type type;

  // optimistic lock
  __device__ __forceinline__ gutil::ull_t read_lock_or_restart(
      bool &need_restart) const {
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

#define COMMON_PDIST 2
#define MACRO_MAX_ENTRIES 14
#define MACRO_BLOCKSIZE 128
// TODO: the best case of MAX_ENTRIES for naive is >= 16

struct InnerNode : public Node {
  static const int MAX_ENTRIES = MACRO_MAX_ENTRIES;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;
  int32_t keys[MAX_ENTRIES];
  NodePtr children[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES - 1;
  }

  // return the position to insert, equal to binary search
  __device__ __forceinline__ int lower_bound(int32_t key) const {
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

  __device__ __forceinline__ NodePtr
  split_alloc(int32_t &sep, DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    NodePtr ptr = node_allocator.malloc_obj<InnerNode>();
    InnerNode *new_inner = ptr.to_ptr<InnerNode>(node_allocator);
    new_inner->type = Node::Type::INNER;

    new_inner->n_key = n_key / 2;
    this->n_key = n_key - new_inner->n_key - 1;
    sep = keys[n_key];
    // copy n keys and n + 1 childs
    memcpy(new_inner->keys, keys + n_key + 1,
           sizeof(int32_t) * new_inner->n_key);
    memcpy(new_inner->children, children + n_key + 1,
           sizeof(NodePtr) * (new_inner->n_key + 1));
    return ptr;
  }

  __device__ __forceinline__ void insert(int32_t key, NodePtr child) {
    assert(n_key < MAX_ENTRIES - 1);
    int pos = lower_bound(key);
    util::memmove_forward(keys + pos + 1, keys + pos,
                          sizeof(int32_t) * (n_key - pos));
    util::memmove_forward(children + pos + 1, children + pos,
                          sizeof(NodePtr) * (n_key + 1 - pos));
    // assert(pos < MAX_ENTRIES);
    keys[pos] = key;
    children[pos] = child;

    // printf("children %p\n", children);
    // assert(pos < MAX_ENTRIES && pos >= 0);
    // assert(((uint32_t)children) % 8 == 0);
    auto tmp = children[pos];
    children[pos] = children[pos + 1];
    children[pos + 1] = tmp;

    n_key++;
  }
};

struct LeafNode : public Node {
  static const int MAX_ENTRIES = MACRO_MAX_ENTRIES;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;
  int32_t keys[MAX_ENTRIES];
  int32_t values[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES;
  }

  __device__ __forceinline__ int lower_bound(int32_t key) const {
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

  __device__ __forceinline__ NodePtr
  split_alloc(int32_t &sep, DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    // LeafNode *new_leaf = node_allocator.malloc<LeafNode>();

    NodePtr ptr = node_allocator.malloc_obj<LeafNode>();
    LeafNode *new_leaf = ptr.to_ptr<LeafNode>(node_allocator);
    new_leaf->type = Node::Type::LEAF;
    new_leaf->n_key = n_key / 2;
    n_key = n_key - new_leaf->n_key;

    sep = keys[n_key - 1];

    memcpy(new_leaf->keys, keys + n_key, sizeof(int32_t) * new_leaf->n_key);
    memcpy(new_leaf->values, values + n_key, sizeof(int32_t) * new_leaf->n_key);

    return ptr;
  }

  __device__ __forceinline__ void insert(int32_t key, int32_t value) {
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
                            sizeof(int32_t) * (n_key - pos));
      util::memmove_forward(values + pos + 1, values + pos,
                            sizeof(int32_t) * (n_key - pos));
      keys[pos] = key;
      values[pos] = value;
    } else {
      keys[0] = key;
      values[0] = value;
    }
    n_key++;
  }
};

__global__ void allocate_root(NodePtr *root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  NodePtr root_ptr = node_allocator.malloc_obj<LeafNode>();
  LeafNode *root = root_ptr.to_ptr<LeafNode>(node_allocator);
  root->type = Node::Type::LEAF;
  *root_p = root_ptr;
}

template <typename T>
__device__ __forceinline__ T *NodePtr::to_ptr(
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) const {
  return reinterpret_cast<T *>(
      reinterpret_cast<InnerNode *>(node_allocator.get_started_ptr()) +
      offset_);
}

class BTree {
 public:
  NodePtr *d_root_p_;
  DynamicAllocator<ALLOC_CAPACITY> allocator_;

 public:
  BTree() {
    CHKERR(cutil::DeviceAlloc(d_root_p_, 1));
    allocate_root<<<1, 1>>>(d_root_p_, allocator_);
    CHKERR(cudaDeviceSynchronize());
  }
};

struct Config {
  int build_gridsize = -1;
  int build_blocksize = -1;
  int probe_gridsize = -1;
  int probe_blocksize = -1;
};

struct prefetch_node_t {
  int8_t pending = 0;
  __device__ __forceinline__ void commit(void *__restrict__ dst_shared,
                                         const void *__restrict__ src_global) {
    // 280 = 16 * 17 + 8
    auto dst = reinterpret_cast<char *>(dst_shared);
    auto src = reinterpret_cast<const char *>(src_global);
    for (int i = 0; i < sizeof(InnerNode) / 8; i += 1) {
      __pipeline_memcpy_async(dst, src, 8);
      dst += 8, src += 8;
    }
    __pipeline_commit();
    // memcpy(dst_shared, src_global, size_and_align);
    ++pending;
    // assert(pending <= INT8_MAX);
  }

  __device__ __forceinline__ void wait() {
    // assert(pending);
    // printf("  tid=%d, wait=%d\n", threadIdx.x, pending - 1);
    __pipeline_wait_prior(pending - 1);
    --pending;
  }
};

}  // namespace btree