#include "util/util.cuh"

#define ALLOC_CAPACITY ((uint64_t(1) << 32))  // 4GB for node

// capacity aligned to 4 bytes
template <uint64_t CAPACITY>
class DynamicAllocator {
 public:
  DynamicAllocator() {
    CHKERR(cutil::DeviceAlloc(d_pool_aligned4_, CAPACITY / 4));
    CHKERR(cutil::DeviceSet(d_pool_aligned4_, 0x00, CAPACITY / 4));
    CHKERR(cutil::DeviceAlloc(d_count_aligned4_, 1));
    CHKERR(cutil::DeviceSet(d_count_aligned4_, 0x00, 1));
  }
  ~DynamicAllocator() {
    // TODO: free them
    // free_all();
  }

  void free_all() {
    if (d_pool_aligned4_ != nullptr) {
      CHKERR(cutil::DeviceFree(d_pool_aligned4_));
    }
    if (d_count_aligned4_ != nullptr) {
      CHKERR(cutil::DeviceFree(d_count_aligned4_));
    }
    // TODO: do not check the error here
    // gutil::DeviceFree(d_pool_aligned4_);
    // gutil::DeviceFree(d_count_aligned4_);
    d_pool_aligned4_ = nullptr;
    d_count_aligned4_ = nullptr;
    // printf("%lu GPU memory is freed\n", CAPACITY / 4 *
    // sizeof(d_pool_aligned4_));
  }

  DynamicAllocator &operator=(const DynamicAllocator &rhs) {
    d_pool_aligned4_ = rhs.d_pool_aligned4_;
    d_count_aligned4_ = rhs.d_count_aligned4_;
    return *this;
  }

  template <typename T>
  __device__ __forceinline__ T *malloc() {
    assert(*d_count_aligned4_ < CAPACITY);
    assert(sizeof(T) % 4 == 0);
    uint32_t old_count = atomicAdd(d_count_aligned4_, sizeof(T) / 4);
    return reinterpret_cast<T *>(d_pool_aligned4_ + old_count);
  }

  __device__ __forceinline__ uint8_t *malloc(int n) {
    assert(*d_count_aligned4_ < CAPACITY);
    assert(n > 0 && n % 4 == 0);
    uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
    return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
  }

  __device__ __forceinline__ uint32_t allocated() const {
    return *d_count_aligned4_ * 4;
  }

 private:
  uint32_t *d_pool_aligned4_;
  uint32_t *d_count_aligned4_;
};
