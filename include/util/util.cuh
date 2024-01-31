#pragma once
#include <fmt/format.h>
#include <stdint.h>
#include <stdio.h>
#define CHKERR(call)                                          \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

namespace util {
template <typename T>
__host__ __device__ __forceinline__ int scalar_cmp(T l, T r) {
  if (l == r)
    return 0;
  else if (l > r)
    return 1;
  else
    return -1;
}
// dest >= src
__host__ __device__ __forceinline__ void memmove_forward(void *dest,
                                                         const void *src,
                                                         size_t n) {
  assert(dest > src);
  if (n == 0) {
    return;
  }
  uint8_t *dest_ = static_cast<uint8_t *>(dest);
  const uint8_t *src_ = static_cast<const uint8_t *>(src);
  do {
    n--;
    dest_[n] = src_[n];
  } while (n > 0);
}
}  // namespace util

namespace gutil {
using ull_t = unsigned long long;

template <typename T>
__device__ __forceinline__ T atomic_exch_64(T *address, T val) {
  return reinterpret_cast<T>(atomicExch(reinterpret_cast<ull_t *>(address),
                                        reinterpret_cast<ull_t>(val)));
}

// optimistic locks API
__device__ __forceinline__ bool is_locked(ull_t version) {
  return ((version & 0b10) == 0b10);
}
__device__ __forceinline__ bool is_obsolete(ull_t version) {
  return ((version & 1) == 1);
}

template <typename T>
__device__ __forceinline__ T atomic_load(const T *addr) {
  const volatile T *vaddr = addr;  // volatile to bypass cache
  __threadfence();  // for seq_cst loads. Remove for acquire semantics.
  const T value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence();
  return value;
}

// addr must be aligned properly.
template <typename T>
__device__ __forceinline__ void atomic_store(T *addr, T value) {
  volatile T *vaddr = addr;  // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other
  // threads
  __threadfence();
  *vaddr = value;
}

}  // namespace gutil

namespace cutil {

std::string rel_fname(bool unique, const char *rel, int32_t n, double skew) {
  std::string flag = unique ? "unique" : "nonunique";
  return fmt::format("data/{}_{}_{}_{}.bin", flag, rel, skew, n);
}

template <typename T, typename N>
std::string fmt_arr(T *arr, N n, bool sort = false) {
  auto vec = std::vector<T>(arr, arr + n);
  if (sort) std::sort(vec.begin(), vec.end());
  return fmt::format("{}", fmt::join(vec, " "));
}

template <typename T>
cudaError_t CpyDeviceToHost(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

template <typename T>
cudaError_t CpyDeviceToHostAsync(T *dst, const T *src, size_t count,
                                 cudaStream_t st) {
  return cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost,
                         st);
}

template <typename T>
cudaError_t CpyHostToDevice(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
}

template <typename T>
cudaError_t CpyHostToDeviceAsync(T *dst, const T *src, size_t count,
                                 cudaStream_t st) {
  return cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice,
                         st);
}

template <typename T>
cudaError_t PinHost(T *src, size_t count) {
  return cudaHostRegister((void *)src, sizeof(T) * count,
                          cudaHostRegisterDefault);
}

template <typename T>
cudaError_t UnpinHost(T *src) {
  return cudaHostUnregister((void *)src);
}

template <typename T>
cudaError_t DeviceAlloc(T *&data, size_t count) {
  // printf("DeviceAlloc: %lu\n", sizeof(T) * count);
  return cudaMalloc((void **)&data, sizeof(T) * count);
}

template <typename T>
cudaError_t DeviceSet(T *data, uint8_t value, size_t count) {
  return cudaMemset(data, value, sizeof(T) * count);
}

template <typename T>
cudaError_t DeviceFree(T *data) {
  return cudaFree(data);
}

static inline uint32_t log2(const uint32_t x) {
  uint32_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  return y;
}

}  // namespace cutil