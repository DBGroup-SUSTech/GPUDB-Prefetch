#pragma once
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

namespace gutil {
using ull_t = unsigned long long;
}

namespace cutil {

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