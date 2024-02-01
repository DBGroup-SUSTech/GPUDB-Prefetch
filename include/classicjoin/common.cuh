#pragma once
#include <cuda_pipeline.h>
// pre-allocated entries
namespace classicjoin {

// Non-partitioned Hash Join in Global Memory
// Non-Unique hash table
// Early Materialization
// Output payloads to the location of input
// 4byte-4byte table
// hash table save key/value
// uses bucket chain to solve conflict, bucket size = 1

struct Config {
  int build_gridsize = -1;
  int build_blocksize = -1;
  int probe_gridsize = -1;
  int probe_blocksize = -1;
};

struct Entry;
struct EntryHeader {
  Entry* next;  // pointer and dynamic allocated entry
  // int next;  // offset and static allocated entries
};

struct Tuple {
  int k;
  int v;
};

struct Entry {
  Tuple tuple;
  EntryHeader header;  // header in the next side
};

__device__ __forceinline__ void aggr_fn_local(int32_t r_payload,
                                              int32_t s_payload,
                                              int32_t* aggr) {
  *aggr += r_payload * s_payload;
}

__device__ __forceinline__ void aggr_fn_atomic(int32_t r_payload,
                                               int32_t s_payload,
                                               int32_t* aggr) {
  atomicAdd(aggr, r_payload * s_payload);
}

__device__ __forceinline__ void aggr_fn_global(int32_t aggr_local,
                                               int32_t* aggr_global) {
  atomicAdd(aggr_global, aggr_local);
}

void col_to_row(int* key, int* payload, Tuple* tuples, int n) {
  for (int i = 0; i < n; ++i) {
    tuples[i].k = key[i];
    tuples[i].v = payload[i];
  }
}

struct prefetch_t {
  int8_t pending = 0;
  __device__ __forceinline__ void commit(void* __restrict__ dst_shared,
                                         const void* __restrict__ src_global,
                                         size_t size_and_align,
                                         size_t zfill = 0UL) {
    // __pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill);
    // __pipeline_commit();
    memcpy(dst_shared, src_global, size_and_align);
    ++pending;
    // assert(pending <= INT8_MAX);
  }

  __device__ __forceinline__ void commit_k(void* __restrict__ dst_shared,
                                           const void* __restrict__ src_global,
                                           size_t size_and_align, int k,
                                           size_t zfill = 0UL) {
    // __pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill);
    // __pipeline_commit();
    memcpy(dst_shared, src_global, size_and_align);
    ++pending;
    // assert(pending <= INT8_MAX);
    // printf("  tid=%d, commit k=%d, pending=%d\n", threadIdx.x, k, pending);
  }
  __device__ __forceinline__ void wait() {
    // assert(pending);
    // printf("  tid=%d, wait=%d\n", threadIdx.x, pending - 1);
    // __pipeline_wait_prior(pending - 1);
    --pending;
  }

  __device__ __forceinline__ void wait_k(int k) {
    // assert(pending);
    // printf("  tid=%d, k=%d, wait=%d\n", threadIdx.x, k, pending - 1);
    // __pipeline_wait_prior(pending - 1);
    --pending;
  }
};

}  // namespace classicjoin