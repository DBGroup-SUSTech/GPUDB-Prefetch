#pragma once
#include "config.cuh"
#include "util/util.cuh"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_pipeline.h>
#include <fstream>

#define G 8
#define EARLY_AGGREGATE
#define PREFETCHF

typedef enum state_t {
  HASH,
  NEXT,
  MATCH,
  PAYLOAD,
  Done,
} state_t;

struct prefetch_t {
  int pending = 0; // number of pending requests
  __device__ __forceinline__ void commit(void *__restrict__ dst_shared,
                                         const void *__restrict__ src_global,
                                         size_t size_and_align,
                                         size_t zfill = 0UL) {
    __pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill);
    __pipeline_commit();
    ++pending;
  }
  __device__ __forceinline__ void wait() {
    assert(pending);
    __pipeline_wait_prior(pending - 1);
    --pending;
  }
};

// Non-partitioned Hash Join in Global Memory
// Unique hash table
// Late materialization
// Output payloads to the location of input
// 4byte-4byte table
// hash table do not save key/value, only save row id
// uses bucket chain to solve conflict, bucket size = 1

namespace join {
namespace gp {
/// @breif build hash table for R
/// @param [in]     r_key       all keys of relation R
/// @param [in]     n           number of keys
/// @param [in out] ht_link     hash table links, init -> 0
/// @param [in out] ht_slot     hash table slots, init -> 0, header of links
/// @param [in]     ht_size_log hash table size = 1 << ht_size_log (>= n)
__global__ void build_ht(int32_t *r_key, int32_t r_n, int32_t *ht_link,
                         int32_t *ht_slot, int32_t ht_size_log) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  // TODO: try vector
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < r_n; i += stride) {
    // TODO: try other hash functions
    // TODO: it is actually late materialization
    int32_t hval = r_key[i] & ht_mask; // keys[i] % ht_size
    int32_t last = atomicExch(ht_slot + hval, i + 1);
    ht_link[i] = last;
  }
}

/// @brief Use S to probe
/// @param [in]     s_key       all keys of S
/// @param [in]     s_payload   all payload of S
/// @param [in]     s_n         number of S items
/// @param [in]     r_key       all keys of R
/// @param [in]     r_payload   all payload of R
/// @param [in]     ht_link     R hash table link
/// @param [in]     ht_slot     R hash table slot
/// @param [in]     ht_size_log R hash table size = ht_size_log << 1
/// @param [in]     o_aggr      output aggregation results for matched r's

__global__ void probe_ht(int32_t *s_key, int32_t *s_payload, int32_t s_n,
                         int32_t *r_key, int32_t *r_payload, int32_t *ht_link,
                         int32_t *ht_slot, int32_t ht_size_log,
                         int32_t *o_aggr) {
  // share_memory:
  extern __shared__ int shared_buffer[];
  // __shared__ int *output_buffer = shared_buffer + blockDim.x * G;

  // register for state information:
  int32_t reg_s_key[G];
  int32_t reg_s_payload[G];
  int32_t reg_r_tuple_id[G];
  // initial state is 0
  state_t reg_state[G];
  int32_t reg_r_key = -1;
  int32_t reg_r_payload = -1;

  int32_t finish_match_num = 0;

  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int s_tuple_id = -1;
  int shared_buffer_idx = -1;

  int aggr_local = 0;

  prefetch_t pref{};

  for (size_t i = tid; i < s_n; i += stride * G) {
    // reset the registers
    int32_t reg_r_key = -1;
    int32_t reg_r_payload = -1;
    int32_t finish_match_num = 0;
#pragma unroll
    for (size_t j = 0; j < G;
         j++) // code 0, access s's key and payload, and calculate the slot_num.
    {
      // set the state to HASH
      reg_state[j] = HASH;
      s_tuple_id = i + j * stride;
      if (s_tuple_id >= s_n) {
        break;
      }
      shared_buffer_idx = blockDim.x * j + threadIdx.x;
      // sequential read, and coalesced access, do not need prefetch
      reg_s_key[j] = s_key[s_tuple_id];
      reg_s_payload[j] = s_payload[s_tuple_id];
      // calculate slot_num
      int slot_num = reg_s_key[j] & ht_mask;
#ifdef PREFETCH
      pref.commit(&shared_buffer[shared_buffer_idx], &ht_slot[slot_num],
                  sizeof(int32_t));
#else
      memcpy(&shared_buffer[shared_buffer_idx], &ht_slot[slot_num],
             sizeof(int32_t));
#endif
      reg_state[j] = NEXT;
    }

    while (finish_match_num != G) {
#pragma unroll
      for (size_t j = 0; j < G; j++) {
        s_tuple_id = i + j * stride;
        switch (reg_state[j]) {
        case NEXT:
          shared_buffer_idx = blockDim.x * j + threadIdx.x;
#ifdef PREFETCH
          // TODO
          pref.wait();
#endif
          reg_r_tuple_id[j] = shared_buffer[shared_buffer_idx] - 1;
          if (reg_r_tuple_id[j] == -1) {
            finish_match_num++;
            reg_state[j] = Done;
          } else {
#ifdef PREFETCH
            pref.commit(&shared_buffer[shared_buffer_idx],
                        &r_key[reg_r_tuple_id[j]], sizeof(int32_t));
#else
            memcpy(&shared_buffer[shared_buffer_idx], &r_key[reg_r_tuple_id[j]],
                   sizeof(int32_t));
#endif
            reg_state[j] = MATCH;
          }
          break;
        // if block in code 0(s_tuple_id >= s_n)
        case HASH:
          finish_match_num++;
          reg_state[j] = Done;
        default:
          break;
        }
      }
#pragma unroll
      for (size_t j = 0; j < G; j++) {
        s_tuple_id = i + j * stride;
        switch (reg_state[j]) {
        case MATCH:
          shared_buffer_idx = blockDim.x * j + threadIdx.x;

#ifdef PREFETCH
          pref.wait();
#endif
          reg_r_key = shared_buffer[shared_buffer_idx];
          if (reg_r_key == reg_s_key[j]) {
#ifdef PREFETCH
            pref.commit(&shared_buffer[shared_buffer_idx],
                        &r_payload[reg_r_tuple_id[j]], sizeof(int32_t));
#else
            memcpy(&shared_buffer[shared_buffer_idx],
                   &r_payload[reg_r_tuple_id[j]], sizeof(int32_t));
#endif
            reg_state[j] = PAYLOAD;
          } else {
#ifdef PREFETCH
            pref.commit(&shared_buffer[shared_buffer_idx],
                        &ht_link[reg_r_tuple_id[j]], sizeof(int32_t));
#else
            memcpy(&shared_buffer[shared_buffer_idx],
                   &ht_link[reg_r_tuple_id[j]], sizeof(int32_t));
#endif
            reg_state[j] = NEXT;
          }
          break;
        default:
          break;
        }
      }
#pragma unroll
      for (size_t j = 0; j < G; j++) {
        s_tuple_id = i + j * stride;
        switch (reg_state[j]) {
        case PAYLOAD:
          shared_buffer_idx = blockDim.x * j + threadIdx.x;
#ifdef PREFETCH
          pref.wait();
#endif
          reg_r_payload = shared_buffer[shared_buffer_idx];
          // printf("%d, %d \n", reg_s_payload[j], reg_r_payload);
          aggr_fn_local(reg_s_payload[j], reg_r_payload, &aggr_local);
#ifdef PREFETCH
          pref.commit(&shared_buffer[shared_buffer_idx],
                      &ht_link[reg_r_tuple_id[j]], sizeof(int32_t));

#else
          memcpy(&shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[j]],
                 sizeof(int32_t));
#endif
          reg_state[j] = NEXT;
          break;
        default:
          break;
        }
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key,
         int32_t *s_payload, int32_t s_n, Config cfg) {
  CHKERR(cudaDeviceReset());

  int32_t *d_r_key = nullptr, *d_r_payload = nullptr;
  int32_t *d_s_key = nullptr, *d_s_payload = nullptr;
  CHKERR(cutil::DeviceAlloc(d_r_key, r_n));
  CHKERR(cutil::DeviceAlloc(d_r_payload, r_n));
  CHKERR(cutil::DeviceAlloc(d_s_key, s_n));
  CHKERR(cutil::DeviceAlloc(d_s_payload, s_n));

  CHKERR(cutil::CpyHostToDevice(d_r_key, r_key, r_n));
  CHKERR(cutil::CpyHostToDevice(d_r_payload, r_payload, r_n));
  CHKERR(cutil::CpyHostToDevice(d_s_key, s_key, s_n));
  CHKERR(cutil::CpyHostToDevice(d_s_payload, s_payload, s_n));

  int32_t ht_size_log = cutil::log2(r_n);
  int32_t ht_size = 1 << ht_size_log;

  fmt::print("ht_size = {}, ht_size_log = {}\n", ht_size, ht_size_log);

  int32_t *d_ht_link = nullptr, *d_ht_slot = nullptr;
  CHKERR(cutil::DeviceAlloc(d_ht_link, ht_size));
  CHKERR(cutil::DeviceAlloc(d_ht_slot, ht_size));
  CHKERR(cutil::DeviceSet(d_ht_link, 0, ht_size));
  CHKERR(cutil::DeviceSet(d_ht_slot, 0, ht_size));

  // int32_t* d_o_payload = nullptr;
  // CHKERR(cutil::DeviceAlloc(d_o_payload, s_n));
  // CHKERR(cutil::DeviceSet(d_o_payload, 0, s_n));

  int32_t *d_aggr;
  CHKERR(cutil::DeviceAlloc(d_aggr, 1));
  CHKERR(cutil::DeviceSet(d_aggr, 0, 1));

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
  fmt::print("Build: {} blocks * {} threads"
             "Probe: {} blocks * {} threads\n",
             cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
             cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_r_key, r_n, d_ht_link, d_ht_slot, ht_size_log);
    CHKERR(
        cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
  }

  {
    CHKERR(
        cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
    int smem_size = G * cfg.probe_blocksize * sizeof(int);
    probe_ht<<<cfg.probe_gridsize, cfg.probe_blocksize, smem_size, stream>>>(
        d_s_key, d_s_payload, s_n, d_r_key, d_r_payload, d_ht_link, d_ht_slot,
        ht_size_log, d_aggr);
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

  fmt::print("Join GP (bucket size = 1)\n"
             "[build(R), {} ms, {} tps (S)]\n"
             "[probe(S), {} ms, {} tps (R)]\n",
             ms_build, r_n * 1.0 / ms_build * 1000, ms_probe,
             s_n * 1.0 / ms_probe * 1000);

  // CHKERR(cutil::CpyDeviceToHost(o_payload, d_o_payload, s_n));
  int32_t aggr;
  CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));
  return aggr;
}
} // namespace gp
} // namespace join