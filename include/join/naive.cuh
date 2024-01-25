#pragma once
#include <fmt/format.h>

#include "config.cuh"
#include "util/util.cuh"

// Non-partitioned Hash Join in Global Memory
// Non-Unique hash table
// Late materialization
// Output payloads to the location of input
// 4byte-4byte table
// hash table do not save key/value, only save row id
// uses bucket chain to solve conflict, bucket size = 1

namespace join {
namespace naive {

/// @breif build hash table for R
/// @param [in]     r_key       all keys of relation R
/// @param [in]     n           number of keys
/// @param [in out] ht_link     hash table links, init -> 0
/// @param [in out] ht_slot     hash table slots, init -> 0, header of links
/// @param [in]     ht_size_log hash table size = 1 << ht_size_log (>= n)
__global__ void build_ht(int32_t* r_key, int32_t r_n, int32_t* ht_link,
                         int32_t* ht_slot, int32_t ht_size_log) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  // TODO: try vector
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < r_n; i += stride) {
    // TODO: try other hash functions
    int32_t hval = r_key[i] & ht_mask;  // keys[i] % ht_size
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
/// @param [in]     o_payload output buffer for matched r's payload
__global__ void probe_ht(int32_t* s_key, int32_t* s_payload, int32_t s_n,
                         int32_t* r_key, int32_t* r_payload, int32_t* ht_link,
                         int32_t* ht_slot, int32_t ht_size_log,
                         int32_t* o_payload) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < s_n; i += stride) {
    int32_t val = s_key[i];
    int32_t hval = val & ht_mask;
    int32_t s_pl = s_payload[i];

    int32_t next = ht_slot[hval];

    while (next) {
      if (val == r_key[next - 1]) {
        int32_t r_pl = r_payload[next - 1];
        o_payload[i] += r_pl + s_pl;  // TODO: aggregation or materialization?
        // break;
      }

      next = ht_link[next - 1];
    }
  }
}

void join(int32_t* r_key, int32_t* r_payload, int32_t r_n, int32_t* s_key,
          int32_t* s_payload, int32_t s_n, int32_t* o_payload, Config cfg) {
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

  int32_t* d_o_payload = nullptr;
  CHKERR(cutil::DeviceAlloc(d_o_payload, s_n));
  CHKERR(cutil::DeviceSet(d_o_payload, 0, s_n));

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
    probe_ht<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_s_key, d_s_payload, s_n, d_r_key, d_r_payload, d_ht_link, d_ht_slot,
        ht_size_log, d_o_payload);
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

  fmt::print(
      "Join Naive (bucket size = 1)\n"
      "[build(R), {} ms, {} tps (S)]\n"
      "[probe(S), {} ms, {} tps (R)]\n",
      ms_build, r_n * 1.0 / ms_build * 1000, ms_probe,
      s_n * 1.0 / ms_probe * 1000);

  CHKERR(cutil::CpyDeviceToHost(o_payload, d_o_payload, s_n));
  return;
}
}  // namespace naive
}  // namespace join