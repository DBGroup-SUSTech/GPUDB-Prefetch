#pragma once
#include <assert.h>
#include <time.h>

#include "classicjoin/common.cuh"
#include "util/util.cuh"
namespace classicjoin {
/// @note Join from https://github.com/TimoKersten/db-engine-paradigms

namespace naive {
/// @brief
/// @param r            R relation
/// @param entries      Pre-allocated hash table entries
/// @param r_n          number of R tuples
/// @param ht_slot      Headers of chains
/// @param ht_size_log  ht size = ht_size_log << 1
/// @return
__global__ void build_ht(Tuple *r, Entry *entries, int r_n,
                         EntryHeader *ht_slot, int ht_size_log) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < r_n; i += stride) {
    Tuple *tuple = &r[i];
    Entry *entry = &entries[i];
    entry->tuple.k = tuple->k;
    entry->tuple.v = tuple->v;

    int hval = tuple->k & ht_mask;
    auto last = gutil::atomic_exch_64(&ht_slot[hval].next, entry);
    entry->header.next = last;
  }
}

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
__global__ void probe_ht(Tuple *s, int s_n, EntryHeader *ht_slot,
                         int ht_size_log, int32_t *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int32_t aggr_local = 0;

  for (size_t i = tid; i < s_n; i += stride) {
    Tuple s_tuple = s[i];  // NO Prefech
    int hval = s_tuple.k & ht_mask;

    Entry *next = ht_slot[hval].next;  // random read

    while (next) {
      Tuple r_tuple = next->tuple;  // random read
      if (r_tuple.k == s_tuple.k) {
        aggr_fn_local(r_tuple.v, s_tuple.v, &aggr_local);
        // break;
      }

      next = next->header.next;  // prefetch
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

__global__ void probe_ht_measure(Tuple *s, int s_n, EntryHeader *ht_slot,
                                 int ht_size_log, int32_t *o_aggr,
                                 clock_t *cycles_hash, clock_t *cycles_slot,
                                 clock_t *cycles_next, clock_t *cycles_match) {
  // TODO
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int32_t aggr_local = 0;
  __shared__ Tuple s_tuple;
  __shared__ int hval;
  __shared__ Entry *next;
  __shared__ Entry entry;

  // clock_t *cycle_hash = cycles_hash[threadIdx.x];
  // clock_t *cycle_slot = cycles_slot[threadIdx.x];
  // clock_t *cycle_next = cycles_next[threadIdx.x];
  // clock_t *cycle_match = cycle_smatch[threadIdx.x];
  clock_t cycle_hash = 0;
  clock_t cycle_slot = 0;
  clock_t cycle_next = 0;
  clock_t cycle_match = 0;

  for (size_t i = tid; i < s_n; i += stride) {
    cycle_hash = clock();
    s_tuple = s[i];
    hval = s_tuple.k & ht_mask;
    // __syncwarp();
    cycle_hash = clock() - cycle_hash;

    cycle_slot = clock();
    next = ht_slot[hval].next;
    // __syncwarp();
    cycle_slot = clock() - cycle_slot;

    while (next) {
      cycle_next = clock();
      entry = *next;
      // __syncwarp();
      cycle_next = clock() - cycle_next;

      cycle_match = clock();
      Tuple r_tuple = entry.tuple;
      if (r_tuple.k == s_tuple.k) {
        aggr_fn_local(r_tuple.v, s_tuple.v, &aggr_local);
      }
      next = next->header.next;  // prefetch
      // __syncwarp();
      cycle_match = clock() - cycle_match;
    }

    printf("%ld, %ld, %ld, %ld\n",  //
           cycle_hash, cycle_slot, cycle_next, cycle_match);
  }
  // cycles_hash[threadIdx.x] = cycle_hash;
  // cycles_slot[threadIdx.x] = cycle_slot;
  // cycles_next[threadIdx.x] = cycle_next;
  // cycles_match[threadIdx.x] = cycle_match;
  aggr_fn_global(aggr_local, o_aggr);
}

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key,
         int32_t *s_payload, int32_t s_n, Config cfg) {
  CHKERR(cudaDeviceReset());

  // Convert to row-format
  Tuple *r = new Tuple[r_n], *d_r = nullptr;
  Tuple *s = new Tuple[s_n], *d_s = nullptr;
  col_to_row(r_key, r_payload, r, r_n);
  col_to_row(s_key, s_payload, s, s_n);

  CHKERR(cutil::DeviceAlloc(d_r, r_n));
  CHKERR(cutil::DeviceAlloc(d_s, s_n));
  CHKERR(cutil::CpyHostToDevice(d_r, r, r_n));
  CHKERR(cutil::CpyHostToDevice(d_s, s, s_n));

  int ht_size_log = cutil::log2(r_n);
  int ht_size = 1 << ht_size_log;
  assert(ht_size == r_n);

  fmt::print("ht_size = {}, ht_size_log = {}\n", ht_size, ht_size_log);

  EntryHeader *d_ht_slot = nullptr;
  Entry *d_entries = nullptr;
  CHKERR(cutil::DeviceAlloc(d_ht_slot, ht_size));
  CHKERR(cutil::DeviceAlloc(d_entries, ht_size));
  CHKERR(cutil::DeviceSet(d_ht_slot, 0, ht_size));
  CHKERR(cutil::DeviceSet(d_entries, 0, ht_size));

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
  fmt::print(
      "Build: {} blocks * {} threads"
      "Probe: {} blocks * {} threads\n",
      cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
      cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_r, d_entries, r_n, d_ht_slot, ht_size_log);
    CHKERR(
        cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
  }

  {
    CHKERR(
        cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
    probe_ht<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
        d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
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

  int32_t aggr;
  CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));
  return aggr;

  delete[] r;
  delete[] s;
}

int join_measure(int32_t *r_key, int32_t *r_payload, int32_t r_n,
                 int32_t *s_key, int32_t *s_payload, int32_t s_n, Config cfg) {
  CHKERR(cudaDeviceReset());

  // Convert to row-format
  Tuple *r = new Tuple[r_n], *d_r = nullptr;
  Tuple *s = new Tuple[s_n], *d_s = nullptr;
  col_to_row(r_key, r_payload, r, r_n);
  col_to_row(s_key, s_payload, s, s_n);

  CHKERR(cutil::DeviceAlloc(d_r, r_n));
  CHKERR(cutil::DeviceAlloc(d_s, r_n));
  CHKERR(cutil::CpyHostToDevice(d_r, r, r_n));
  CHKERR(cutil::CpyHostToDevice(d_s, s, s_n));

  int ht_size_log = cutil::log2(r_n);
  int ht_size = 1 << ht_size_log;
  assert(ht_size == r_n);

  fmt::print("ht_size = {}, ht_size_log = {}\n", ht_size, ht_size_log);

  EntryHeader *d_ht_slot = nullptr;
  Entry *d_entries = nullptr;
  CHKERR(cutil::DeviceAlloc(d_ht_slot, ht_size));
  CHKERR(cutil::DeviceAlloc(d_entries, ht_size));
  CHKERR(cutil::DeviceSet(d_ht_slot, 0, ht_size));
  CHKERR(cutil::DeviceSet(d_ht_slot, 0, ht_size));

  int32_t *d_aggr;
  CHKERR(cutil::DeviceAlloc(d_aggr, 1));
  CHKERR(cutil::DeviceSet(d_aggr, 0, 1));

  clock_t *d_cycles_hash, *h_cycles_hash = new clock_t[32];
  clock_t *d_cycles_slot, *h_cycles_slot = new clock_t[32];
  clock_t *d_cycles_next, *h_cycles_next = new clock_t[32];
  clock_t *d_cycles_match, *h_cycles_match = new clock_t[32];
  CHKERR(cutil::DeviceAlloc(d_cycles_hash, 32));
  CHKERR(cutil::DeviceAlloc(d_cycles_slot, 32));
  CHKERR(cutil::DeviceAlloc(d_cycles_next, 32));
  CHKERR(cutil::DeviceAlloc(d_cycles_match, 32));

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
  fmt::print(
      "Build: {} blocks * {} threads"
      "Probe: {} blocks * {} threads\n",
      cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
      cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_r, d_entries, r_n, d_ht_slot, ht_size_log);
    CHKERR(
        cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
  }

  {
    CHKERR(
        cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
    // probe_ht<<<cfg.probe_gridsize, cfg.probe_blocksize, 0, stream>>>(
    //     d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    probe_ht_measure<<<1, 1, 0, stream>>>(d_s, 32, d_ht_slot, ht_size_log,
                                          d_aggr, d_cycles_hash, d_cycles_slot,
                                          d_cycles_next, d_cycles_match);
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

  CHKERR(cutil::CpyDeviceToHost(h_cycles_hash, d_cycles_hash, 32));
  CHKERR(cutil::CpyDeviceToHost(h_cycles_slot, d_cycles_slot, 32));
  CHKERR(cutil::CpyDeviceToHost(h_cycles_next, d_cycles_next, 32));
  CHKERR(cutil::CpyDeviceToHost(h_cycles_match, d_cycles_match, 32));
  // for (int i = 0; i < 32; ++i) {
  //   fmt::print("{} - {} - {} - {}\n", h_cycles_hash[i], h_cycles_slot[i],
  //              h_cycles_next[i], h_cycles_match[i]);
  // }

  int32_t aggr;
  CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));
  return aggr;

  delete[] r;
  delete[] s;
}

}  // namespace naive
// TODO
}  // namespace classicjoin