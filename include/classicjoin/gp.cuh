#pragma once

#include "classicjoin/common.cuh"
#include "classicjoin/config.cuh"
#include "util/util.cuh"
#include <cstring>

namespace classicjoin {
namespace gp {

struct ConfigGP : public Config {
  int method = 2; // one prefetch methods
};

// TODO: use prefetch in build_ht
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

// for prefetch  ---------------------------------------------------------
constexpr int PDIST = PDIST_CONFIG::PDIST; // prefetch distance & group size
constexpr int PADDING = 1;                 // solve bank conflict
constexpr int THREADS_PER_BLOCK = 128;     // threads per block
#define VSMEM_1(index) v[index * blockDim.x + threadIdx.x]
#define VSMEM_2(index, offset)                                                 \
  v[2 * (index * blockDim.x + threadIdx.x) + offset]

// TODO: fsm_shared
// TODO: compare 3 methods
// 1. three status, prefetch entry.tuple, then header
// 2. two status, prefetch the whole entry
// 3. two status, prefetch entryheader, directly access body
enum class state_t : int {
  HASH = 0,  // get new tuple, prefetch Next
  NEXT = 1,  // get Next*, prefetch Entry.tuple
  MATCH = 2, // get Entry.tuple, prefetch Entry.Header

  DONE = 4
};

struct fsm_t {
  Tuple s_tuple; // 8 Byte
  Entry *next;   // 8 Byte
  state_t state; // 4 Byte
};

struct fsm_shared_t {
  Tuple s_tuple[THREADS_PER_BLOCK];
  Entry *next[THREADS_PER_BLOCK];
  state_t state[THREADS_PER_BLOCK];
};

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
__launch_bounds__(128, 1) //
    __global__ void probe_ht_1(Tuple *s, int s_n, EntryHeader *ht_slot,
                               int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ uint64_t v[]; // prefetch buffer

  fsm_t fsm[PDIST]{};
  prefetch_t pref{};

  int aggr_local = 0;

  for (int i = tid; i < s_n; i += stride * PDIST) {
    int finish_match_num = 0;
    for (int j = 0; j < PDIST; j++) {
      int s_tuple_id = i + j * stride;
      if (s_tuple_id < s_n) {
        fsm[j].state = state_t::HASH;
        fsm[j].s_tuple = s[s_tuple_id];
        // hash
        int hval = fsm[j].s_tuple.k & ht_mask;
        // prefetch
        pref.commit(&VSMEM_1(j), &ht_slot[hval], sizeof(void *));
        fsm[j].state = state_t::NEXT;
      } else {
        finish_match_num++;
      }
    }
    while (finish_match_num != PDIST) {
#pragma unroll
      for (int j = 0; j < PDIST; j++) {
        if (fsm[j].state == state_t::NEXT) {
          pref.wait();
          fsm[j].next = reinterpret_cast<Entry *>(VSMEM_1(j));
          if (fsm[j].next) {
            pref.commit(&VSMEM_1(j), &(fsm[j].next->tuple), sizeof(Tuple));
            fsm[j].state = state_t::MATCH;
          } else {
            finish_match_num++;
            fsm[j].state = state_t::DONE;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < PDIST; j++) {
        if (fsm[j].state == state_t::MATCH) {
          pref.wait();
          Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM_1(j));
          if (r_tuple->k == fsm[j].s_tuple.k) {
            aggr_fn_local(r_tuple->v, fsm[j].s_tuple.v, &aggr_local);
          }
          fsm[j].state = state_t::NEXT;
          pref.commit(&VSMEM_1(j), &(fsm[j].next->header), sizeof(void *));
        }
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
__launch_bounds__(128, 1) //
    __global__
    void probe_ht_2_registers(Tuple *s, int s_n, EntryHeader *ht_slot,
                              int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ uint64_t v[]; // prefetch buffer

  fsm_t fsm[PDIST]{};
  prefetch_t pref{};

  bool finish_match = true;
  int aggr_local = 0;

  for (int i = tid; i < s_n; i += stride * PDIST) {
#pragma unroll
    for (int j = 0; j < PDIST; j++) {
      int s_tuple_id = i + j * stride;
      if (s_tuple_id < s_n) {
        fsm[j].state = state_t::HASH;
        fsm[j].s_tuple = s[s_tuple_id];
        // hash
        int hval = fsm[j].s_tuple.k & ht_mask;
        // prefetch
        pref.commit(&VSMEM_2(j, 1), &ht_slot[hval], 8);
        fsm[j].state = state_t::NEXT;
      } else {
        fsm[j].state = state_t::DONE;
      }
    }
#pragma unroll
    for (int j = 0; j < PDIST; j++) {
      if (fsm[j].state == state_t::NEXT) {
        pref.wait();
        fsm[j].next = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
        if (!fsm[j].next) {
          fsm[j].state = state_t::DONE;
          continue;
        }
        pref.commit(&VSMEM_2(j, 0), &(fsm[j].next->tuple), 16);
        fsm[j].state = state_t::MATCH;
      }
    }
#pragma unroll
    for (int j = 0; j < PDIST; j++) {
      if (fsm[j].state == state_t::MATCH) {
        finish_match = false;
        pref.wait();
        while (!finish_match) {
          Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM_2(j, 0));
          fsm[j].next = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
          if (r_tuple->k == fsm[j].s_tuple.k) {
            aggr_fn_local(r_tuple->v, fsm[j].s_tuple.v, &aggr_local);
          }
          if (fsm[j].next) {
            memcpy(&VSMEM_2(j, 0), &(fsm[j].next->tuple), 16);
          } else {
            finish_match = true;
            fsm[j].state = state_t::DONE;
          }
        }
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

__device__ __forceinline__ void handle_interruption(Entry *next,
                                                    const int &s_tuple_k,
                                                    const int &s_tuple_v,
                                                    int &aggr_local) {
  while (next) {
    Tuple r_tuple = next->tuple;
    if (r_tuple.k == s_tuple_k) {
      aggr_local += r_tuple.v * s_tuple_v;
    }
    next = next->header.next;
  }
}

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
__launch_bounds__(128, 1) //
    __global__
    void probe_ht_3_registers(Tuple *s, int s_n, EntryHeader *ht_slot,
                              int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ uint64_t v[]; // prefetch buffer

  fsm_t fsm[PDIST]{};
  prefetch_t pref{};

  int aggr_local = 0;

  for (int i = tid; i < s_n; i += stride * PDIST) {
    for (int j = 0; j < PDIST; j++) {
      int s_tuple_id = i + j * stride;
      if (s_tuple_id < s_n) {
        fsm[j].state = state_t::HASH;
        fsm[j].s_tuple = s[s_tuple_id];
        // hash
        int hval = fsm[j].s_tuple.k & ht_mask;
        // prefetch
        pref.commit(&VSMEM_2(j, 1), &ht_slot[hval], 8);
        fsm[j].state = state_t::NEXT;
      } else {
        fsm[j].state = state_t::DONE;
      }
    }
#pragma unroll
    for (int j = 0; j < PDIST; j++) {
      if (fsm[j].state == state_t::NEXT) {
        pref.wait();
        fsm[j].next = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
        if (!fsm[j].next) {
          fsm[j].state = state_t::DONE;
          continue;
        }
        pref.commit(&VSMEM_2(j, 0), &(fsm[j].next->tuple), 16);
        fsm[j].state = state_t::MATCH;
      }
    }
#pragma unroll
    for (int j = 0; j < PDIST; j++) {
      if (fsm[j].state == state_t::MATCH) {
        pref.wait();
        Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM_2(j, 0));
        fsm[j].next = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
        if (r_tuple->k == fsm[j].s_tuple.k) {
          aggr_fn_local(r_tuple->v, fsm[j].s_tuple.v, &aggr_local);
        }
        if (fsm[j].next) {
          handle_interruption(fsm[j].next, fsm[j].s_tuple.k, fsm[j].s_tuple.v,
                              aggr_local);
        }
        fsm[j].state = state_t::DONE;
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
__launch_bounds__(128, 1) //
    __global__ void probe_ht_2_smem(Tuple *s, int s_n, EntryHeader *ht_slot,
                                    int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ uint64_t v[]; // prefetch buffer
  extern __shared__ fsm_shared_t fsm[];

  prefetch_t pref{};

  bool first_get_next = true;
  int aggr_local = 0;

  for (int i = tid; i < s_n; i += stride * PDIST) {
    int finish_match_num = 0;
    first_get_next = true;
    for (int j = 0; j < PDIST; j++) {
      int s_tuple_id = i + j * stride;
      if (s_tuple_id < s_n) {
        fsm[j].state[threadIdx.x] = state_t::HASH;
        fsm[j].s_tuple[threadIdx.x] = s[s_tuple_id];
        // hash
        int hval = fsm[j].s_tuple[threadIdx.x].k & ht_mask;
        // prefetch
        pref.commit(&VSMEM_2(j, 1), &ht_slot[hval], 8);
        fsm[j].state[threadIdx.x] = state_t::NEXT;
      } else {
        finish_match_num++;
      }
    }
    while (finish_match_num != PDIST) {
#pragma unroll
      for (int j = 0; j < PDIST; j++) {
        if (fsm[j].state[threadIdx.x] == state_t::NEXT) {
          if (first_get_next) {
            pref.wait();
            fsm[j].next[threadIdx.x] = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
            if (!fsm[j].next[threadIdx.x]) {
              finish_match_num++;
              fsm[j].state[threadIdx.x] = state_t::DONE;
              continue;
            }
          }
          pref.commit(&VSMEM_2(j, 0), &(fsm[j].next[threadIdx.x]->tuple), 16);
          fsm[j].state[threadIdx.x] = state_t::MATCH;
        }
      }
      first_get_next = false;
#pragma unroll
      for (int j = 0; j < PDIST; j++) {
        if (fsm[j].state[threadIdx.x] == state_t::MATCH) {
          pref.wait();
          Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM_2(j, 0));
          fsm[j].next[threadIdx.x] = reinterpret_cast<Entry *>(VSMEM_2(j, 1));
          if (r_tuple->k == fsm[j].s_tuple[threadIdx.x].k) {
            aggr_fn_local(r_tuple->v, fsm[j].s_tuple[threadIdx.x].v,
                          &aggr_local);
          }
          if (fsm[j].next[threadIdx.x]) {
            fsm[j].state[threadIdx.x] = state_t::NEXT;
          } else {
            finish_match_num++;
            fsm[j].state[threadIdx.x] = state_t::DONE;
          }
        }
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key,
         int32_t *s_payload, int32_t s_n, ConfigGP cfg) {
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
  fmt::print("use method {}\n"
             "Build: {} blocks * {} threads\n"
             "Probe: {} blocks * {} threads\n",
             cfg.method, cfg.build_gridsize, cfg.build_blocksize,
             cfg.probe_gridsize, cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_r, d_entries, r_n, d_ht_slot, ht_size_log);
    CHKERR(
        cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
  }

  // print_ht_kernel<<<1, 1, 0, stream>>>(d_ht_slot, ht_size);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
    if (cfg.method == 1) {
      const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                   stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 2) {
      const int smeme_size = PDIST * cfg.probe_blocksize * 2 * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_2_registers<<<cfg.probe_gridsize, cfg.probe_blocksize,
                             smeme_size, stream>>>(d_s, s_n, d_ht_slot,
                                                   ht_size_log, d_aggr);

    } else if (cfg.method == 3) {
      const int smeme_size = PDIST * cfg.probe_blocksize * 2 * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_3_registers<<<cfg.probe_gridsize, cfg.probe_blocksize,
                             smeme_size, stream>>>(d_s, s_n, d_ht_slot,
                                                   ht_size_log, d_aggr);

    } else if (cfg.method == 4) {
      const int smeme_size =
          PDIST * cfg.probe_blocksize * 2 * sizeof(uint64_t) +
          THREADS_PER_BLOCK * PDIST;
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_2_smem<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                        stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else {
      assert(0);
    }
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

  int32_t aggr;
  CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));

  delete[] r;
  delete[] s;
  return aggr;
}

} // namespace gp

} // namespace classicjoin