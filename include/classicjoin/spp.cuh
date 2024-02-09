#pragma once

#include "classicjoin/common.cuh"
#include "util/util.cuh"
#include <cstdint>
#include <cstdio>
#include <fmt/core.h>

namespace classicjoin {
namespace spp {

struct ConfigSPP : public Config {
  int method = 4; // one prefetch methods
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
constexpr int PDIST = 8;             // prefetch distance & group size
constexpr int K = 2;                 // the number of code stage (0,1, ...K)
constexpr int D = 4;                 // iteration distance
constexpr int STATE_NUM = K * D + 1; // state number
constexpr int PROBE_NUM = 16;
constexpr int PADDING = 1;             // solve bank conflict
constexpr int THREADS_PER_BLOCK = 128; // threads per block
#define VSMEM(index) v[index * blockDim.x + threadIdx.x]

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

enum class pipline_state_t : int {
  NORMAL = 0,   // pipeline is running without any interruption
  INTERRUPTION, // pipline is interrupted by PAYLOAD
  RENORMAL,     // pipline will be normal in next itr
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

struct prefetch_handler_t {
  struct prefetch_t pref;
  __device__ __forceinline__ void commit(pipline_state_t pipline_state,
                                         void *__restrict__ dst_shared,
                                         const void *__restrict__ src_global,
                                         size_t size_and_align,
                                         size_t zfill = 0UL) {
    if (pipline_state != pipline_state_t::INTERRUPTION) {
      pref.commit(dst_shared, src_global, size_and_align);
    }
    // when pipline_state is INTERRUPTION
    else {
      *(uint64_t *)dst_shared = *(uint64_t *)src_global;
    }
  }
  __device__ __forceinline__ void wait(pipline_state_t pipline_state) {
    if (pipline_state != pipline_state_t::INTERRUPTION) {
      pref.wait();
    }
  }
};

__device__ void print_state(pipline_state_t pipline_state, fsm_t fsm[],
                            int rematch_virtual_s_tuple_id, int hash_virtual_id,
                            int next_virtual_id, int match_virtual_id) {
  // if (threadIdx.x == 0) {
  //   printf("pipline state: %d\t %d\t %d\t %d\t",
  //          static_cast<int>(pipline_state), hash_virtual_id, next_virtual_id,
  //          match_virtual_id);
  //   if (pipline_state == pipline_state_t::INTERRUPTION) {
  //     printf(" rematch_virtual_id:%d", rematch_virtual_s_tuple_id);
  //   }
  //   printf("\n");
  //   for (int i = 0; i < STATE_NUM; i++) {
  //     printf("%d\t", static_cast<int>(fsm[i].state));
  //   }
  //   printf("\n");
  //   for (int i = 0; i < STATE_NUM; i++) {
  //     printf("%d\t", fsm[i].s_tuple.k);
  //   }
  //   printf("\n");
  //   // for (int i = 0; i < STATE_NUM; i++) {
  //   //   printf("%p\t\t",(void *)fsm[i].next);
  //   // }
  //   // printf("\n");
  // }
}

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

  fsm_t fsm[STATE_NUM]{};
  pipline_state_t pipline_state = pipline_state_t::NORMAL;
  prefetch_handler_t pref_handler{};

  int hash_virtual_id = 0;
  int next_virtual_id = -D;
  int match_virtual_id = -2 * D;

  int rematch_virtual_s_tuple_id = -1;
  int finish_num = 0;

  int aggr_local = 0;

  int idx = -1;

  while (finish_num != PROBE_NUM) {
    // hash stage:
    idx = hash_virtual_id % STATE_NUM;
    if (pipline_state != pipline_state_t::INTERRUPTION) {
      ++hash_virtual_id;
      if (hash_virtual_id <= PROBE_NUM) {
        fsm[idx].state = state_t::HASH;
        fsm[idx].s_tuple = s[tid + stride * (hash_virtual_id - 1)];
        int hval = fsm[idx].s_tuple.k & ht_mask;
        pref_handler.commit(pipline_state, &VSMEM(idx), &ht_slot[hval],
                            sizeof(void *));
        fsm[idx].state = state_t::NEXT;
      }
      pipline_state = pipline_state_t::NORMAL;
    }
    print_state(pipline_state, fsm, rematch_virtual_s_tuple_id, hash_virtual_id,
                next_virtual_id, match_virtual_id);

    // next stage
    if (pipline_state == pipline_state_t::NORMAL) {
      idx = next_virtual_id % STATE_NUM;
      ++next_virtual_id;
    } else {
      idx = rematch_virtual_s_tuple_id % STATE_NUM;
    }
    if (pipline_state == pipline_state_t::INTERRUPTION ||
        (pipline_state == pipline_state_t::NORMAL &&
         (next_virtual_id <= PROBE_NUM) && next_virtual_id >= 1)) {
      pref_handler.wait(pipline_state);
      fsm[idx].next = reinterpret_cast<Entry *>(VSMEM(idx));
      if (fsm[idx].next) {
        fsm[idx].state = state_t::MATCH;
        pref_handler.commit(pipline_state, &VSMEM(idx), &(fsm[idx].next->tuple),
                            sizeof(Tuple));

      } else {
        ++finish_num;
        fsm[idx].state = state_t::DONE;
        if (pipline_state == pipline_state_t::INTERRUPTION) {
          pipline_state = pipline_state_t::RENORMAL;
        }
      }
    }
    print_state(pipline_state, fsm, rematch_virtual_s_tuple_id, hash_virtual_id,
                next_virtual_id, match_virtual_id);

    // match stage
    if (pipline_state != pipline_state_t::RENORMAL) {
      if (pipline_state == pipline_state_t::NORMAL) {
        idx = match_virtual_id % STATE_NUM;
        ++match_virtual_id;
      } else {
        idx = rematch_virtual_s_tuple_id % STATE_NUM;
      }
      if (pipline_state == pipline_state_t::INTERRUPTION ||
          (pipline_state == pipline_state_t::NORMAL &&
           (match_virtual_id <= PROBE_NUM) && match_virtual_id >= 1)) {
        if (fsm[idx].state == state_t::MATCH) {
          pref_handler.wait(pipline_state);
          Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(idx));
          if (r_tuple->k == fsm[idx].s_tuple.k) {
            if (pipline_state == pipline_state_t::NORMAL) {
              rematch_virtual_s_tuple_id = match_virtual_id - 1;
            }
            pipline_state = pipline_state_t::INTERRUPTION;
            aggr_fn_local(r_tuple->v, fsm[idx].s_tuple.v, &aggr_local);
          }
          pref_handler.commit(pipline_state, &VSMEM(idx),
                              &fsm[idx].next->header, sizeof(void *));
          fsm[idx].state = state_t::NEXT;
        }
      }
    }
    print_state(pipline_state, fsm, rematch_virtual_s_tuple_id, hash_virtual_id,
                next_virtual_id, match_virtual_id);
  }
  aggr_fn_global(aggr_local, o_aggr);
}

__global__ void probe_ht_3(Tuple *s, int s_n, EntryHeader *ht_slot,
                           int ht_size_log, int *o_aggr);

__global__ void probe_ht_1_smem(Tuple *s, int s_n, EntryHeader *ht_slot,
                                int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ uint64_t v[]; // prefetch buffer

  __shared__ fsm_shared_t fsm[STATE_NUM];
  pipline_state_t pipline_state = pipline_state_t::NORMAL;
  prefetch_handler_t pref_handler{};

  int hash_virtual_id = 0;
  int next_virtual_id = -D;
  int match_virtual_id = -2 * D;

  int rematch_virtual_s_tuple_id = -1;
  int finish_num = 0;

  int aggr_local = 0;

  int idx = -1;

  while (finish_num != PROBE_NUM) {
    // hash stage:
    idx = hash_virtual_id % STATE_NUM;
    if (pipline_state != pipline_state_t::INTERRUPTION) {
      ++hash_virtual_id;
      if (hash_virtual_id <= PROBE_NUM) {
        fsm[idx].state[threadIdx.x] = state_t::HASH;
        fsm[idx].s_tuple[threadIdx.x] = s[tid + stride * (hash_virtual_id - 1)];
        int hval = fsm[idx].s_tuple[threadIdx.x].k & ht_mask;
        pref_handler.commit(pipline_state, &VSMEM(idx), &ht_slot[hval],
                            sizeof(void *));
        fsm[idx].state[threadIdx.x] = state_t::NEXT;
      }
      pipline_state = pipline_state_t::NORMAL;
    }

    // next stage
    if (pipline_state == pipline_state_t::NORMAL) {
      idx = next_virtual_id % STATE_NUM;
      ++next_virtual_id;
    } else {
      idx = rematch_virtual_s_tuple_id % STATE_NUM;
    }
    if (pipline_state == pipline_state_t::INTERRUPTION ||
        (pipline_state == pipline_state_t::NORMAL &&
         (next_virtual_id <= PROBE_NUM) && next_virtual_id >= 1)) {
      pref_handler.wait(pipline_state);
      fsm[idx].next[threadIdx.x] = reinterpret_cast<Entry *>(VSMEM(idx));
      if (fsm[idx].next[threadIdx.x]) {
        fsm[idx].state[threadIdx.x] = state_t::MATCH;
        pref_handler.commit(pipline_state, &VSMEM(idx), &(fsm[idx].next[threadIdx.x]->tuple),
                            sizeof(Tuple));

      } else {
        ++finish_num;
        fsm[idx].state[threadIdx.x] = state_t::DONE;
        if (pipline_state == pipline_state_t::INTERRUPTION) {
          pipline_state = pipline_state_t::RENORMAL;
        }
      }
    }

    // match stage
    if (pipline_state != pipline_state_t::RENORMAL) {
      if (pipline_state == pipline_state_t::NORMAL) {
        idx = match_virtual_id % STATE_NUM;
        ++match_virtual_id;
      } else {
        idx = rematch_virtual_s_tuple_id % STATE_NUM;
      }
      if (pipline_state == pipline_state_t::INTERRUPTION ||
          (pipline_state == pipline_state_t::NORMAL &&
           (match_virtual_id <= PROBE_NUM) && match_virtual_id >= 1)) {
        if (fsm[idx].state[threadIdx.x] == state_t::MATCH) {
          pref_handler.wait(pipline_state);
          Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(idx));
          if (r_tuple->k == fsm[idx].s_tuple[threadIdx.x].k) {
            if (pipline_state == pipline_state_t::NORMAL) {
              rematch_virtual_s_tuple_id = match_virtual_id - 1;
            }
            pipline_state = pipline_state_t::INTERRUPTION;
            aggr_fn_local(r_tuple->v, fsm[idx].s_tuple[threadIdx.x].v, &aggr_local);
          }
          pref_handler.commit(pipline_state, &VSMEM(idx),
                              &fsm[idx].next[threadIdx.x]->header, sizeof(void *));
          fsm[idx].state[threadIdx.x] = state_t::NEXT;
        }
      }
    }
  }
  aggr_fn_global(aggr_local, o_aggr);
}

// __global__ void print_ht_kernel(EntryHeader *ht_slot, int n) {
//   for (int i = 0; i < n; ++i) {
//     printf("%d: %p\n", i, ht_slot[i].next);
//   }
// }
// void print_ht(EntryHeader *d_ht_slot, int n) {
//   print_ht_kernel<<<1, 1>>>(d_ht_slot, n);
//   CHKERR(cudaDeviceSynchronize());
// }

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key,
         int32_t *s_payload, int32_t s_n, ConfigSPP cfg) {
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
      const int smeme_size = STATE_NUM * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                   stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 3) {
      assert(0);
      const int smeme_size = STATE_NUM * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_3<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                   stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 4) {
      const int smeme_size = STATE_NUM * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe_ht_1_smem<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
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

  fmt::print("Join SPP (bucket size = 1)\n"
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

} // namespace spp

} // namespace classicjoin