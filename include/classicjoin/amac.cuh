#pragma once

#include "classicjoin/common.cuh"
#include "util/util.cuh"

namespace classicjoin {
namespace amac {

struct ConfigAMAC : public Config {
  int method = 4;  // three prefetch methods
};

// for prefetch  ---------------------------------------------------------
constexpr int PDIST = 8;                // prefetch distance & group size
constexpr int THREADS_PER_BLOCK = 128;  // warps per thread
#define VSMEM(index) v[index * blockDim.x + threadIdx.x]

namespace build {

enum class state_t : int {
  HASH = 0,
  INSERT = 1,
  DONE = 4,
};

struct fsm_shared_t {
  Tuple r_tuple[THREADS_PER_BLOCK];
  Entry *entry[THREADS_PER_BLOCK];
  state_t state[THREADS_PER_BLOCK];
};

/// @brief
/// @param r            R relation
/// @param entries      Pre-allocated hash table entries
/// @param r_n          number of R tuples
/// @param ht_slot      Headers of chains
/// @param ht_size_log  ht size = ht_size_log << 1
/// @return
__global__ void build_ht_naive(Tuple *r, Entry *entries, int r_n,
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

__global__ void build_ht_prefetch(Tuple *r, Entry *entries, int r_n,
                                  EntryHeader *ht_slot, int ht_size_log) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  assert(blockDim.x == THREADS_PER_BLOCK);

  __shared__ fsm_shared_t fsm[PDIST];
  for (int k = 0; k < PDIST; ++k) {
    fsm[k].state[threadIdx.x] = state_t::HASH;
  }

  extern __shared__ uint64_t v[];

  prefetch_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    switch (fsm[k].state[threadIdx.x]) {
      case state_t::HASH: {
        if (i < r_n) {
          fsm[k].r_tuple[threadIdx.x] = r[i];
          fsm[k].entry[threadIdx.x] = &entries[i];
          i += stride;
          int hval = fsm[k].r_tuple[threadIdx.x].k & ht_mask;
          pref.commit(&VSMEM(k), &ht_slot[hval], 8);
          fsm[k].state[threadIdx.x] = state_t::INSERT;
        } else {
          fsm[k].state[threadIdx.x] = state_t::DONE;
          ++all_done;
        }
        break;
      }

      case state_t::INSERT: {
        pref.wait();
        auto entry = fsm[k].entry[threadIdx.x];
        int hval = fsm[k].r_tuple[threadIdx.x].k & ht_mask;
        auto last = gutil::atomic_exch_64(&ht_slot[hval].next, entry);
        entry->tuple = fsm[k].r_tuple[threadIdx.x];
        entry->header.next = last;
        fsm[k].state[threadIdx.x] = build::state_t::HASH;
        --k;  // TODO
        break;
      }

      default:
        break;
    }
    ++k;
  }
}

}  // namespace build

namespace probe {

// TODO: fsm_shared
// TODO: compare 3 methods
// 1. three status, prefetch entry.tuple, then header
// 2. two status, prefetch the whole entry
// 3. two status, prefetch entryheader, directly access body
enum class state_t : int {
  HASH = 0,   // get new tuple, prefetch Next
  NEXT = 1,   // get Next*, prefetch Entry.tuple
  MATCH = 2,  // get Entry.tuple, prefetch Entry.Header

  DONE = 4
};

struct fsm_t {
  Tuple s_tuple;  // 8 Byte
  Entry *next;    // 8 Byte
  state_t state;  // 4 Byte
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
__launch_bounds__(128, 1)  //
    __global__ void probe_ht_1(Tuple *s, int s_n, EntryHeader *ht_slot,
                               int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  extern __shared__ uint64_t v[];  // prefetch buffer

  fsm_t fsm[PDIST]{};
  prefetch_t pref{};
  int all_done = 0, k = 0;

  int aggr_local = 0;

  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);

    switch (fsm[k].state) {
      case state_t::HASH: {
        if (i < s_n) {
          fsm[k].state = state_t::NEXT;

          fsm[k].s_tuple = s[i];
          i += stride;
          int hval = fsm[k].s_tuple.k & ht_mask;
          // pref.commit(&VSMEM(k), &(ht_slot[hval].next), 8);
          pref.commit(&VSMEM(k), &ht_slot[hval], 8);

        } else {
          fsm[k].state = state_t::DONE;
          ++all_done;
        }

        break;
      }

      case state_t::NEXT: {
        pref.wait();
        fsm[k].next = reinterpret_cast<Entry *>(VSMEM(k));
        if (fsm[k].next) {
          fsm[k].state = state_t::MATCH;
          pref.commit(&VSMEM(k), &(fsm[k].next->tuple), 8);

        } else {
          fsm[k].state = state_t::HASH;
          --k;  // fill it with new item, or the bandwidth may be underutilized
        }

        break;
      }

      case state_t::MATCH: {
        pref.wait();
        Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(k));

        if (r_tuple->k == fsm[k].s_tuple.k) {
          aggr_fn_local(r_tuple->v, fsm[k].s_tuple.v, &aggr_local);
        }

        fsm[k].state = state_t::NEXT;
        pref.commit(&VSMEM(k), &(fsm[k].next->header), 8);

        break;
      }
    }
    ++k;
  }

  aggr_fn_global(aggr_local, o_aggr);
}

__launch_bounds__(128, 1)  //
    __global__ void probe_ht_1_smem(Tuple *s, int s_n, EntryHeader *ht_slot,
                                    int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  extern __shared__ uint64_t v[];  // prefetch buffer

  __shared__ fsm_shared_t fsm[PDIST];
  for (int k = 0; k < PDIST; ++k) fsm[k].state[threadIdx.x] = state_t::HASH;

  prefetch_t pref{};
  int all_done = 0, k = 0;

  int aggr_local = 0;

  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);

    switch (fsm[k].state[threadIdx.x]) {
      case state_t::HASH: {
        if (i < s_n) {
          fsm[k].state[threadIdx.x] = state_t::NEXT;

          fsm[k].s_tuple[threadIdx.x] = s[i];
          i += stride;
          int hval = fsm[k].s_tuple[threadIdx.x].k & ht_mask;
          // pref.commit(&VSMEM(k), &(ht_slot[hval].next), 8);
          pref.commit(&VSMEM(k), &ht_slot[hval], 8);

        } else {
          fsm[k].state[threadIdx.x] = state_t::DONE;
          ++all_done;
        }

        break;
      }

      case state_t::NEXT: {
        pref.wait();
        fsm[k].next[threadIdx.x] = reinterpret_cast<Entry *>(VSMEM(k));
        if (fsm[k].next[threadIdx.x]) {
          fsm[k].state[threadIdx.x] = state_t::MATCH;
          pref.commit(&VSMEM(k), &(fsm[k].next[threadIdx.x]->tuple), 8);

        } else {
          fsm[k].state[threadIdx.x] = state_t::HASH;
          --k;  // fill it with new item, or the bandwidth may be underutilized
        }

        break;
      }

      case state_t::MATCH: {
        pref.wait();
        Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(k));

        if (r_tuple->k == fsm[k].s_tuple[threadIdx.x].k) {
          aggr_fn_local(r_tuple->v, fsm[k].s_tuple[threadIdx.x].v, &aggr_local);
        }

        fsm[k].state[threadIdx.x] = state_t::NEXT;
        pref.commit(&VSMEM(k), &(fsm[k].next[threadIdx.x]->header), 8);

        break;
      }
    }
    ++k;
  }

  aggr_fn_global(aggr_local, o_aggr);
}

__global__ void probe_ht_2(Tuple *s, int s_n, EntryHeader *ht_slot,
                           int ht_size_log, int *o_aggr);

__global__ void probe_ht_3(Tuple *s, int s_n, EntryHeader *ht_slot,
                           int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  extern __shared__ uint64_t v[];  // prefetch buffer

  fsm_t fsm[PDIST]{};
  prefetch_t pref{};
  int all_done = 0, k = 0;

  int aggr_local = 0;

  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);

    switch (fsm[k].state) {
      case state_t::HASH: {
        if (i < s_n) {
          fsm[k].state = state_t::NEXT;

          fsm[k].s_tuple = s[i];
          i += stride;
          int hval = fsm[k].s_tuple.k & ht_mask;
          // pref.commit(&VSMEM(k), &(ht_slot[hval].next), 8);
          pref.commit(&VSMEM(k), &ht_slot[hval], 8);

        } else {
          fsm[k].state = state_t::DONE;
          ++all_done;
        }

        break;
      }

      case state_t::NEXT: {
        pref.wait();
        fsm[k].next = reinterpret_cast<Entry *>(VSMEM(k));
        if (fsm[k].next) {
          fsm[k].state = state_t::MATCH;
          pref.commit(&VSMEM(k), &(fsm[k].next->tuple), 8);

        } else {
          fsm[k].state = state_t::HASH;
          --k;  // fill it with new item, or the bandwidth may be underutilized
        }

        break;
      }

      case state_t::MATCH: {
        pref.wait();
        Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(k));

        if (r_tuple->k == fsm[k].s_tuple.k) {
          aggr_fn_local(r_tuple->v, fsm[k].s_tuple.v, &aggr_local);
        }

        // DO NOT PREFTECH
        fsm[k].next = fsm[k].next->header.next;  // should already in cache

        if (fsm[k].next) {
          fsm[k].state = state_t::MATCH;
          pref.commit(&VSMEM(k), &(fsm[k].next->tuple), 8);
        } else {
          fsm[k].state = state_t::HASH;
          --k;
        }

        // fsm[k].state = state_t::NEXT;
        // pref.commit(&VSMEM(k), &(fsm[k].next->header), 8);

        break;
      }
    }
    ++k;
  }

  aggr_fn_global(aggr_local, o_aggr);
}

}  // namespace probe

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
         int32_t *s_payload, int32_t s_n, ConfigAMAC cfg) {
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
      "use method {}\n"
      "Build: {} blocks * {} threads\n"
      "Probe: {} blocks * {} threads\n",
      cfg.method, cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize,
      cfg.probe_blocksize);

  {
    CHKERR(
        cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
    build::
        build_ht_naive<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
            d_r, d_entries, r_n, d_ht_slot, ht_size_log);
    // const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(uint64_t);
    // build::build_ht_prefetch<<<cfg.build_gridsize, cfg.build_blocksize,
    // smeme_size,
    //                     stream>>>(d_r, d_entries, r_n, d_ht_slot,
    //                     ht_size_log);

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
      probe::probe_ht_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                          stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 3) {
      const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe::probe_ht_3<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                          stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 4) {
      const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(uint64_t);
      fmt::print("smem_size = {}\n", smeme_size);
      probe::probe_ht_1_smem<<<cfg.probe_gridsize, cfg.probe_blocksize,
                               smeme_size, stream>>>(d_s, s_n, d_ht_slot,
                                                     ht_size_log, d_aggr);
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

  fmt::print(
      "Join AMAC (bucket size = 1)\n"
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

}  // namespace amac

}  // namespace classicjoin