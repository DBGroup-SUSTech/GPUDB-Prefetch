#pragma once
#include <cooperative_groups.h>
#include <cstdint>

#include "classicjoin/common.cuh"
#include "classicjoin/config.cuh"
#include "util/util.cuh"

namespace cg = cooperative_groups;

namespace classicjoin {
namespace imv {

struct ConfigIMV : public Config {
  int method = 2; // all lane has its own states in shared memory
};

// for prefetch  ---------------------------------------------------------
constexpr int PDIST = PDIST_CONFIG::PDIST; // prefetch distance & group size
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // warps per thread
#define VSMEM(index) v[index * blockDim.x + threadIdx.x]
constexpr unsigned MASK_ALL_LANES = 0xFFFFFFFF;

namespace build {
enum class state_t : int {
  HASH = 1,
  INSERT = 2,
  DONE = 4,
};

// struct fsm_t {
//   Tuple s_tuple;  // 8 Byte
//   Entry *next;    // 8 Byte
//   state_t state;  // 4 Byte
// };

// struct fsm_shared_t {
//   // lane private states
//   Tuple s_tuple[THREADS_PER_BLOCK];
//   Entry *next[THREADS_PER_BLOCK];
//   // warp private states
//   state_t state[THREADS_PER_BLOCK];
//   bool active[THREADS_PER_BLOCK];  // TODO: buffering it in register
// };
struct fsm_shared_t {
  Tuple r_tuple[THREADS_PER_BLOCK];
  Entry *entry[THREADS_PER_BLOCK];
  state_t state[THREADS_PER_BLOCK];
  bool active[THREADS_PER_BLOCK];
}; // namespace build

// TODO: use prefetch in build_ht
/// @brief
/// @param r            R relation
/// @param entries      Pre-allocated hash table entries
/// @param r_n          number of R tuples
/// @param ht_slot      Headers of chains
/// @param ht_size_log  ht size = ht_size_log << 1
/// @return
__global__ void build_ht_prefetch(Tuple *r, Entry *entries, int r_n,
                                  EntryHeader *ht_slot, int ht_size_log) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  assert(blockDim.x == THREADS_PER_BLOCK);

  __shared__ build::fsm_shared_t fsm[PDIST];
  for (int k = 0; k < PDIST; ++k) {
    fsm[k].state[threadIdx.x] = build::state_t::HASH;
  }

  extern __shared__ uint64_t v[]; // prefetch buffer

  prefetch_t pref{};
  int all_done = 0, k = 0;

  while (all_done < PDIST) {
    k = (k == PDIST) ? 0 : k;
    switch (fsm[k].state[threadIdx.x]) {
    case build::state_t::HASH: {
      bool active = (i < r_n);
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);

      if (active_mask) {
        if (active) {
          fsm[k].r_tuple[threadIdx.x] = r[i];
          fsm[k].entry[threadIdx.x] = &entries[i];
          i += stride;
          int hval = fsm[k].r_tuple[threadIdx.x].k & ht_mask;

          pref.commit(&VSMEM(k), &ht_slot[hval], 8);
        }

        __syncwarp();

        fsm[k].state[threadIdx.x] = build::state_t::INSERT;
        fsm[k].active[threadIdx.x] = active;
      } else {
        fsm[k].state[threadIdx.x] = build::state_t::DONE;
        ++all_done;
      }
      break;
    }

    case build::state_t::INSERT: {
      bool active = fsm[k].active[threadIdx.x];

      if (active) {
        pref.wait();
        auto entry = fsm[k].entry[threadIdx.x];
        int hval = fsm[k].r_tuple[threadIdx.x].k & ht_mask;
        auto last = gutil::atomic_exch_64(&ht_slot[hval].next, entry);
        entry->tuple = fsm[k].r_tuple[threadIdx.x];
        entry->header.next = last;
      }

      __syncwarp();
      fsm[k].state[threadIdx.x] = build::state_t::HASH;
      break;
    }
    default:
      break;
    }
    ++k;
  }
}

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
} // namespace build

namespace probe {
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
  bool active;
};

struct fsm_shared_t {
  // lane private states
  Tuple s_tuple[THREADS_PER_BLOCK];
  Entry *next[THREADS_PER_BLOCK];
  // warp private states
  state_t state[THREADS_PER_BLOCK];
  bool active[THREADS_PER_BLOCK]; // TODO: buffering it in register
};

/// @brief
/// @param s           S relation
/// @param s_n         number of S items
/// @param ht_slot     header of chains
/// @param ht_size_log log2(htsize)
/// @param entries     hash table entries
/// @return
// IMV with seperate stages
__global__ void probe_ht_1(Tuple *s, int s_n, EntryHeader *ht_slot,
                           int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  assert(blockDim.x == THREADS_PER_BLOCK);

  cg::thread_block_tile<32> warp =
      cg::tiled_partition<32>(cg::this_thread_block());

  // warp info
  unsigned warpid = threadIdx.x / 32;
  unsigned warplane = threadIdx.x % 32;
  unsigned prefixlanes = 0xffffffff >> (32 - warplane);

  // shared memory data for IMV
  __shared__ probe::fsm_shared_t fsm[PDIST]; // states
  for (int k = 0; k < PDIST; ++k) {
    fsm[k].state[threadIdx.x] = probe::state_t::HASH;
    fsm[k].active[threadIdx.x] = false;
  }
  __shared__ probe::fsm_shared_t rvs; // RVS in IMV paper
  extern __shared__ uint64_t v[];     // prefetch buffer

  int8_t rvs_cnt = 0; // number of active lanes in rvs

  // prefetch primitives from AMAC
  prefetch_t pref{};
  int all_done = 0, k = 0;

  // output
  int aggr_local = 0;

  // fully vectorized loops
  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    warp.sync();
    // assert(__activemask() == MASK_ALL_LANES);
    // transfer states
    switch (fsm[k].state[threadIdx.x]) {
    case probe::state_t::HASH: {
      bool active = (i < s_n);
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);

      if (active_mask) {
        if (active) {
          // printf("k=%d,tid=%d, state::HASH, i = %d, key = %d\n", k, tid, i,
          //        s[i].k);
          fsm[k].s_tuple[threadIdx.x] = s[i];
          i += stride;
          int hval = fsm[k].s_tuple[threadIdx.x].k & ht_mask;

          pref.commit(&VSMEM(k), &ht_slot[hval], 8);
          // pref.commit_k(&VSMEM(k), &ht_slot[hval], 8, k,
          // (int)state_t::HASH);
        }

        warp.sync();

        fsm[k].state[threadIdx.x] = probe::state_t::NEXT;
        fsm[k].active[threadIdx.x] = active;

      } else {
        fsm[k].state[threadIdx.x] = probe::state_t::DONE;
        ++all_done;
      }

      // assert(__activemask() == MASK_ALL_LANES);

      break;
    }

    case probe::state_t::NEXT: {
      bool active = fsm[k].active[threadIdx.x];

      if (active) {
        pref.wait();
        // pref.wait_k(k, (int)state_t::NEXT);
        fsm[k].next[threadIdx.x] = reinterpret_cast<Entry *>(VSMEM(k));
        // assert(((uint64_t)fsm[k].next[threadIdx.x]) % 8 == 0 &&
        //        "Line 147 misaligned");
        active = (fsm[k].next[threadIdx.x] != 0);
        // printf("k=%d,tid=%d, next = %p\n", k, tid,
        // fsm[k].next[threadIdx.x]);
      }

      warp.sync();
      // assert(__activemask() == MASK_ALL_LANES);

      // integration
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);
      int active_cnt = __popc(active_mask);

      // {
      //   int sum = rvs_cnt;
      //   int flag = (sum == __shfl_sync(MASK_ALL_LANES, sum, 0));
      //   int same_mask = __ballot_sync(MASK_ALL_LANES, flag);
      //   assert(same_mask == MASK_ALL_LANES);
      // };

      // TODO: optimize with shfl_sync
      if (active_cnt + rvs_cnt < 32) { // empty
        int prefix_cnt = __popc(active_mask & prefixlanes);
        if (active) {
          int offset = warpid * 32 + rvs_cnt + prefix_cnt;
          rvs.s_tuple[offset] = fsm[k].s_tuple[threadIdx.x];
          rvs.next[offset] = fsm[k].next[threadIdx.x];
        }

        warp.sync();
        // assert(__activemask() == MASK_ALL_LANES);

        rvs_cnt += active_cnt;

        // empty, switch to hash
        fsm[k].state[threadIdx.x] = probe::state_t::HASH;
        fsm[k].active[threadIdx.x] = false;
        --k;

      } else { // full
        int inactive_mask = ~active_mask;
        int prefix_cnt = __popc(inactive_mask & prefixlanes);
        int remain_cnt = rvs_cnt + active_cnt - 32;
        if (!active) {
          int offset = warpid * 32 + remain_cnt + prefix_cnt;
          fsm[k].s_tuple[threadIdx.x] = rvs.s_tuple[offset];
          fsm[k].next[threadIdx.x] = rvs.next[offset];
          // assert(((uint64_t)fsm[k].next[threadIdx.x]) % 8 == 0 &&
          //        "Line 179 misaligned");
        }

        warp.sync();
        // assert(__activemask() == MASK_ALL_LANES);

        rvs_cnt = remain_cnt;
        // printf("k=%d,tid=%d, FULL, rvs_cnt = %d\n", k, tid, rvs_cnt);

        pref.commit(&VSMEM(k), &(fsm[k].next[threadIdx.x]->tuple), 8);
        // pref.commit_k(&VSMEM(k), &(fsm[k].next[threadIdx.x]->tuple), 8, k,
        //               (int)state_t::NEXT);
        // full, switch to match
        fsm[k].state[threadIdx.x] = probe::state_t::MATCH;
        fsm[k].active[threadIdx.x] = true;
      }

      warp.sync();
      // assert(__activemask() == MASK_ALL_LANES);

      break;
    }

    case probe::state_t::MATCH: {
      bool active = fsm[k].active[threadIdx.x];

      if (active) {
        pref.wait();
        // pref.wait_k(k, (int)state_t::MATCH);
        Tuple *r_tuple = reinterpret_cast<Tuple *>(&VSMEM(k));
        // assert(((uint64_t)r_tuple) % 8 == 0 && "Line 197 misaligned");
        Tuple *s_tuple = &fsm[k].s_tuple[threadIdx.x];
        // printf("k=%d,tid=%d, active, r_tuple=%d, s_tuple=%d\n", k, tid,
        //        r_tuple->k, s_tuple->k);
        if (r_tuple->k == s_tuple->k) {
          aggr_fn_local(r_tuple->v, s_tuple->v, &aggr_local);
        }
      }
      warp.sync();
      // assert(__activemask() == MASK_ALL_LANES);

      pref.commit(&VSMEM(k), &(fsm[k].next[threadIdx.x]->header), 8);
      // pref.commit_k(&VSMEM(k), &(fsm[k].next[threadIdx.x]->header), 8, k,
      //               (int)state_t::MATCH);
      fsm[k].state[threadIdx.x] = probe::state_t::NEXT;
      warp.sync();
      // assert(__activemask() == MASK_ALL_LANES);

      break;
    }
    default:
      break;
    }
    ++k;
  }

  /// @note without warp.sync() there will be errors
  warp.sync();
  // assert(__activemask() == MASK_ALL_LANES);

  // handle RVS
  if (warplane < rvs_cnt) {
    Tuple s_tuple = rvs.s_tuple[threadIdx.x];
    Entry *next = rvs.next[threadIdx.x];
    // printf("tid = %d, warp lane = %d, rvs_cnt = %d, next = %p\n",
    // threadIdx.x,
    //        warplane, rvs_cnt, next);

    while (next) {
      assert(next);
      Tuple r_tuple = next->tuple;
      if (r_tuple.k == s_tuple.k) {
        aggr_fn_local(r_tuple.v, s_tuple.v, &aggr_local);
      }
      next = next->header.next;
    }
  }

  aggr_fn_global(aggr_local, o_aggr);
}

__global__ void probe_ht_2stage(Tuple *s, int s_n, EntryHeader *ht_slot,
                                int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  assert(blockDim.x == THREADS_PER_BLOCK);

  cg::thread_block_tile<32> warp =
      cg::tiled_partition<32>(cg::this_thread_block());

  // warp info
  unsigned warpid = threadIdx.x / 32;
  unsigned warplane = threadIdx.x % 32;
  unsigned prefixlanes = 0xffffffff >> (32 - warplane);

  // shared memory data for IMV
  __shared__ probe::fsm_shared_t fsm[PDIST]; // states
  for (int k = 0; k < PDIST; ++k)
    fsm[k].state[threadIdx.x] = probe::state_t::HASH;

  __shared__ probe::fsm_shared_t rvs;            // RVS in IMV paper
  __shared__ Entry v[THREADS_PER_BLOCK * PDIST]; // prefetch buffer

  int8_t rvs_cnt = 0; // number of active lanes in rvs

  // prefetch primitives from AMAC
  prefetch_t pref{};
  int all_done = 0, k = 0;

  // output
  int aggr_local = 0;

  // fully vectorized loops
  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    // warp.sync();

    // transfer states
    state_t state = fsm[k].state[threadIdx.x];
    switch (state) {
    case state_t::HASH: {
      bool active = (i < s_n);
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);

      if (active_mask) {
        if (active) {
          fsm[k].s_tuple[threadIdx.x] = s[i];
          i += stride;
          int hval = fsm[k].s_tuple[threadIdx.x].k & ht_mask;

          pref.commit(&(VSMEM(k).header), &ht_slot[hval], sizeof(EntryHeader));
        }

        // warp.sync();

        fsm[k].state[threadIdx.x] = probe::state_t::NEXT;
        fsm[k].active[threadIdx.x] = active;

      } else {
        fsm[k].state[threadIdx.x] = probe::state_t::DONE;
        ++all_done;
      }

      break;
    }

    case state_t::MATCH:
    case state_t::NEXT: {
      bool active = fsm[k].active[threadIdx.x];

      // MATCH
      if (active) {
        pref.wait();
        Entry *entry = reinterpret_cast<Entry *>(&VSMEM(k));
        if (state == state_t::MATCH) {
          Tuple *r_tuple = &entry->tuple;
          if (r_tuple->k == fsm[k].s_tuple[threadIdx.x].k) {
            aggr_fn_local(r_tuple->v,                    //
                          fsm[k].s_tuple[threadIdx.x].v, //
                          &aggr_local);                  //
          }
        }
        Entry *next = entry->header.next;
        fsm[k].next[threadIdx.x] = next;
        active = (next != 0);
      }

      // integration
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);
      int active_cnt = __popc(active_mask);

      // TODO: optimize with shfl_sync
      if (active_cnt + rvs_cnt < 32) { // empty
        int prefix_cnt = __popc(active_mask & prefixlanes);
        if (active) {
          int offset = warpid * 32 + rvs_cnt + prefix_cnt;
          rvs.s_tuple[offset] = fsm[k].s_tuple[threadIdx.x];
          rvs.next[offset] = fsm[k].next[threadIdx.x];
        }

        // warp.sync();

        rvs_cnt += active_cnt;

        // empty, switch to hash
        fsm[k].state[threadIdx.x] = probe::state_t::HASH;
        fsm[k].active[threadIdx.x] = false;
        --k;

      } else { // full
        int inactive_mask = ~active_mask;
        int prefix_cnt = __popc(inactive_mask & prefixlanes);
        int remain_cnt = rvs_cnt + active_cnt - 32;
        if (!active) {
          int offset = warpid * 32 + remain_cnt + prefix_cnt;
          fsm[k].s_tuple[threadIdx.x] = rvs.s_tuple[offset];
          fsm[k].next[threadIdx.x] = rvs.next[offset];
        }

        // warp.sync();

        rvs_cnt = remain_cnt;

        pref.commit(&VSMEM(k), fsm[k].next[threadIdx.x], sizeof(Entry));
        fsm[k].state[threadIdx.x] = probe::state_t::MATCH;
        fsm[k].active[threadIdx.x] = true;
      }

      // warp.sync();

      break;
    }

    default:
      break;
    }
    ++k;
  }

  /// @note without warp.sync() there will be errors
  warp.sync();

  // handle RVS
  if (warplane < rvs_cnt) {
    Tuple s_tuple = rvs.s_tuple[threadIdx.x];
    Entry *next = rvs.next[threadIdx.x];

    while (next) {
      assert(next);
      Tuple r_tuple = next->tuple;
      if (r_tuple.k == s_tuple.k) {
        aggr_fn_local(r_tuple.v, s_tuple.v, &aggr_local);
      }
      next = next->header.next;
    }
  }

  aggr_fn_global(aggr_local, o_aggr);
}

__global__ void probe_ht_2stage_regs(Tuple *s, int s_n, EntryHeader *ht_slot,
                                     int ht_size_log, int *o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;

  assert(blockDim.x == THREADS_PER_BLOCK);

  cg::thread_block_tile<32> warp =
      cg::tiled_partition<32>(cg::this_thread_block());

  // warp info
  unsigned warpid = threadIdx.x / 32;
  unsigned warplane = threadIdx.x % 32;
  unsigned prefixlanes = 0xffffffff >> (32 - warplane);

  // shared memory data for IMV
  __shared__ probe::fsm_shared_t fsm[PDIST]; // states
  for (int k = 0; k < PDIST; ++k)
    fsm[k].state[threadIdx.x] = probe::state_t::HASH;

  // __shared__ probe::fsm_shared_t rvs; // RVS in IMV paper

  __shared__ Entry v[THREADS_PER_BLOCK * PDIST]; // prefetch buffer

  fsm_t rvs_reg;

  int8_t rvs_cnt = 0; // number of active lanes in rvs

  // prefetch primitives from AMAC
  prefetch_t pref{};
  int all_done = 0, k = 0;

  // output
  int aggr_local = 0;

  // fully vectorized loops
  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    // warp.sync();

    // transfer states
    state_t state = fsm[k].state[threadIdx.x];
    switch (state) {
    case state_t::HASH: {
      bool active = (i < s_n);
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);

      if (active_mask) {
        if (active) {
          fsm[k].s_tuple[threadIdx.x] = s[i];
          i += stride;
          int hval = fsm[k].s_tuple[threadIdx.x].k & ht_mask;

          pref.commit(&(VSMEM(k).header), &ht_slot[hval], sizeof(EntryHeader));
        }

        // warp.sync();

        fsm[k].state[threadIdx.x] = probe::state_t::NEXT;
        fsm[k].active[threadIdx.x] = active;

      } else {
        fsm[k].state[threadIdx.x] = probe::state_t::DONE;
        ++all_done;
      }

      break;
    }

    case state_t::MATCH:
    case state_t::NEXT: {
      bool active = fsm[k].active[threadIdx.x];

      // MATCH
      if (active) {
        pref.wait();
        Entry *entry = reinterpret_cast<Entry *>(&VSMEM(k));
        if (state == state_t::MATCH) {
          Tuple *r_tuple = &entry->tuple;
          if (r_tuple->k == fsm[k].s_tuple[threadIdx.x].k) {
            aggr_fn_local(r_tuple->v,                    //
                          fsm[k].s_tuple[threadIdx.x].v, //
                          &aggr_local);                  //
          }
        }
        Entry *next = entry->header.next;
        fsm[k].next[threadIdx.x] = next;
        active = (next != 0);
      }

      // integration
      int active_mask = __ballot_sync(MASK_ALL_LANES, active);
      int active_cnt = __popc(active_mask);

      // pack two int32_t to a unsigned long
      unsigned long fsm_k_stuple =
          static_cast<unsigned long>(fsm[k].s_tuple[threadIdx.x].k) << 32 |
          fsm[k].s_tuple[threadIdx.x].v;
      Entry *fsm_k_next = fsm[k].next[threadIdx.x];

      // TODO: optimize with shfl_sync
      if (active_cnt + rvs_cnt < 32) { // empty
        int offset = warplane - rvs_cnt;
        int src_lane = __fns(active_mask, 0, offset + 1);

        unsigned long tmp_tuple =
            __shfl_sync(MASK_ALL_LANES, fsm_k_stuple, src_lane);
        Entry *tmp_next = (Entry *)__shfl_sync(
            MASK_ALL_LANES, (unsigned long)fsm_k_next, src_lane);

        if (warplane >= rvs_cnt) {
          rvs_reg.s_tuple.k = static_cast<int>(tmp_tuple >> 32);
          rvs_reg.s_tuple.v = static_cast<int>(tmp_tuple & 0xFFFFFFFF);
          rvs_reg.next = tmp_next;
        }

        rvs_cnt += active_cnt;

        // empty, switch to hash
        fsm[k].state[threadIdx.x] = probe::state_t::HASH;
        fsm[k].active[threadIdx.x] = false;
        --k;

      } else { // full
        int inactive_mask = ~active_mask;
        int prefix_cnt = __popc(inactive_mask & prefixlanes);
        int remain_cnt = rvs_cnt + active_cnt - 32;
        int offset = remain_cnt + prefix_cnt;
        int32_t tmp_s_tuple_k =
            __shfl_sync(MASK_ALL_LANES, rvs_reg.s_tuple.k, offset);
        int32_t tmp_s_tuple_v =
            __shfl_sync(MASK_ALL_LANES, rvs_reg.s_tuple.v, offset);
        Entry *tmp_next = (Entry *)__shfl_sync(
            MASK_ALL_LANES, (unsigned long)rvs_reg.next, offset);
        if (!active) {
          fsm[k].s_tuple[threadIdx.x].k = tmp_s_tuple_k;
          fsm[k].s_tuple[threadIdx.x].v = tmp_s_tuple_v;
          fsm[k].next[threadIdx.x] = tmp_next;
        }
        rvs_cnt = remain_cnt;

        // warp.sync();

        pref.commit(&VSMEM(k), fsm[k].next[threadIdx.x], sizeof(Entry));
        fsm[k].state[threadIdx.x] = probe::state_t::MATCH;
        fsm[k].active[threadIdx.x] = true;
      }

      // warp.sync();

      break;
    }

    default:
      break;
    }
    ++k;
  }

  /// @note without warp.sync() there will be errors
  warp.sync();

  // handle RVS
  if (warplane < rvs_cnt) {
    Tuple s_tuple = rvs_reg.s_tuple;
    Entry *next = rvs_reg.next;

    while (next) {
      assert(next);
      Tuple r_tuple = next->tuple;
      if (r_tuple.k == s_tuple.k) {
        aggr_fn_local(r_tuple.v, s_tuple.v, &aggr_local);
      }
      next = next->header.next;
    }
  }

  aggr_fn_global(aggr_local, o_aggr);
}

} // namespace probe

__global__ void print_ht_kernel(EntryHeader *ht_slot, int n) {
  for (int i = 0; i < n; ++i) {
    printf("%d: %p\n", i, ht_slot[i].next);
  }
}
void print_ht(EntryHeader *d_ht_slot, int n) {
  print_ht_kernel<<<1, 1>>>(d_ht_slot, n);
  CHKERR(cudaDeviceSynchronize());
}

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key,
         int32_t *s_payload, int32_t s_n, ConfigIMV cfg) {
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
    // const int smeme_size = PDIST * cfg.build_blocksize * sizeof(uint64_t);
    // build::build_ht_prefetch<<<cfg.build_gridsize, cfg.build_blocksize,
    //                            smeme_size, stream>>>(d_r, d_entries, r_n,
    //                                                  d_ht_slot, ht_size_log);
    build::
        build_ht_naive<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
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
      probe::probe_ht_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smeme_size,
                          stream>>>(d_s, s_n, d_ht_slot, ht_size_log, d_aggr);
    } else if (cfg.method == 2) {
      const int smeme_size = PDIST * cfg.probe_blocksize * sizeof(Entry);
      fmt::print("smem_size = {}\n", smeme_size);
      probe::probe_ht_2stage<<<cfg.probe_gridsize, cfg.probe_blocksize, 0,
                               stream>>>(d_s, s_n, d_ht_slot, ht_size_log,
                                         d_aggr);
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

  fmt::print("Join IMV (bucket size = 1)\n"
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

} // namespace imv
} // namespace classicjoin