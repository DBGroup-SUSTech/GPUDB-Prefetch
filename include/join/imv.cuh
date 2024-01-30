#pragma once
#include <cuda_pipeline.h>
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
namespace imv {

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

// for prefetch
// -----------------------------------------------------------------
constexpr int PDIST = 16;            // prefetch distance & group size
constexpr int PADDING = 1;           // solve bank conflict
constexpr int W = 32;                // vector width
constexpr int WARPS_PER_THREAD = 4;  // fix 4 warps per thread
#define VSMEM(index) v[index + (PDIST + PADDING) * threadIdx.x]

constexpr unsigned int MASK_ALL_LANES = 0xFFFFFFFF;

enum class state_t {
  HASH = 0,  // load s_key, s_payload, then hash, prefetch bucket header (next =
             // ht_slot[hash])
  NEXT,      // load next, prefetch r_key[next]
  MATCH,     // load r_key[next], matching, and prefetch r_payload[next]
  PAYLOAD,   // load r_payload[next], aggregation, prefetch r_key[next]
  DONE,
};

struct fsm_t {
  int32_t s_key;
  int32_t s_payload;
  int32_t next;
  state_t state;
};

struct fsm_shared_t {
  int32_t s_key[32 * WARPS_PER_THREAD];
  int32_t s_payload[32 * WARPS_PER_THREAD];
  int32_t next[32 * WARPS_PER_THREAD];
  // state_t state; // leave it outside
};

struct prefetch_t {
  int pending = 0;  // number of pending requests
  __device__ __forceinline__ void commit(void* __restrict__ dst_shared,
                                         const void* __restrict__ src_global,
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

// ---------------------------------------------------------------------------

/// @brief Use S to probe
/// @param [in]     s_key       all keys of S
/// @param [in]     s_payload   all payload of S
/// @param [in]     s_n         number of S items
/// @param [in]     r_key       all keys of R
/// @param [in]     r_payload   all payload of R
/// @param [in]     ht_link     R hash table link
/// @param [in]     ht_slot     R hash table slot
/// @param [in]     ht_size_log R hash table size = ht_size_log << 1
/// @param [in]     o_aggr      output buffer for aggregation result
__global__ void probe_ht(int32_t* s_key, int32_t* s_payload, int32_t s_n,
                         int32_t* r_key, int32_t* r_payload, int32_t* ht_link,
                         int32_t* ht_slot, int32_t ht_size_log,
                         int32_t* o_aggr) {
  int ht_size = 1 << ht_size_log;
  int ht_mask = ht_size - 1;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i = tid;
  assert(blockDim.x == WARPS_PER_THREAD * 32);

  unsigned warpid = (threadIdx.x / 32);
  unsigned warplane = (threadIdx.x % 32);
  unsigned prefixlanes = (0xffffffff >> (32 - warplane));

  // prefetch
  prefetch_t pref{};
  extern __shared__ int32_t v[];

  /// @note all state and cnt in a warp should be the same
  // fsm_t fsm[PDIST]{};      // TODO: vector<key,payload,next> + scalar<state>
  // __shared__ fsm_t RVS{};  // TODO: vector<key,payload,next> + scalar<state>

  __shared__ fsm_shared_t fsm[PDIST];  // TODO: DVS, save in register instead
  state_t fsm_state[PDIST];            // state flag stored in register

  __shared__ fsm_shared_t RVS_next;    // RVS after next
  __shared__ fsm_shared_t RVS_match0;  // RVS after unmatch
  __shared__ fsm_shared_t RVS_match1;  // RVS after match
  __shared__ fsm_shared_t DVS_match0;  // Buffering the full part of <RVS_match>
  // __shared__ state_t fsm_state[PDIST];   // TODO: save in shared mem
  int RVS_next_cnt = 0;
  int RVS_match0_cnt = 0;  // TODO: scalar<cnt>
  int RVS_match1_cnt = 0;
  bool DVS_match0_full = false;

  bool active = 0;
  int all_done = 0, k = 0;
  int32_t aggr_local = 0;

  // cache
  while (all_done < PDIST) {
    k = ((k == PDIST) ? 0 : k);
    switch (fsm_state[k]) {
      case state_t::HASH: {
        if (i < s_n) {
          // TODO: sequential read, no prefetch
          int32_t s_key_v = s_key[i];                    // cache in register
          fsm[k].s_key[threadIdx.x] = s_key_v;           // v_load
          fsm[k].s_payload[threadIdx.x] = s_payload[i];  // v_load

          int32_t hval = s_key_v & ht_mask;  // v_hash
          i += stride;                       // read next vector

          pref.commit(&VSMEM(k), &ht_slot[hval], sizeof(int32_t));
          active = 1;
        } else {
          ++all_done;
          active = 0;
        }

        fsm_state[k] = state_t::NEXT;

        break;
      }
      case state_t::NEXT: {
        int next = 0;

        if (active) {
          pref.wait();
          next = VSMEM(k);
          fsm[k].next[threadIdx.x] = next;  // TODO: opt
          active = (next != 0);
        }

        // integration -------------------------------------------
        // active = active && (next != 0);
        int active_mask = __ballot_sync(MASK_ALL_LANES, active);
        int active_cnt = __popc(active_mask);
        if (active_cnt + RVS_next_cnt < W) {  // empty
          int prefix_cnt = __popc(active_mask & prefixlanes);
          if (active) {
            int offset = warpid * 32 + RVS_next_cnt + prefix_cnt;
            RVS_next.s_key[offset] = fsm[k].s_key[threadIdx.x];
            RVS_next.s_payload[offset] = fsm[k].s_payload[threadIdx.x];
            RVS_next.next[offset] = next;
          }

          RVS_next_cnt += active_cnt;

          // empty, switch to HASH
          active = false;  // empty
          fsm_state[k] = state_t::HASH;

        } else {  // full
          int inactive_mask = ~active_mask;
          int prefix_cnt = __popc(inactive_mask & prefixlanes);
          int remain_cnt = RVS_next_cnt + active_cnt - 32;
          if (!active) {
            int offset = warpid * 32 + remain_cnt + prefix_cnt;
            fsm[k].s_key[threadIdx.x] = RVS_next.s_key[offset];
            fsm[k].s_payload[threadIdx.x] = RVS_next.s_payload[offset];
            next = RVS_next.next[offset];
            fsm[k].next[threadIdx.x] = next;
          }

          RVS_next_cnt = remain_cnt;

          // full, continue to MATCH
          active = true;  // full
          fsm_state[k] = state_t::MATCH;
          pref.commit(&VSMEM(k), &r_key[next - 1], sizeof(int32_t));
        }
        // finish integration --------------------------------------

        // if (fsm[k].next) {
        //   fsm[k].state = state_t::MATCH;
        //   pref.commit(&VSMEM(k), &r_key[fsm[k].next[threadIdx.x] - 1],
        //               sizeof(int32_t));
        // } else {
        //   fsm[k].state = state_t::HASH;
        // }
        break;
      }
      case state_t::MATCH: {
        int32_t r_key_v = 0, s_key_v = 0;
        if (active) {
          pref.wait();
          r_key_v = VSMEM(k);
          s_key_v = fsm[k].s_key[threadIdx.x];
        }
        // integration -------------------------------------------
        // branch 1  -> PAYLOAD
        // branch 0  -> HASH
        // pack into RVS_match<0/1>

        // if RVS_match0 is full, move to DVS_match0, remain -> RVS_match0
        // if RVS_match1 is full, contine
        // else continue with DVS_match0

        int branch0 = active && (r_key_v != s_key_v);
        int branch0_mask = __ballot_sync(MASK_ALL_LANES, branch0);
        int branch0_cnt = __popc(branch0_mask);

        if (branch0_cnt + RVS_match0_cnt < W) {  // flush branch 0
          int prefix_cnt = __popc(branch0 & prefixlanes);
          if (branch0) {
            int offset = warpid * 32 + RVS_match0_cnt + prefix_cnt;
            RVS_match0.s_key[offset] = s_key_v;
            RVS_match0.s_payload[offset] = fsm[k].s_payload[threadIdx.x];
            RVS_match0.next[offset] = fsm[k].next[threadIdx.x];
          }
          RVS_match0_cnt += branch0_cnt;

        } else {  // branch 0 full -> Move to DVS buffer
          assert(DVS_match0_full == 0);
          if (branch0) {
            DVS_match0.s_key[threadIdx.x] = s_key_v;
            DVS_match0.s_payload[threadIdx.x] = fsm[k].s_payload[threadIdx.x];
            DVS_match0.next[threadIdx.x] = fsm[k].next[threadIdx.x];
          }

          int inactive_mask = ~branch0_mask;  // empty slot in DVS_match0
          int prefix_cnt = __popc(inactive_mask & prefixlanes);
          int remain_cnt = RVS_match0_cnt + branch0_cnt - 32;
          if (!branch0) {
            int offset = warpid * 32 + remain_cnt + prefix_cnt;
            DVS_match0.s_key[threadIdx.x] = RVS_match0.s_key[offset];
            DVS_match0.s_payload[threadIdx.x] = RVS_match0.s_payload[offset];
            DVS_match0.next[threadIdx.x] = RVS_match0.next[offset];
          }
          RVS_match0_cnt = remain_cnt;
          DVS_match0_full = 1;
        }

        int branch1 = active && (r_key_v == s_key_v);
        int branch1_mask = __ballot_sync(MASK_ALL_LANES, branch1);
        int branch1_cnt = __popc(branch1_mask);

        if (branch1_cnt + RVS_match1_cnt < W) {  // flush branch 1
          int prefix_cnt = __popc(branch1 & prefixlanes);
          if (branch1) {
            int offset = warpid * 32 + RVS_match1_cnt + prefix_cnt;
            RVS_match1.s_key[offset] = s_key_v;
            RVS_match1.s_payload[offset] = fsm[k].s_payload[threadIdx.x];
            RVS_match1.next[offset] = fsm[k].next[threadIdx.x];
          }
          RVS_match1_cnt += branch1_cnt;

          // empty, check DVS_match0 -> NEXT or switch to HASH
          if (DVS_match0_full) {
            fsm[k].s_key[threadIdx.x] = DVS_match0.s_key[threadIdx.x];
            fsm[k].s_payload[threadIdx.x] = DVS_match0.s_payload[threadIdx.x];
            int next = DVS_match0.next[threadIdx.x];
            fsm[k].next[threadIdx.x] = next;

            // switch to  DVS_match0, fully active
            active = true;
            fsm_state[k] = state_t::NEXT;
            pref.commit(&VSMEM(k), &ht_link[next - 1], sizeof(int32_t));

          } else {
            // still empty, inactive
            active = false;
            fsm_state[k] = state_t::HASH;
          }

        } else {  // fill branch 1
          int inactive_mask = ~branch1_mask;
          int prefix_cnt = __popc(inactive_mask & prefixlanes);
          int remain_cnt = RVS_match1_cnt + branch1_cnt - 32;
          if (!branch1) {
            int offset = warpid * 32 + remain_cnt + prefix_cnt;
            fsm[k].s_key[threadIdx.x] = RVS_match1.s_key[offset];
            fsm[k].s_payload[threadIdx.x] = RVS_match1.s_payload[offset];
            int next = RVS_match1.next[offset];
            fsm[k].next[threadIdx.x] = next;
          }
          RVS_match1_cnt = remain_cnt;

          // full contine to PAYLOAD
          active = true;
          fsm_state[k] = state_t::PAYLOAD;
          int next = fsm[k].next[threadIdx.x];
          pref.commit(&VSMEM(k), &r_payload[next - 1], sizeof(int32_t));
        }

        // if (r_key_v == fsm[k].s_key) {
        //   fsm[k].state = state_t::PAYLOAD;
        //   pref.commit(&VSMEM(k), &r_payload[fsm[k].next - 1],
        //   sizeof(int32_t));
        // } else {
        //   fsm[k].state = state_t::NEXT;
        //   pref.commit(&VSMEM(k), &ht_link[fsm[k].next - 1], sizeof(int32_t));
        // }
        break;
      }
      case state_t::PAYLOAD: {
        if (active) {
          pref.wait();
          int32_t r_payload_v = VSMEM(k);
          printf("%d\n", aggr_local);
          aggr_fn_local(r_payload_v, fsm[k].s_payload[threadIdx.x],
                        &aggr_local);

          fsm_state[k] = state_t::NEXT;
          pref.commit(&VSMEM(k), &ht_link[fsm[k].next[threadIdx.x] - 1],
                      sizeof(int32_t));
        }
        break;
      }
    }
    ++k;
  }

  // TODO: clear all remaining RVSs
  aggr_fn_global(aggr_local, o_aggr);
}

int join(int32_t* r_key, int32_t* r_payload, int32_t r_n, int32_t* s_key,
         int32_t* s_payload, int32_t s_n, Config cfg) {
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

  int32_t* d_aggr;
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
    CHKERR(cudaEventRecordWithFlags(  //
        start_build, stream, cudaEventRecordExternal));
    build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(
        d_r_key, r_n, d_ht_link, d_ht_slot, ht_size_log);
    CHKERR(cudaEventRecordWithFlags(  //
        end_build, stream, cudaEventRecordExternal));
  }

  {
    CHKERR(cudaEventRecordWithFlags(  //
        start_probe, stream, cudaEventRecordExternal));
    const int smem_size =
        (PDIST + PADDING) * cfg.probe_blocksize * sizeof(int32_t);
    probe_ht<<<cfg.probe_gridsize, cfg.probe_blocksize, smem_size, stream>>>(
        d_s_key, d_s_payload, s_n, d_r_key, d_r_payload, d_ht_link, d_ht_slot,
        ht_size_log, d_aggr);
    CHKERR(cudaEventRecordWithFlags(  //
        end_probe, stream, cudaEventRecordExternal));
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

  // CHKERR(cutil::CpyDeviceToHost(o_payload, d_o_payload, s_n));
  int32_t aggr;
  CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));
  return aggr;
}
}  // namespace imv
}  // namespace join