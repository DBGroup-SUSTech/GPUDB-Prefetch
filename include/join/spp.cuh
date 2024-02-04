#pragma once
#include "config.cuh"
#include "join/config.cuh"
#include "util/util.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_pipeline.h>

// Non-partitioned Hash Join in Global Memory
// Unique hash table
// Late materialization
// Output payloads to the location of input
// 4byte-4byte table
// hash table do not save key/value, only save row id
// uses bucket chain to solve conflict, bucket size = 1
// still has bug

namespace join
{
namespace spp
{

#define D 2
#define K 3
// the virtual_tuplle_id in the thread tid
#define REAL_TUPLE_ID(virtual_tuple_id) ((tid) + (stride) * (virtual_tuple_id))

typedef enum element_state_t
{
    HASH = 0,
    NEXT,
    MATCH,
    PAYLOAD,
    DONE,
} element_state_t;

typedef enum pipline_state_t
{
    NORMAL = 0,   // pipeline is running without any interruption
    INTERRUPTION, // pipline is interrupted by PAYLOAD
    RENORMAL,     // pipline will be normal in next itr
} pipline_line_state_t;

struct prefetch_t
{
    int pending = 0; // number of pending requests
    __device__ __forceinline__ void commit(void *__restrict__ dst_shared, const void *__restrict__ src_global,
                                           size_t size_and_align, size_t zfill = 0UL)
    {
        __pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill);
        __pipeline_commit();
        ++pending;
    }
    __device__ __forceinline__ void wait()
    {
        assert(pending);
        __pipeline_wait_prior(pending - 1);
        --pending;
    }
};

struct prefetch_handler_t
{
    struct prefetch_t pref;
    __device__ __forceinline__ void commit(pipline_state_t pipline_state, void *__restrict__ dst_shared,
                                           const void *__restrict__ src_global, size_t size_and_align,
                                           size_t zfill = 0UL)
    {
        if (pipline_state != pipline_state_t::INTERRUPTION)
        {
            pref.commit(dst_shared, src_global, size_and_align);
        }
        // when pipline_state is INTERRUPTION
        else
        {
            *(int32_t *)dst_shared = *(int32_t *)src_global;
        }
    }
    __device__ __forceinline__ void wait(pipline_state_t pipline_state)
    {
        if (pipline_state != pipline_state_t::INTERRUPTION)
        {
            pref.wait();
        }
    }
};

/// @breif build hash table for R
/// @param [in]     r_key       all keys of relation R
/// @param [in]     n           number of keys
/// @param [in out] ht_link     hash table links, init -> 0
/// @param [in out] ht_slot     hash table slots, init -> 0, header of links
/// @param [in]     ht_size_log hash table size = 1 << ht_size_log (>= n)
__global__ void build_ht(int32_t *r_key, int32_t r_n, int32_t *ht_link, int32_t *ht_slot, int32_t ht_size_log)
{
    int ht_size = 1 << ht_size_log;
    int ht_mask = ht_size - 1;

    // TODO: try vector
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < r_n; i += stride)
    {
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
__global__ void probe_ht_1(int32_t *s_key, int32_t *s_payload, int32_t s_n, int32_t *r_key, int32_t *r_payload,
                           int32_t *ht_link, int32_t *ht_slot, int32_t ht_size_log, int32_t *o_aggr)
{
    extern __shared__ int shared_buffer[];

    int32_t reg_s_key[K * D + 1];
    int32_t reg_s_payload[K * D + 1];
    int32_t reg_r_tuple_id[K * D + 1];
    element_state_t reg_state[K * D + 1];

    int32_t reg_r_key = -1;
    int32_t reg_r_payload = -1;

    int ht_size = 1 << ht_size_log;
    int ht_mask = ht_size - 1;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int probe_num = 32;

    int aggr_local = 0;

    // prefetch_t pref{};
    prefetch_handler_t pref_handler{};

    int hash_virtual_id = 0;
    int next_virtual_id = -D;
    int match_virtual_id = -2 * D;
    int payload_virtual_id = -3 * D;

    int rematch_virtual_s_tuple_id = -1;
    int finish_num = 0;
    pipline_state_t pipline_state = pipline_state_t::NORMAL;

    int idx = -1;
    int shared_buffer_idx = -1;

    while (finish_num != probe_num)
    {
        // hash stage:
        if (pipline_state != pipline_state_t::INTERRUPTION)
        {
            idx = hash_virtual_id % (K * D + 1);
            ++hash_virtual_id;
            if (hash_virtual_id <= probe_num)
            {
                shared_buffer_idx = blockDim.x * idx + threadIdx.x;
                reg_state[idx] = element_state_t::HASH;
                // reg_s_key[idx] = s_key[REAL_TUPLE_ID(hash_virtual_id - 1)];
                // reg_s_payload[idx] = s_payload[REAL_TUPLE_ID(hash_virtual_id - 1)];
                reg_s_key[idx] = s_key[tid + stride * (hash_virtual_id - 1)];
                reg_s_payload[idx] = s_payload[tid + stride * (hash_virtual_id - 1)];
                // hash
                int slot_num = reg_s_key[idx] & ht_mask;
                // pref.commit(&shared_buffer[shared_buffer_idx], &ht_slot[slot_num], sizeof(int32_t));
                pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &ht_slot[slot_num],
                                    sizeof(int32_t));

                reg_state[idx] = element_state_t::NEXT;
            }
            pipline_state = pipline_state_t::NORMAL;
        }

        // next stage:
        if (pipline_state == pipline_state_t::NORMAL)
        {
            idx = next_virtual_id % (K * D + 1);
            ++next_virtual_id;
        }
        else
        {
            idx = rematch_virtual_s_tuple_id % (K * D + 1);
        }
        if (pipline_state == pipline_state_t::INTERRUPTION ||
            (pipline_state == pipline_state_t::NORMAL && (next_virtual_id <= probe_num) && next_virtual_id >= 1))
        {
            shared_buffer_idx = blockDim.x * idx + threadIdx.x;
            // pref.wait();
            pref_handler.wait(pipline_state);
            // r_tuple_id = next - 1
            reg_r_tuple_id[idx] = shared_buffer[shared_buffer_idx] - 1;
            if (reg_r_tuple_id[idx] == -1)
            {
                finish_num++;
                reg_state[idx] = element_state_t::DONE;

                if (pipline_state == pipline_state_t::INTERRUPTION)
                {
                    pipline_state = pipline_state_t::RENORMAL;
                }
            }
            else
            {
                // pref.commit(&shared_buffer[shared_buffer_idx], &r_key[reg_r_tuple_id[idx]], sizeof(int32_t));
                pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &r_key[reg_r_tuple_id[idx]],
                                    sizeof(int32_t));
                reg_state[idx] = element_state_t::MATCH;
            }
        }

        // match stage
        if (pipline_state != pipline_state_t::RENORMAL)
        {
            if (pipline_state == pipline_state_t::NORMAL)
            {
                idx = match_virtual_id % (K * D + 1);
                ++match_virtual_id;
            }
            else
            {
                idx = rematch_virtual_s_tuple_id % (K * D + 1);
            }
            if (pipline_state == pipline_state_t::INTERRUPTION ||
                (pipline_state == pipline_state_t::NORMAL && (match_virtual_id <= probe_num) && match_virtual_id >= 1))
            {
                if (reg_state[idx] == element_state_t::MATCH)
                {
                    shared_buffer_idx = blockDim.x * idx + threadIdx.x;
                    // pref.wait();
                    pref_handler.wait(pipline_state);
                    reg_r_key = shared_buffer[shared_buffer_idx];
                    if (reg_r_key == reg_s_key[idx])
                    {
                        // pref.commit(&shared_buffer[shared_buffer_idx], &r_payload[reg_r_tuple_id[idx]],
                        //             sizeof(int32_t));
                        pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx],
                                            &r_payload[reg_r_tuple_id[idx]], sizeof(int32_t));
                        reg_state[idx] = element_state_t::PAYLOAD;
                    }
                    else
                    {
                        // pref.commit(&shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]],
                        // sizeof(int32_t));
                        pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx],
                                            &ht_link[reg_r_tuple_id[idx]], sizeof(int32_t));
                        reg_state[idx] = element_state_t::NEXT;
                    }
                }
            }
        }

        // payload stage
        if (pipline_state != pipline_state_t::RENORMAL)
        {
            if (pipline_state == pipline_state_t::NORMAL)
            {
                rematch_virtual_s_tuple_id = payload_virtual_id;
                idx = payload_virtual_id % (K * D + 1);
                payload_virtual_id++;
            }
            else
            {
                idx = rematch_virtual_s_tuple_id % (K * D + 1);
            }
            if (pipline_state == pipline_state_t::INTERRUPTION ||
                (pipline_state == pipline_state_t::NORMAL && (payload_virtual_id <= probe_num) &&
                 payload_virtual_id >= 1))
            {
                if (reg_state[idx] == element_state_t::PAYLOAD)
                {
                    shared_buffer_idx = blockDim.x * idx + threadIdx.x;

                    // pref.wait();
                    pref_handler.wait(pipline_state);
                    // wait() should before seting pipline_state_t::INTERRUPTION, commit() should after.
                    pipline_state = pipline_state_t::INTERRUPTION;
                    reg_r_payload = shared_buffer[shared_buffer_idx];
                    aggr_fn_local(reg_r_payload, reg_s_payload[idx], &aggr_local);
                    // pref.commit(&shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]], sizeof(int32_t));
                    pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]],
                                        sizeof(int32_t));

                    reg_state[idx] = element_state_t::NEXT;
                }
            }
        }
    }
    aggr_fn_global(aggr_local, o_aggr);
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
__launch_bounds__(256, 1) __global__
    void probe_ht_2(int32_t *s_key, int32_t *s_payload, int32_t s_n, int32_t *r_key, int32_t *r_payload,
                    int32_t *ht_link, int32_t *ht_slot, int32_t ht_size_log, int32_t *o_aggr)
{
    extern __shared__ int shared_buffer[];

    int32_t reg_s_key[K * D + 1];
    int32_t reg_s_payload[K * D + 1];
    int32_t reg_r_tuple_id[K * D + 1];
    element_state_t reg_state[K * D + 1];

    int32_t reg_r_key = -1;
    int32_t reg_r_payload = -1;

    int ht_size = 1 << ht_size_log;
    int ht_mask = ht_size - 1;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int probe_num = 32;

    int aggr_local = 0;

    // prefetch_t pref{};
    prefetch_handler_t pref_handler{};

    int hash_virtual_id = 0;
    int next_virtual_id = -D;
    int match_virtual_id = -2 * D;
    int payload_virtual_id = -3 * D;

    int rematch_virtual_s_tuple_id = -1;
    int finish_num = 0;
    pipline_state_t pipline_state = pipline_state_t::NORMAL;

    int idx = -1;
    int shared_buffer_idx = -1;

    while (finish_num != probe_num)
    {
        // hash stage:
        if (pipline_state != pipline_state_t::INTERRUPTION)
        {
            idx = hash_virtual_id % (K * D + 1);
            ++hash_virtual_id;
            if (hash_virtual_id <= probe_num)
            {
                shared_buffer_idx = blockDim.x * idx + threadIdx.x;
                reg_state[idx] = element_state_t::HASH;
                // reg_s_key[idx] = s_key[REAL_TUPLE_ID(hash_virtual_id - 1)];
                // reg_s_payload[idx] = s_payload[REAL_TUPLE_ID(hash_virtual_id - 1)];
                reg_s_key[idx] = s_key[tid + stride * (hash_virtual_id - 1)];
                reg_s_payload[idx] = s_payload[tid + stride * (hash_virtual_id - 1)];
                // hash
                int slot_num = reg_s_key[idx] & ht_mask;
                // pref.commit(&shared_buffer[shared_buffer_idx], &ht_slot[slot_num], sizeof(int32_t));
                pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &ht_slot[slot_num],
                                    sizeof(int32_t));

                reg_state[idx] = element_state_t::NEXT;
            }
            pipline_state = pipline_state_t::NORMAL;
        }

        // next stage:
        if (pipline_state == pipline_state_t::NORMAL)
        {
            idx = next_virtual_id % (K * D + 1);
            ++next_virtual_id;
        }
        else
        {
            idx = rematch_virtual_s_tuple_id % (K * D + 1);
        }
        if (pipline_state == pipline_state_t::INTERRUPTION ||
            (pipline_state == pipline_state_t::NORMAL && (next_virtual_id <= probe_num) && next_virtual_id >= 1))
        {
            shared_buffer_idx = blockDim.x * idx + threadIdx.x;
            // pref.wait();
            pref_handler.wait(pipline_state);
            // r_tuple_id = next - 1
            reg_r_tuple_id[idx] = shared_buffer[shared_buffer_idx] - 1;
            if (reg_r_tuple_id[idx] == -1)
            {
                finish_num++;
                reg_state[idx] = element_state_t::DONE;

                if (pipline_state == pipline_state_t::INTERRUPTION)
                {
                    pipline_state = pipline_state_t::RENORMAL;
                }
            }
            else
            {
                // pref.commit(&shared_buffer[shared_buffer_idx], &r_key[reg_r_tuple_id[idx]], sizeof(int32_t));
                pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &r_key[reg_r_tuple_id[idx]],
                                    sizeof(int32_t));
                reg_state[idx] = element_state_t::MATCH;
            }
        }

        // match stage
        if (pipline_state != pipline_state_t::RENORMAL)
        {
            if (pipline_state == pipline_state_t::NORMAL)
            {
                idx = match_virtual_id % (K * D + 1);
                ++match_virtual_id;
            }
            else
            {
                idx = rematch_virtual_s_tuple_id % (K * D + 1);
            }
            if (pipline_state == pipline_state_t::INTERRUPTION ||
                (pipline_state == pipline_state_t::NORMAL && (match_virtual_id <= probe_num) && match_virtual_id >= 1))
            {
                if (reg_state[idx] == element_state_t::MATCH)
                {
                    shared_buffer_idx = blockDim.x * idx + threadIdx.x;
                    // pref.wait();
                    pref_handler.wait(pipline_state);
                    reg_r_key = shared_buffer[shared_buffer_idx];
                    if (reg_r_key == reg_s_key[idx])
                    {
                        // pref.commit(&shared_buffer[shared_buffer_idx], &r_payload[reg_r_tuple_id[idx]],
                        //             sizeof(int32_t));
                        pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx],
                                            &r_payload[reg_r_tuple_id[idx]], sizeof(int32_t));
                        reg_state[idx] = element_state_t::PAYLOAD;
                    }
                    else
                    {
                        // pref.commit(&shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]],
                        // sizeof(int32_t));
                        pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx],
                                            &ht_link[reg_r_tuple_id[idx]], sizeof(int32_t));
                        reg_state[idx] = element_state_t::NEXT;
                    }
                }
            }
        }

        // payload stage
        if (pipline_state != pipline_state_t::RENORMAL)
        {
            if (pipline_state == pipline_state_t::NORMAL)
            {
                rematch_virtual_s_tuple_id = payload_virtual_id;
                idx = payload_virtual_id % (K * D + 1);
                payload_virtual_id++;
            }
            else
            {
                idx = rematch_virtual_s_tuple_id % (K * D + 1);
            }
            if (pipline_state == pipline_state_t::INTERRUPTION ||
                (pipline_state == pipline_state_t::NORMAL && (payload_virtual_id <= probe_num) &&
                 payload_virtual_id >= 1))
            {
                if (reg_state[idx] == element_state_t::PAYLOAD)
                {
                    shared_buffer_idx = blockDim.x * idx + threadIdx.x;

                    // pref.wait();
                    pref_handler.wait(pipline_state);
                    // wait() should before seting pipline_state_t::INTERRUPTION, commit() should after.
                    pipline_state = pipline_state_t::INTERRUPTION;
                    reg_r_payload = shared_buffer[shared_buffer_idx];
                    aggr_fn_local(reg_r_payload, reg_s_payload[idx], &aggr_local);
                    // pref.commit(&shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]], sizeof(int32_t));
                    pref_handler.commit(pipline_state, &shared_buffer[shared_buffer_idx], &ht_link[reg_r_tuple_id[idx]],
                                        sizeof(int32_t));

                    reg_state[idx] = element_state_t::NEXT;
                }
            }
        }
    }
    aggr_fn_global(aggr_local, o_aggr);
}

int join(int32_t *r_key, int32_t *r_payload, int32_t r_n, int32_t *s_key, int32_t *s_payload, int32_t s_n, Config cfg)
{
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
               cfg.build_gridsize, cfg.build_blocksize, cfg.probe_gridsize, cfg.probe_blocksize);

    {
        CHKERR(cudaEventRecordWithFlags(start_build, stream, cudaEventRecordExternal));
        build_ht<<<cfg.build_gridsize, cfg.build_blocksize, 0, stream>>>(d_r_key, r_n, d_ht_link, d_ht_slot,
                                                                         ht_size_log);
        CHKERR(cudaEventRecordWithFlags(end_build, stream, cudaEventRecordExternal));
    }

    {
        CHKERR(cudaEventRecordWithFlags(start_probe, stream, cudaEventRecordExternal));
        int smem_size = (K * D + 1) * cfg.probe_blocksize * sizeof(int32_t);
        probe_ht_1<<<cfg.probe_gridsize, cfg.probe_blocksize, smem_size, stream>>>(
            d_s_key, d_s_payload, s_n, d_r_key, d_r_payload, d_ht_link, d_ht_slot, ht_size_log, d_aggr);
        CHKERR(cudaEventRecordWithFlags(end_probe, stream, cudaEventRecordExternal));
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
               ms_build, r_n * 1.0 / ms_build * 1000, ms_probe, s_n * 1.0 / ms_probe * 1000);

    // CHKERR(cutil::CpyDeviceToHost(o_payload, d_o_payload, s_n));
    int32_t aggr;
    CHKERR(cutil::CpyDeviceToHost(&aggr, d_aggr, 1));
    return aggr;
}
} // namespace spp
} // namespace join