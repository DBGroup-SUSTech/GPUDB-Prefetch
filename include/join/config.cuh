#pragma once
namespace join {
struct Config {
  int build_gridsize = -1;
  int build_blocksize = -1;
  int probe_gridsize = -1;
  int probe_blocksize = -1;
};

__device__ void aggr_fn_local(int32_t r_payload, int32_t s_payload,
                              int32_t* aggr) {
  *aggr += r_payload * s_payload;
}

__device__ void aggr_fn_atomic(int32_t r_payload, int32_t s_payload,
                               int32_t* aggr) {
  atomicAdd(aggr, r_payload * s_payload);
}

__device__ void aggr_fn_global(int32_t aggr_local, int32_t* aggr_global) {
  atomicAdd(aggr_global, aggr_local);
}
}  // namespace join