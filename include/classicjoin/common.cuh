#pragma once
// pre-allocated entries
namespace classicjoin {

// Non-partitioned Hash Join in Global Memory
// Non-Unique hash table
// Early Materialization
// Output payloads to the location of input
// 4byte-4byte table
// hash table save key/value
// uses bucket chain to solve conflict, bucket size = 1

struct EntryHeader {
  EntryHeader* next;  // pointer and dynamic allocated entry
  // int next;  // offset and static allocated entries
};

struct Tuple {
  int k;
  int v;
};

struct Entry {
  EntryHeader header;
  Tuple tuple;
};

struct Config {
  int build_gridsize = -1;
  int build_blocksize = -1;
  int probe_gridsize = -1;
  int probe_blocksize = -1;
};

__device__ __forceinline__ void aggr_fn_local(int32_t r_payload,
                                              int32_t s_payload,
                                              int32_t* aggr) {
  *aggr += r_payload * s_payload;
}

__device__ __forceinline__ void aggr_fn_atomic(int32_t r_payload,
                                               int32_t s_payload,
                                               int32_t* aggr) {
  atomicAdd(aggr, r_payload * s_payload);
}

__device__ __forceinline__ void aggr_fn_global(int32_t aggr_local,
                                               int32_t* aggr_global) {
  atomicAdd(aggr_global, aggr_local);
}

void col_to_row(int* key, int* payload, Tuple* tuples, int n) {
  for (int i = 0; i < n; ++i) {
    tuples[i].k = key[i];
    tuples[i].v = payload[i];
  }
}

}  // namespace classicjoin