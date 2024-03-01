#include <gtest/gtest.h>
#include <time.h>

#include <bitset>
#include <iostream>

#include "util/util.cuh"

constexpr unsigned MASK_ALL_LANES = 0xFFFFFFFF;
__global__ void imv_shm(int *v1, int *v2) {
  assert(blockDim.x == 32);
  assert(gridDim.x == 1);
  // rvs
  int dvs = 0;
  int dvs_cnt = 0;
  __shared__ int rvs[32];
  int rvs_cnt = 0;

  // warp info
  unsigned warplane = threadIdx.x % 32;
  unsigned prefixlanes = 0xffffffff >> (32 - warplane);

  // fetch n1
  {
    dvs = v1[warplane];
    bool active = dvs;
    int active_mask = __ballot_sync(MASK_ALL_LANES, active);
    int active_cnt = __popc(active_mask);

    clock_t t = clock();
    if (active_cnt + rvs_cnt < 32) {
      int prefix_cnt = __popc(active_mask & prefixlanes);
      if (active) {
        int offset = rvs_cnt + prefix_cnt;
        rvs[offset] = dvs;
      }
      rvs_cnt += active_cnt;
      dvs_cnt = 0;
    } else {
      int inactive_mask = ~active_mask;
      int prefix_cnt = __popc(inactive_mask & prefixlanes);
      int remain_cnt = rvs_cnt + active_cnt - 32;
      if (!active) {
        int offset = remain_cnt + prefix_cnt;
        dvs = rvs[offset];
      }
      rvs_cnt = remain_cnt;
      dvs_cnt = 32;
    }
    __syncwarp();
    t = clock() - t;
    printf("shm t = %ld\n", t);
  }

  // printf("A: tid=%d, rvs_cnt = %d, dvs_cnt = %d, rvs = %d, dvs = %d\n",  //
  //  warplane, rvs_cnt, dvs_cnt, rvs[warplane], dvs);                //

  // fetch n2
  {
    dvs = v2[warplane];
    bool active = dvs;
    int active_mask = __ballot_sync(MASK_ALL_LANES, active);
    int active_cnt = __popc(active_mask);

    if (active_cnt + rvs_cnt < 32) {
      int prefix_cnt = __popc(active_mask & prefixlanes);
      if (active) {
        int offset = rvs_cnt + prefix_cnt;
        rvs[offset] = dvs;
      }
      rvs_cnt += active_cnt;
      dvs_cnt = 0;
    } else {
      int inactive_mask = ~active_mask;
      int prefix_cnt = __popc(inactive_mask & prefixlanes);
      int remain_cnt = rvs_cnt + active_cnt - 32;
      if (!active) {
        int offset = remain_cnt + prefix_cnt;
        dvs = rvs[offset];
      }
      rvs_cnt = remain_cnt;
      dvs_cnt = 32;
    }
  }

  __syncwarp();

  v1[warplane] = dvs;
  v2[warplane] = rvs[warplane];
}

__global__ void imv_reg(int *v1, int *v2) {
  assert(blockDim.x == 32);
  assert(gridDim.x == 1);

  // rvs
  int dvs = 0;
  int rvs = 0;
  int dvs_cnt = 0;
  int rvs_cnt = 0;

  // warp info
  unsigned warplane = threadIdx.x % 32;
  unsigned prefixlanes = 0xffffffff >> (32 - warplane);

  // fetch n1
  {
    dvs = v1[warplane];

    bool active = dvs;
    int active_mask = __ballot_sync(MASK_ALL_LANES, active);
    int active_cnt = __popc(active_mask);

    clock_t t = clock();

    if (active_cnt + rvs_cnt < 32) {
      int offset = warplane - rvs_cnt;
      int src_lane = __fns(active_mask, 0, offset + 1);
      int tmp = __shfl_sync(MASK_ALL_LANES, dvs, src_lane);
      if (warplane >= rvs_cnt) rvs = tmp;
      rvs_cnt += active_cnt;
      dvs_cnt = 0;
    } else {
      int inactive_mask = ~active_mask;
      int prefix_cnt = __popc(inactive_mask & prefixlanes);
      int remain_cnt = rvs_cnt + active_cnt - 32;
      int offset = remain_cnt + prefix_cnt;
      int tmp = __shfl_sync(MASK_ALL_LANES, rvs, offset);
      if (!active) dvs = tmp;
      rvs_cnt = remain_cnt;
      dvs_cnt = 32;
    }
    t = clock() - t;
    printf("reg t = %ld\n", t);
  }

  // printf("A: tid=%d, rvs_cnt = %d, dvs_cnt = %d, rvs = %d, dvs = %d\n",  //
  //        warplane, rvs_cnt, dvs_cnt, rvs, dvs);                          //

  // fetch n2
  {
    dvs = v2[warplane];
    bool active = dvs;
    int active_mask = __ballot_sync(MASK_ALL_LANES, active);
    int active_cnt = __popc(active_mask);

    if (active_cnt + rvs_cnt < 32) {
      int offset = warplane - rvs_cnt;
      int src_lane = __fns(active_mask, 0, offset);
      // printf("lane %d offset %d src %d\n", warplane, offset, src_lane);
      int tmp = __shfl_sync(MASK_ALL_LANES, dvs, src_lane);
      if (warplane >= active_cnt) rvs = tmp;
      rvs_cnt += active_cnt;
      dvs_cnt = 0;
    } else {
      int inactive_mask = ~active_mask;
      int prefix_cnt = __popc(inactive_mask & prefixlanes);
      int remain_cnt = rvs_cnt + active_cnt - 32;
      int offset = remain_cnt + prefix_cnt;
      int tmp = __shfl_sync(MASK_ALL_LANES, rvs, offset);
      if (!active) dvs = tmp;
      rvs_cnt = remain_cnt;
      dvs_cnt = 32;
    }
  }

  v1[warplane] = dvs;
  v2[warplane] = rvs;
  // printf("jB: tid=%d, rvs_cnt = %d, dvs_cnt = %d, rvs = %d, dvs = %d\n",  //
  //        warplane, rvs_cnt, dvs_cnt, rvs, dvs);                           //
}

TEST(imv, shm) {
  int v1[32]{0}, v2[32]{0};
  for (int i = 0; i < 32; ++i) {
    if (rand() % 3) v1[i] = i;
    if (rand() % 3) v2[i] = i;
  }
  for (int i = 0; i < 32; ++i) {
    printf("%d ", v1[i]);
  }
  printf("\n");
  for (int i = 0; i < 32; ++i) {
    printf("%d ", v2[i]);
  }
  printf("\n");

  int *d_v1, *d_v2;
  cutil::DeviceAlloc(d_v1, 32);
  cutil::DeviceAlloc(d_v2, 32);
  cutil::CpyHostToDevice(d_v1, v1, 32);
  cutil::CpyHostToDevice(d_v2, v2, 32);

  cudaEvent_t start, end;
  imv_shm<<<1, 32>>>(d_v1, d_v2);
}
TEST(imv, reg) {
  int v1[32]{0}, v2[32]{0};
  for (int i = 0; i < 32; ++i) {
    if (rand() % 3) v1[i] = i;
    if (rand() % 3) v2[i] = i;
  }
  for (int i = 0; i < 32; ++i) {
    printf("%d ", v1[i]);
  }
  printf("\n");
  for (int i = 0; i < 32; ++i) {
    printf("%d ", v2[i]);
  }
  printf("\n");

  int *d_v1, *d_v2;
  cutil::DeviceAlloc(d_v1, 32);
  cutil::DeviceAlloc(d_v2, 32);
  cutil::CpyHostToDevice(d_v1, v1, 32);
  cutil::CpyHostToDevice(d_v2, v2, 32);

  imv_reg<<<1, 32>>>(d_v1, d_v2);
  CHKERR(cudaDeviceSynchronize());
}