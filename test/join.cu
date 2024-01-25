#include <gtest/gtest.h>

#include "datagen/generator_ETHZ.cuh"
#include "join/gp.cuh"
#include "join/imv.cuh"
#include "join/naive.cuh"
#include "join/spp.cuh"
#include "util/args.cuh"

TEST(skew, naive) {
  // TODO
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  assert(r_n <= s_n);

  std::string r_fname = cutil::rel_fname(true, "r", r_n);
  std::string s_fname = cutil::rel_fname(false, "s", s_n);

  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(
      !datagen::create_relation_zipf(s_fname.c_str(), s_key, s_n, r_n, skew));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
  fmt::print(
      "Create relation S from R, with {} tuples ({} MB) "
      "using zipf keys, skew = {}\n",
      s_n, s_n * sizeof(int32_t) / 1024 / 1024, skew);

  // fmt::print("R: {}\n", fmt_arr(r_key, r_n));
  // fmt::print("S: {}\n", fmt_arr(s_key, s_n));

  int32_t *r_payload = new int32_t[r_n];
  int32_t *s_payload = new int32_t[s_n];

  // Payload set to equal with key
  std::copy_n(r_key, r_n, r_payload);
  std::copy_n(s_key, s_n, s_payload);

  fmt::print("R payload: {}\n", cutil::fmt_arr(r_payload, 20));
  fmt::print("S payload: {}\n", cutil::fmt_arr(s_payload, 20));

  int els_per_thread = 4;
  int threads_per_block = 512;
  join::Config config;
  {  // build kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }
  {  // probe kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
    config.probe_gridsize = blocks_per_grid;
    config.probe_blocksize = threads_per_block;
  }

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      join::naive::join(r_key, r_payload, r_n, s_key, s_payload, s_n, config));
}

TEST(zipf, gp) {
  // TODO
}