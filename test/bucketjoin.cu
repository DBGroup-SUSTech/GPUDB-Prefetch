#include <fmt/format.h>
#include <gtest/gtest.h>

#include "bucketjoin/naive.cuh"
#include "datagen/generator_ETHZ.cuh"
#include "util/args.cuh"

TEST(skew_r_unique_s, naive) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, r_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using zipf keys, skew= {} \n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024, skew);
  fmt::print(
      "Create relation S from R, with {} tuples ({} MB) "
      "using unique keys\n",
      s_n, s_n * sizeof(int32_t) / 1024 / 1024);

  // fmt::print("R: {}\n", fmt_arr(r_key, r_n));
  // fmt::print("S: {}\n", fmt_arr(s_key, s_n));

  int32_t *r_payload = new int32_t[r_n];
  int32_t *s_payload = new int32_t[s_n];

  // Payload set to equal with key
  std::copy_n(r_key, r_n, r_payload);
  std::copy_n(s_key, s_n, s_payload);

  // fmt::print("R payload: {}\n", cutil::fmt_arr(r_payload, 20));
  // fmt::print("S payload: {}\n", cutil::fmt_arr(s_payload, 20));

  bucketjoin::Config config;
  {  // build kernel
    // int els_per_thread = 1;
    // int threads_per_block = 256;
    // const int els_per_block = threads_per_block * els_per_thread;
    // const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
    // config.build_gridsize = blocks_per_grid;
    // config.build_blocksize = threads_per_block;
    config.build_gridsize = 144;
    config.build_blocksize = 256;
  }
  {  // probe kernel
    int els_per_thread = 4;
    int threads_per_block = 512;
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
    config.probe_gridsize = blocks_per_grid;
    config.probe_blocksize = threads_per_block;
  }
  //   const int blocksize = args::get<int>("BSIZE");
  //   const int gridsize = args::get<int>("GSIZE");
  //   config.build_blocksize = blocksize;
  //   config.build_gridsize = gridsize;
  //   config.probe_blocksize = blocksize;
  //   config.probe_gridsize = gridsize;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      bucketjoin::naive::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                              config));
}
