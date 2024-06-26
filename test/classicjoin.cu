#include <fmt/core.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "classicjoin/amac.cuh"
#include "classicjoin/gp.cuh"
#include "classicjoin/imv.cuh"
#include "classicjoin/naive.cuh"
#include "classicjoin/spp.cuh"
#include "datagen/generator_ETHZ.cuh"
#include "util/args.cuh"

TEST(unique, naive) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  int els_per_thread = 4;
  int threads_per_block = 256;
  classicjoin::Config config;
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
      classicjoin::naive::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                               config));
}

TEST(measure, DISABLED_naive) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  int els_per_thread = 4;
  int threads_per_block = 512;
  classicjoin::Config config;
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
  //   config.build_blocksize = 256;
  //   config.build_gridsize = 100;
  //   config.probe_blocksize = 128;
  //   config.probe_gridsize = 72;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::naive::join_measure(r_key, r_payload, r_n, s_key, s_payload,
                                       s_n, config));
}

TEST(unique, amac) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  //   int els_per_thread = 4;
  //   int threads_per_block = 512;
  classicjoin::amac::ConfigAMAC config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  //   {  // probe kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //     config.probe_gridsize = blocks_per_grid;
  //     config.probe_blocksize = threads_per_block;
  //   }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::amac::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                              config));
}

TEST(unique, imv) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  //   int els_per_thread = 4;
  //   int threads_per_block = 512;
  classicjoin::imv::ConfigIMV config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  // {  // probe kernel
  //   const int els_per_block = threads_per_block * els_per_thread;
  //   const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //   config.probe_gridsize = blocks_per_grid;
  //   config.probe_blocksize = threads_per_block;
  // }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::imv::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                             config));
}

TEST(unique, gp) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  int els_per_thread = 16;
  int threads_per_block = 128;
  classicjoin::gp::ConfigGP config;
  {  // build kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }
  // { // probe kernel
  //   const int els_per_block = threads_per_block * els_per_thread;
  //   const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //   config.probe_gridsize = blocks_per_grid;
  //   config.probe_blocksize = threads_per_block;
  // }
  //   config.build_blocksize = 256;
  //   config.build_gridsize = 100;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;
  config.method = 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::gp::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                            config));
}

TEST(unique, spp) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(true, "r_uniq", r_n, 0);
  std::string s_fname = cutil::rel_fname(true, "s_uniq", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(!datagen::create_relation_unique(r_fname.c_str(), r_key, r_n, r_n));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using unique keys\n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024);
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

  int els_per_thread = 4;
  int threads_per_block = 256;
  classicjoin::spp::ConfigSPP config;
  {  // build kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }
  // { // probe kernel
  //   const int els_per_block = threads_per_block * els_per_thread;
  //   const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //   config.probe_gridsize = blocks_per_grid;
  //   config.probe_blocksize = threads_per_block;
  // }
  //   config.build_blocksize = 256;
  //   config.build_gridsize = 100;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::spp::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                             config));
}

TEST(skew_r_unique_s, naive) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

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

  int els_per_thread = 4;
  int threads_per_block = 512;
  classicjoin::Config config;
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
      classicjoin::naive::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                               config));
}

TEST(skew_r_unique_s, imv) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

  fmt::print(
      "Create relation R with {} tuples ({} MB) "
      "using zipf keys, skew= {} \n",
      r_n, r_n * sizeof(int32_t) / 1024 / 1024, skew);
  fmt::print(
      "Create relation S from R, with {} tuples ({} MB) "
      "using unique keys\n",
      s_n, s_n * sizeof(int32_t) / 1024 / 1024);

  //   fmt::print("R: {}\n", cutil::fmt_arr(r_key, r_n));
  //   fmt::print("S: {}\n", cutil::fmt_arr(s_key, s_n));

  int32_t *r_payload = new int32_t[r_n];
  int32_t *s_payload = new int32_t[s_n];

  // Payload set to equal with key
  std::copy_n(r_key, r_n, r_payload);
  std::copy_n(s_key, s_n, s_payload);

  //   fmt::print("R payload: {}\n", cutil::fmt_arr(r_payload, 20));
  //   fmt::print("S payload: {}\n", cutil::fmt_arr(s_payload, 20));

  const int threads_per_block = 128;
  const int els_per_thread = 1024;
  classicjoin::imv::ConfigIMV config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  {  // probe kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
    config.probe_gridsize = blocks_per_grid;
    config.probe_blocksize = threads_per_block;
  }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::imv::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                             config));
}

TEST(skew_r_unique_s, amac) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

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

  const int threads_per_block = 128;
  const int els_per_thread = 1024;
  classicjoin::amac::ConfigAMAC config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  {  // probe kernel
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
    config.probe_gridsize = blocks_per_grid;
    config.probe_blocksize = threads_per_block;
  }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::amac::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                              config));
}

TEST(skew_r_unique_s, gp) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

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

  classicjoin::gp::ConfigGP config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  // { // probe kernel
  //   const int els_per_block = threads_per_block * els_per_thread;
  //   const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //   config.probe_gridsize = blocks_per_grid;
  //   config.probe_blocksize = threads_per_block;
  // }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 144;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::gp::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                            config));
}

TEST(skew_r_unique_s, spp) {
  int32_t r_n = args::get<int32_t>("RN");
  int32_t s_n = args::get<int32_t>("SN");
  double skew = args::get<double>("SKEW");
  // assert(r_n <= s_n);
  std::string r_fname = cutil::rel_fname(false, "r", r_n, skew);
  std::string s_fname = cutil::rel_fname(true, "s", s_n, 0);
  int32_t *r_key = new int32_t[r_n];
  int32_t *s_key = new int32_t[s_n];

  // generate key = [0..r_n]
  assert(
      !datagen::create_relation_zipf(r_fname.c_str(), r_key, r_n, r_n, skew));
  assert(!datagen::create_relation_unique(s_fname.c_str(), s_key, s_n, s_n));

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

  classicjoin::spp::ConfigSPP config;
  //   {  // build kernel
  //     const int els_per_block = threads_per_block * els_per_thread;
  //     const int blocks_per_grid = (r_n + els_per_block - 1) / els_per_block;
  //     config.build_gridsize = blocks_per_grid;
  //     config.build_blocksize = threads_per_block;
  //   }
  // { // probe kernel
  //   const int els_per_block = threads_per_block * els_per_thread;
  //   const int blocks_per_grid = (s_n + els_per_block - 1) / els_per_block;
  //   config.probe_gridsize = blocks_per_grid;
  //   config.probe_blocksize = threads_per_block;
  // }
  config.build_blocksize = 128;
  config.build_gridsize = 72 * 2;
  config.probe_blocksize = 128;
  config.probe_gridsize = 72 * 2;

  fmt::print(
      "Query:\n"
      "\tSELECT SUM(R.payload*S.payload) FROM R JOIN S\n"
      "Result:\n"
      "\t{}\n",
      classicjoin::spp::join(r_key, r_payload, r_n, s_key, s_payload, s_n,
                             config));
}