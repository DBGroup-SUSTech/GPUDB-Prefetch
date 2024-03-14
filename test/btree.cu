#include <fmt/format.h>
#include <gtest/gtest.h>

#include "btree/amac.cuh"
#include "btree/common.cuh"
#include "btree/gp-sep.cuh"
#include "btree/gp.cuh"
#include "btree/imv.cuh"
#include "btree/naive.cuh"
#include "btree/spp.cuh"
#include "datagen/generator_ETHZ.cuh"
#include "util/args.cuh"

TEST(unique, naive) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 8;
  int threads_per_block = 512;
  btree::Config config;

  {
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }

  {
    const int els_per_block = (threads_per_block / 32) *
                              btree::naive::LANES_PER_WARP * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.probe_gridsize = blocks_per_grid;
    config.probe_blocksize = threads_per_block;
  }
  btree::naive::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}

TEST(unique, gp) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 8;  // 8/32 per warp is the best
  int threads_per_block = 512;
  btree::Config config;

  {
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }

  {
    config.probe_gridsize = args::get<int>("GS");   // 144
    config.probe_blocksize = args::get<int>("BS");  // 128
  }
  btree::gp::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}


TEST(unique, spp) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 8;  // 8/32 per warp is the best
  int threads_per_block = 512;
  btree::Config config;

  {
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }

  {
    config.probe_gridsize = args::get<int>("GS");   // 144
    config.probe_blocksize = args::get<int>("BS");  // 128
  }
  btree::spp::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}

/*
TEST(unique, spp) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 8;     //
  int threads_per_block = 8;  //
  btree::Config config;

  const int els_per_block = threads_per_block * els_per_thread;
  const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
  config.build_gridsize = blocks_per_grid;
  config.build_blocksize = threads_per_block;
  config.probe_gridsize = blocks_per_grid;
  config.probe_blocksize = threads_per_block;
  btree::spp::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}

*/

TEST(unique, amac) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 4;
  int threads_per_block = 256;
  btree::amac::ConfigAMAC config;
  {
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }
  // config.probe_gridsize = blocks_per_grid;
  // config.probe_blocksize = threads_per_block;
  // config.probe_blocksize = 128;
  // config.probe_gridsize = 72 * 2;
  // config.probe_gridsize = 1;
  {
    config.probe_gridsize = args::get<int>("GS");   // 144
    config.probe_blocksize = args::get<int>("BS");  // 128
  }
  btree::amac::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}

TEST(unique, imv) {
  int32_t n = args::get<int32_t>("N");
  std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
  std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
  int32_t *keys = new int32_t[n];
  int32_t *values = new int32_t[n];

  assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
  assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

  fmt::print("Creating {} unique keys ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);
  fmt::print("Creating {} unique values ({} MB)\n", n,
             n * sizeof(int32_t) / 1024 / 1024);

  int els_per_thread = 4;
  int threads_per_block = 256;
  btree::Config config;
  {
    const int els_per_block = threads_per_block * els_per_thread;
    const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
    config.build_gridsize = blocks_per_grid;
    config.build_blocksize = threads_per_block;
  }
  // config.probe_gridsize = blocks_per_grid;
  // config.probe_blocksize = threads_per_block;
  // config.probe_blocksize = 128;
  // config.probe_gridsize = 72 * 2;
  // config.probe_gridsize = 1;
  {
    config.probe_gridsize = args::get<int>("GS");   // 144
    config.probe_blocksize = args::get<int>("BS");  // 128
  }
  btree::imv::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}

// TEST(unique, gp_sep) {
//   int32_t n = args::get<int32_t>("N");
//   std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
//   std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
//   int64_t *keys = new int64_t[n];
//   int64_t *values = new int64_t[n];  // ? keys == values

//   assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
//   assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

//   fmt::print("Creating {} unique keys ({} MB)\n", n,
//              n * sizeof(int64_t) / 1024 / 1024);
//   fmt::print("Creating {} unique values ({} MB)\n", n,
//              n * sizeof(int64_t) / 1024 / 1024);

//   int els_per_thread = 8;     // group size of GP ?
//   int threads_per_block = 8;  //?
//   btree::Config config;

//   const int els_per_block = threads_per_block * els_per_thread;
//   const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
//   config.build_gridsize = blocks_per_grid;
//   config.build_blocksize = threads_per_block;
//   config.probe_gridsize = blocks_per_grid;
//   config.probe_blocksize = threads_per_block;
//   btree::gp_sep::index(keys, values, n, config);
//   fmt::print("Insert and Lookup {} tuples into BTree\n", n);
// }

// TEST(unique, spp) {
//   int32_t n = args::get<int32_t>("N");
//   std::string key_fname = cutil::rel_fname(true, "key_uniq", n, 0);
//   std::string val_fname = cutil::rel_fname(true, "val_uniq", n, 0);
//   int64_t *keys = new int64_t[n];
//   int64_t *values = new int64_t[n];

//   assert(!datagen::create_relation_unique(key_fname.c_str(), keys, n, n));
//   assert(!datagen::create_relation_unique(val_fname.c_str(), values, n, n));

//   fmt::print("Creating {} unique keys ({} MB)\n", n,
//              n * sizeof(int64_t) / 1024 / 1024);
//   fmt::print("Creating {} unique values ({} MB)\n", n,
//              n * sizeof(int64_t) / 1024 / 1024);

//   int els_per_thread = 8;     //
//   int threads_per_block = 8;  //
//   btree::Config config;

//   const int els_per_block = threads_per_block * els_per_thread;
//   const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
//   config.build_gridsize = blocks_per_grid;
//   config.build_blocksize = threads_per_block;
//   config.probe_gridsize = blocks_per_grid;
//   config.probe_blocksize = threads_per_block;
//   btree::spp::index(keys, values, n, config);
//   fmt::print("Insert and Lookup {} tuples into BTree\n", n);
// }
// * /