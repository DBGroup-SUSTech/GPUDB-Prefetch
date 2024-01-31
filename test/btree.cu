#include <fmt/format.h>
#include <gtest/gtest.h>

#include "btree/common.cuh"
#include "btree/naive.cuh"
#include "datagen/generator_ETHZ.cuh"
#include "util/args.cuh"

TEST(unique, naive) {
  // btree::InnerNode node;
  // printf("%p, %p\n", (char *)&node.n_key - (char *)&node,
  //        (char *)&node.version_lock_obsolete - (char *)&node);
  // return;

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

  const int els_per_block = threads_per_block * els_per_thread;
  const int blocks_per_grid = (n + els_per_block - 1) / els_per_block;
  config.build_gridsize = blocks_per_grid;
  config.build_blocksize = threads_per_block;
  config.probe_gridsize = blocks_per_grid;
  config.probe_blocksize = threads_per_block;
  btree::naive::index(keys, values, n, config);
  fmt::print("Insert and Lookup {} tuples into BTree\n", n);
}