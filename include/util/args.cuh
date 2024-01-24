#pragma once
#include <assert.h>
#include <fmt/format.h>
// #include "cxxopts.hpp"

// auto parse_args(int argc, char **argv) {
//   cxxopts::Options options("GPU Hash Join",
//                            "Joins relation R (build) and S (probe) on GPU");
//   options.add_options()                         //
//       ("R,RN", "Number of rows in relation R")  //
//       ("S,SN", "Number of rows in relation S")  //
//       ("s,skew", "Skewness factor on zipf distribution, only for relation
//       S");
//   auto result = options.parse(argc, argv);
//   return result;
// }

namespace args {
template <typename T>
T get(const char *name) {
  char *str = getenv(name);
  assert(str);
  fmt::print("get_arg unimplemented");
  return 0;
}

template <>
int get<int>(const char *name) {
  char *str = getenv(name);
  assert(str);
  return std::stoi(str);
}

template <>
double get<double>(const char *name) {
  char *str = getenv(name);
  assert(str);
  return std::stod(str);
}
}  // namespace args
