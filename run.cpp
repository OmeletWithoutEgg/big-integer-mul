#include "bigint.h"
#include "mul_ntt_arm.hpp"
#include <vector>
#include <cstdint>
#include <random>
#include <iostream>

NTT<998244353, 3, (1 << 10)> ntt;
void test_correct() {
  const int sz = 16;
  std::vector<uint32_t> a = {1, 2};
  std::vector<uint32_t> b = {3, 4, 5};
  a.resize(sz);
  b.resize(sz);
  ntt.transform_forward(a.data(), sz);
  ntt.transform_forward(b.data(), sz);
  ntt.dot_product(a.data(), b.data(), sz);

  std::cerr << "safe\n";

  ntt.transform_inverse(a.data(), sz);
  for (int i = 0; i < sz; i++)
    std::cout << a[i] << ' ';
}

void test_speed(int argc, char **argv) {
  const int sz = 1 << 10;
  std::vector<uint32_t> a(sz);
  std::vector<uint32_t> b(sz);
  std::mt19937 rng(atoi(argv[1]));

  u32 sum = 0;
  for (int i = 0; i < 500; i++) {
    for (u32 &x : a) x = rng() % 998244353;
    for (u32 &x : b) x = rng() % 998244353;
    ntt.transform_forward(a.data(), sz);
    ntt.transform_forward(b.data(), sz);
    ntt.pointwise_product(a.data(), b.data(), sz);
    ntt.transform_inverse(a.data(), sz);
    for (int i = 0; i < sz; i++)
      sum += a[i];
  }
  std::cout << sum << std::endl;
}

int main(int argc, char **argv) {
  test_speed(argc, argv);
}
