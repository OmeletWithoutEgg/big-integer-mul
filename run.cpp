#include "bigint.h"
#include "mul_ntt_arm.hpp"
#include <vector>
#include <cstdint>
#include <random>
#include <iostream>

NTT<998244353, 3> ntt;
void test_ntt_correctness() {
  const int sz = 32;
  std::vector<uint32_t> a = {1, 2};
  std::vector<uint32_t> b = {3, 4, 5, 6, 7, 8};
  a.resize(sz);
  b.resize(sz);
  ntt.convolve(a.data(), b.data(), sz);

  std::cerr << "ntt output:";
  for (int i = 0; i < sz; i++)
    std::cout << ntt.buf1[i] << ' ';
  std::cout << '\n';
}

void test_montgomery_simd_correctness() {
  constexpr u32 mod = 998244353;
  constexpr int N = 100;

  using Mont = Montgomery<mod>;

  std::mt19937 rng(42);
  std::uniform_int_distribution<u32> dist(0, mod - 1);

  // 測試用隨機資料
  u32 a_arr[N], b_arr[N], expected[N], result[N];

  for (int i = 0; i < N; ++i) {
    a_arr[i] = dist(rng);
    b_arr[i] = dist(rng);
  }

  // 標準 scalar Montgomery 計算
  for (int i = 0; i < N; ++i) {
    u32 a_mont = Mont::from(a_arr[i]);
    u32 b_mont = Mont::from(b_arr[i]);
    u32 prod = Mont::redc(a_mont, b_mont);
    expected[i] = Mont::get(prod);
  }

  // SIMD 每 4 個為一組處理
  for (int i = 0; i < N; i += 4) {
    uint32x4_t a_vec = vld1q_u32(&a_arr[i]);
    uint32x4_t b_vec = vld1q_u32(&b_arr[i]);

    uint32x4_t a_mont_vec = Mont::from_32x4(a_vec);
    uint32x4_t b_mont_vec = Mont::from_32x4(b_vec);
    uint32x4_t prod_vec = Mont::redc_32x4(a_mont_vec, b_mont_vec);
    uint32x4_t res_vec = Mont::get_32x4(prod_vec);

    vst1q_u32(&result[i], res_vec);
  }

  // 檢查結果
  for (int i = 0; i < N; ++i) {
    if (result[i] != expected[i]) {
      printf("Mismatch at %d: a=%u b=%u → SIMD=%u, scalar=%u\n",
             i, a_arr[i], b_arr[i], result[i], expected[i]);
      assert(false);
    }
  }

  puts("All SIMD vs scalar Montgomery results match!");
}


void test_context_montgomery() {
  srand(time(0));
  constexpr u32 mod = 998244353;
  using Mont = Montgomery<mod>;

  for (int test = 0; test < 1000; ++test) {
    u32 b = rand() % mod;
    typename Mont::MulConstContext ctx{b};

    u32 input[4];
    for (int i = 0; i < 4; ++i) input[i] = rand() % mod;
    uint32x4_t a = vld1q_u32(input);

    uint32x4_t simd_result = Mont::redc_32x4_by_context(a, ctx);

    u32 simd_output[4];
    vst1q_u32(simd_output, simd_result);

    for (int i = 0; i < 4; ++i) {
      u32 expected = Mont::redc(input[i], b);
      if (simd_output[i] != expected) {
        printf("Mismatch at i=%d: expected %u, got %u\n", i, expected, simd_output[i]);
      }
      assert(simd_output[i] == expected);
    }
  }

  puts("All tests passed!");
}

void test_speed(int argc, char **argv) {
  const int sz = 1 << 12;
  std::vector<uint32_t> a(sz);
  std::vector<uint32_t> b(sz);
  std::mt19937 rng(atoi(argv[1]));

  u32 sum = 0;
  for (int i = 0; i < 50; i++) {
    for (u32 &x : a) x = rng() % 998244353;
    for (u32 &x : b) x = rng() % 998244353;
    ntt.convolve(a.data(), b.data(), sz);
    for (int i = 0; i < sz; i++)
      sum += ntt.buf1[i];
  }
  std::cout << "test_speed: sum = ";
  std::cout << sum << std::endl;
}

int main(int argc, char **argv) {
  /* test_ntt_correctness(); */
  test_montgomery_simd_correctness();
  test_context_montgomery();
  test_speed(argc, argv);
}
