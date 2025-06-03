#include "bigint.h"
#include "mul_ntt.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

std::ostream &operator<<(std::ostream &o, __uint128_t x) {
  if (x < 10)
    return o << int(x);
  return o << x / 10 << int(x % 10);
}

extern "C" {

static uint64_t xorshift64(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

void bigint_urandom(uint64_t *seed, bigint *r, uint32_t bits) {
  assert(bits % LIMB_BITS == 0);
  assert(bits * 2 <= BIGINT_BITS);
  for (size_t i = 0; i < bits / LIMB_BITS; i++) {
    r->limbs[i] = (xorshift64(seed) & ((1ull << LIMB_BITS) - 1));
  }
  for (size_t i = bits / LIMB_BITS; i < BIGINT_LIMBS; i++) {
    r->limbs[i] = 0;
  }
}
static constexpr u32 modpow(u64 e, u32 p, u32 mod) {
  u64 r = 1;
  while (p) {
    if (p & 1)
      r = r * e % mod;
    e = e * e % mod;
    p >>= 1;
  }
  return r;
}

// --- bigint 乘法 (base 2^LIMB_BITS) ---
void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  constexpr uint32_t M1 = 985661441; // G = 3 for M1, M2, M3
  constexpr uint32_t M2 = 998244353;
  constexpr uint32_t M3 = 1004535809;
  static_assert(M1 < M2 && M2 < M3);
  constexpr uint64_t r12 = modpow(M1, M2 - 2, M2);
  constexpr uint64_t r13 = modpow(M1, M3 - 2, M3);
  constexpr uint64_t r23 = modpow(M2, M3 - 2, M3);
  constexpr uint64_t M1M2 = 1ULL * M1 * M2;

  constexpr int maxn = 1 << 20;
  static super_fast_NTT::NTT ntt1(M1);
  static super_fast_NTT::NTT ntt2(M2);
  static super_fast_NTT::NTT ntt3(M3);

  std::vector<uint32_t> va1(a->limbs, a->limbs + BIGINT_LIMBS);
  std::vector<uint32_t> vb1(b->limbs, b->limbs + BIGINT_LIMBS);
  auto va2 = va1;
  auto vb2 = vb1;
  auto va3 = va1;
  auto vb3 = vb1;

  ntt1.inplace_convolve<true, false>(va1, vb1);
  ntt2.inplace_convolve<true, false>(va2, vb2);
  ntt3.inplace_convolve<true, false>(va3, vb3);

  memset(res->limbs, 0, sizeof(res->limbs));
  using u128 = __uint128_t;
  u128 carry = 0;
  size_t i;
  for (i = 0; i < va1.size(); i++) {
    uint64_t A = va1[i];
    uint64_t B = va2[i];
    uint64_t C = va3[i];
    B = (B - A + M2) * r12 % M2;
    C = (C - A + M3) * r13 % M3;
    C = (C - B + M3) * r23 % M3;

    u128 sum = A + B * u128(M1) + C * u128(M1M2) + carry;
    res->limbs[i] = sum;
    carry = sum >> 32;
  }
  while (carry) {
    assert(i < BIGINT_LIMBS);
    res->limbs[i++] = carry;
    carry >>= 32;
  }
}
}
