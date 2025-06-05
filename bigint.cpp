#include "bigint.h"

#ifdef USE_SUPER_FAST_NTT
#include "mul_ntt.hpp"
#else
#include "mul_ntt_arm.hpp"
#endif

#include <bit>
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

  std::vector<uint32_t> va1(a->limbs, a->limbs + BIGINT_LIMBS);
  std::vector<uint32_t> vb1(b->limbs, b->limbs + BIGINT_LIMBS);
  auto va2 = va1;
  auto vb2 = vb1;
  auto va3 = va1;
  auto vb3 = vb1;

#ifdef USE_SUPER_FAST_NTT
  static super_fast_NTT::NTT ntt1(M1);
  static super_fast_NTT::NTT ntt2(M2);
  static super_fast_NTT::NTT ntt3(M3);

  ntt1.inplace_convolve<true, false>(va1, vb1);
  ntt2.inplace_convolve<true, false>(va2, vb2);
  ntt3.inplace_convolve<true, false>(va3, vb3);
#else
  constexpr int maxn = std::bit_ceil(u32(BIGINT_LIMBS));
  // std::cerr << "maxn = " << maxn << '\n';
  constexpr uint32_t G1 = 3, G2 = 3, G3 = 3;
  static NTT<M1, G1, maxn> ntt1;
  static NTT<M2, G2, maxn> ntt2;
  static NTT<M3, G3, maxn> ntt3;

  auto inplace_convolve = [&](auto &ntt, auto &va, auto &vb) {
    constexpr int n = BIGINT_LIMBS;
    const u32 mod = ntt.mont.mod, mod2 = mod * 2;
    // 
    for (u32 &x : va) {
      if (x >= mod2) x -= mod2;
      if (x >= mod) x -= mod;
      if (x >= mod) x -= mod;
    }
    for (u32 &x : vb) {
      if (x >= mod2) x -= mod2;
      if (x >= mod) x -= mod;
      if (x >= mod) x -= mod;
    }
    ntt.transform_forward(va.data(), n);
    ntt.transform_forward(vb.data(), n);
    ntt.pointwise_product(va.data(), vb.data(), n);
    // for (size_t i = 0; i < n; i++)
    //   va[i] = ntt.mont.redc(va[i], vb[i]);
    ntt.transform_inverse(va.data(), n);
  };
  inplace_convolve(ntt1, va1, vb1);
  inplace_convolve(ntt2, va2, vb2);
  inplace_convolve(ntt3, va3, vb3);
#endif

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
