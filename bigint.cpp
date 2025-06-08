#include "mul_ntt_arm.hpp"
#include "bigint.h"

#include <algorithm>
#include <bit>
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
  return static_cast<u32>(r);
}

using u128 = __uint128_t;
static constexpr uint32_t M1 = 880803841;
static constexpr uint32_t M2 = 897581057;
static constexpr uint32_t M3 = 998244353;
static_assert(countr_zero(M1 - 1) == 23 && countr_zero(M2 - 1) == 23 &&
              countr_zero(M3 - 1) == 23);
static_assert(BIGINT_LIMBS <= (1 << 23));
static_assert(M1 < M2 && M2 < M3);
static_assert((u128(BIGINT_LIMBS) << (LIMB_BITS * 2)) < u128(M1) * M2 * M3);
static constexpr uint64_t r12 = modpow(M1, M2 - 2, M2);
static constexpr uint64_t r13 = modpow(M1, M3 - 2, M3);
static constexpr uint64_t r23 = modpow(M2, M3 - 2, M3);
static constexpr uint64_t M1M2 = 1ULL * M1 * M2;
static NTT<M1> ntt1;
static NTT<M2> ntt2;
static NTT<M3> ntt3;

void recover_by_crt(bigint *res, uint32_t va1[], uint32_t va2[],
                    uint32_t va3[]) {
  u64 carry = 0;
  const size_t n = BIGINT_LIMBS;
  size_t i;
  for (i = 0; i < n; i++) {
    u64 A = va1[i];
    u64 B = va2[i];
    u64 C = va3[i];
    /* B = ntt2.mont.mul(ntt2.mont.sub(B, A), r12); */
    /* C = ntt3.mont.mul(ntt3.mont.sub(C, A), r13); */
    /* C = ntt3.mont.mul(ntt3.mont.sub(C, B), r23); */
    B = (B - A + M2) * r12 % M2;
    C = (C - A + M3) * r13 % M3;
    C = (C - B + M3) * r23 % M3;
    u64 lower = A + B * u64(M1) + u32(carry);
    u64 upper = (carry >> 32) + C * (M1M2 >> 32);
    upper += lower >> 32;
    lower = u32(lower);
    lower += C * u32(M1M2);
    res->limbs[i] = u32(lower);
    carry = upper + (lower >> 32);

    /* u128 sum = A + B * u128(M1) + C * u128(M1M2) + carry; */
    /* res->limbs[i] = sum; */
    /* carry = sum >> 32; */
  }
  assert(carry == 0);
}

// --- bigint 乘法 (base 2^LIMB_BITS) ---
void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  constexpr int n = BIGINT_LIMBS;
  ntt1.convolve(a->limbs, b->limbs, n);
  ntt2.convolve(a->limbs, b->limbs, n);
  ntt3.convolve(a->limbs, b->limbs, n);

  recover_by_crt(res, ntt1.buf1, ntt2.buf1, ntt3.buf1);
}
}
