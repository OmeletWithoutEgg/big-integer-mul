#include "bigint.h"
#include "mul_ntt.hpp"
#include <cstring>

extern "C" {

void bigint_set_zero(bigint *r) { memset(r->limbs, 0, sizeof(r->limbs)); }

void bigint_copy(bigint *dest, const bigint *src) {
  memcpy(dest->limbs, src->limbs, sizeof(src->limbs));
}

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

// --- bigint 乘法 (base 2^LIMB_BITS) ---
void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  BigInteger A{std::vector(a->limbs, a->limbs + BIGINT_LIMBS / 2)};
  BigInteger B{std::vector(b->limbs, b->limbs + BIGINT_LIMBS / 2)};
  A *= B;
  memset(res->limbs, 0, sizeof(res->limbs));
  for (size_t i = 0; i < A.data_.size(); i++)
    res->limbs[i] = A.data_[i];
}
}
