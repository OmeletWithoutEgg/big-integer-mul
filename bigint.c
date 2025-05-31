#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bigint.h"

// --- 基本工具函數 ---

void bigint_set_zero(bigint *r) { memset(r->limbs, 0, sizeof(r->limbs)); }

void bigint_copy(bigint *dest, const bigint *src) {
  memcpy(dest->limbs, src->limbs, sizeof(src->limbs));
}

// --- 簡單 XORSHIFT64 亂數產生器 ---

static uint64_t xorshift64(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

void bigint_urandom(uint64_t *seed, bigint *r) {
  for (size_t i = 0; i < BIGINT_LIMBS / 2; i++) {
    r->limbs[i] = (xorshift64(seed) & ((1ull << LIMB_BITS) - 1));
  }
  for (size_t i = BIGINT_LIMBS / 2; i < BIGINT_LIMBS; i++) {
    r->limbs[i] = 0;
  }
}

// --- bigint 乘法 (base 2^LIMB_BITS) ---

void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  uint64_t tmp[BIGINT_LIMBS] = {0};

  for (size_t i = 0; i < BIGINT_LIMBS; i++) {
    for (size_t j = 0; j < BIGINT_LIMBS; j++) {
      tmp[i + j] += a->limbs[i] * b->limbs[j];
    }
  }

  // Carry propagation（以 2^LIMB_BITS 為 base）
  for (size_t i = 0; i < BIGINT_LIMBS - 1; i++) {
    tmp[i + 1] += tmp[i] >> LIMB_BITS;
    tmp[i] &= ((1ull << LIMB_BITS) - 1);
  }

  for (size_t i = 0; i < BIGINT_LIMBS; i++) {
    res->limbs[i] = tmp[i];
  }
}

/* // --- 印出 bigint（以 16 進位表示） --- */

/* void bigint_print(const bigint *x) { */
/*     for (int i = BIGINT_LIMBS - 1; i >= 0; i--) { */
/*         printf("%04x", x->limbs[i]); */
/*     } */
/*     printf("\n"); */
/* } */
