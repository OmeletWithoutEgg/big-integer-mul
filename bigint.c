#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bigint.h"

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

typedef __uint128_t Long;

#define MOD 9097271247288401921ull
#define MAXN (1 << 18)
#define G 6

static uint64_t modadd(uint64_t a, uint64_t b) {
  return a + b >= MOD ? a + b - MOD : a + b;
}
static uint64_t modsub(uint64_t a, uint64_t b) {
  return a < b ? a - b + MOD : a - b;
}
static uint64_t modmul(Long a, Long b) {
  return (uint64_t)(a * b % MOD);
}
static uint64_t modpow(uint64_t e, uint64_t p) {
  uint64_t r = 1;
  while (p) {
    if (p & 1)
      r = modmul(r, e);
    e = modmul(e, e);
    p >>= 1;
  }
  return r;
}
static uint64_t modinv(uint64_t x) { return modpow(x, MOD - 2); }

static uint64_t fft_roots[MAXN];
static uint64_t fft_roots_prepared = 0;
static void prepare_fft_roots() {
  uint64_t r = modpow(G, (MOD - 1) / MAXN);
  for (int i = MAXN >> 1; i; i >>= 1) {
    fft_roots[i] = 1;
    for (int j = 1; j < i; j++)
      fft_roots[i + j] = modmul(fft_roots[i + j - 1], r);
    r = modmul(r, r);
  }
}

static void fft(uint64_t f[]) {
  const int n = BIGINT_LIMBS;
  if (!fft_roots_prepared) {
    prepare_fft_roots();
    fft_roots_prepared = 1;
  }
  for (int i = 0, j = 0; i < n; i++) {
    if (i < j) {
      uint64_t tmp = f[i];
      f[i] = f[j];
      f[j] = tmp;
    }
    for (int k = n >> 1; (j ^= k) < k; k >>= 1)
      ;
  }
  for (int s = 1; s < n; s *= 2)
    for (int i = 0; i < n; i += s * 2)
      for (int j = 0; j < s; j++) {
        uint64_t a = f[i + j], b = modmul(f[i + j + s], fft_roots[s + j]);
        f[i + j] = modadd(a, b);
        f[i + j + s] = modsub(a, b);
      }
}

static void ifft(uint64_t f[]) {
  fft(f);
  const int n = BIGINT_LIMBS;
  const uint64_t invn = modinv(n);
  for (int i = 0; i < n; i++)
    f[i] = modmul(f[i], invn);
  for (int i = 1; i < n - i; i++) {
    const int j = n - i;
    uint64_t tmp = f[i];
    f[i] = f[j];
    f[j] = tmp;
  }
}

void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  uint64_t a_hat[BIGINT_LIMBS] = {0};
  uint64_t b_hat[BIGINT_LIMBS] = {0};
  memcpy(a_hat, a->limbs, sizeof(a->limbs));
  memcpy(b_hat, b->limbs, sizeof(b->limbs));
  /* for (int i = 0; i < 10; i++) { */
  /*   fprintf(stderr, "%" PRIu32 " ", a_hat[i]); */
  /* } */
  /* fprintf(stderr, "\n"); */
  /* for (int i = 0; i < 10; i++) { */
  /*   fprintf(stderr, "%" PRIu32 " ", b_hat[i]); */
  /* } */
  /* fprintf(stderr, "\n"); */
  /* fprintf(stderr, "BIGINT_LIMBS = %d\n", BIGINT_LIMBS); */

  fft(a_hat);
  fft(b_hat);

  for (size_t i = 0; i < BIGINT_LIMBS; i++) {
    a_hat[i] = modmul(a_hat[i], b_hat[i]);
  }

  ifft(a_hat);

  // Carry propagation（以 2^LIMB_BITS 為 base）
  for (size_t i = 0; i < BIGINT_LIMBS - 1; i++) {
    a_hat[i + 1] += a_hat[i] >> LIMB_BITS;
    a_hat[i] &= ((1ull << LIMB_BITS) - 1);
  }

  memcpy(res->limbs, a_hat, sizeof(res->limbs));
}
