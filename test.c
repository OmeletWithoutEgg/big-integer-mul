#include <assert.h>
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>

#include "bigint.h"

// 將 bigint 轉為 mpz_t（假設 base-2^LIMB_BITS little-endian）
static void bigint_to_mpz(mpz_t rop, const bigint *x) {
  mpz_set_ui(rop, 0);
  for (int i = BIGINT_LIMBS - 1; i >= 0; i--) {
    mpz_mul_2exp(rop, rop, LIMB_BITS); // rop *= 2^LIMB_BITS
    mpz_add_ui(rop, rop, x->limbs[i]);
  }
}

// 測試 bigint_mul 與 GMP 的乘法是否一致
int main() {
  int i;
  uint64_t seed = 0x12345678;
  bigint a, b, res;
  mpz_t za, zb, zmul, zres;

  mpz_inits(za, zb, zmul, zres, NULL);

#define NTEST 100
  for (i = 0; i < NTEST; i++) {

    bigint_urandom(&seed, &a, 1 << 17);
    bigint_urandom(&seed, &b, 1 << 17);
    bigint_mul(&res, &a, &b);

    // GMP 計算
    bigint_to_mpz(za, &a);
    bigint_to_mpz(zb, &b);
    mpz_mul(zmul, za, zb);
    bigint_to_mpz(zres, &res);

    if (mpz_cmp(zmul, zres) == 0) {
      printf("[PASS] bigint_mul matches GMP\n");
    } else {
      printf("[FAIL] bigint_mul does NOT match GMP\n");
      gmp_printf("Expected: %Zd\n", zmul);
      gmp_printf("Got     : %Zd\n", zres);
      assert(0 && "bigint_mul test failed");
    }
  }
  mpz_clears(za, zb, zmul, zres, NULL);
}
