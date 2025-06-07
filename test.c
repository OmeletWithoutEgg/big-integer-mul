#include <assert.h>
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>

#include "bigint.h"

static void bigint_to_mpz(mpz_t rop, const bigint *x) {
  mpz_import(
    rop,
    BIGINT_LIMBS,    // word count
    -1,              // order: -1 = least significant word first (little-endian)
    sizeof(x->limbs[0]), // size of each word
    0,               // endianness of words: 0 = native
    0,               // nails (bits to skip): 0
    x->limbs         // input data
  );
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

    bigint_urandom(&seed, &a, 1 << 16);
    bigint_urandom(&seed, &b, 1 << 16);
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
