/*
 * Copyright (c) 2024-2025 The mlkem-native project authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>
#include <time.h>

#include "bigint.h"

#include "hal.h"

/* #define NWARMUP 50 */
/* #define NITERATIONS 300 */
/* #define NTESTS 500 */

#define NWARMUP 10
#define NITERATIONS 60
#define NTESTS 100

static int cmp_uint64_t(const void *a, const void *b) {
  uint64_t va = (*((const uint64_t *)a));
  uint64_t vb = (*((const uint64_t *)b));
  if (va < vb)
    return -1;
  else if (va > vb)
    return 1;
  else
    return 0;
}

static void print_median(const char *text, uint64_t cyc[NTESTS]) {
  printf("%10s cycles = %" PRIu64 "\n", text, cyc[NTESTS >> 1] / NITERATIONS);
}

static int percentiles[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};

static void print_percentile_legend(void) {
  unsigned i;
  printf("%21s", "percentile");
  for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++) {
    printf("%10d", percentiles[i]);
  }
  printf("\n");
}

static void print_percentiles(const char *text, uint64_t cyc[NTESTS]) {
  unsigned i;
  printf("%10s percentiles:", text);
  for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++) {
    printf("%10" PRIu64, (cyc)[NTESTS * percentiles[i] / 100] / NITERATIONS);
  }
  printf("\n");
}

#define BENCH_SIZE (BIGINT_BITS / 2)

static int bench_mpz(void) {
  int i, j;
  uint64_t t0, t1;
  uint64_t cycles_mpz[NTESTS];

  gmp_randstate_t state;
  mpz_t a, b, result;

  // 初始化 GMP 的整數變數
  mpz_init(a);
  mpz_init(b);
  mpz_init(result);

  // 初始化隨機狀態，使用當前時間作為 seed
  gmp_randinit_default(state);
  gmp_randseed_ui(state, (unsigned long int)time(NULL));

  for (i = 0; i < NTESTS; i++) {
    // 產生兩個 BENCH_SIZE 位元的隨機大整數（base 2）
    mpz_urandomb(a, state, BENCH_SIZE);
    mpz_urandomb(b, state, BENCH_SIZE);

    for (j = 0; j < NWARMUP; j++) {
      mpz_mul(result, a, b);
    }

    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++) {
      mpz_mul(result, a, b);
    }
    t1 = get_cyclecounter();
    cycles_mpz[i] = t1 - t0;
  }

  qsort(cycles_mpz, NTESTS, sizeof(uint64_t), cmp_uint64_t);
  print_median("mpz", cycles_mpz);
  printf("\n");
  print_percentile_legend();
  print_percentiles("mpz", cycles_mpz);

  mpz_clear(a);
  mpz_clear(b);
  mpz_clear(result);
  gmp_randclear(state);

  return 0;
}

static int bench_mul(void) {
  int i, j;
  uint64_t t0, t1;
  uint64_t cycles_mul[NTESTS];
  uint64_t seed = (uint64_t)time(NULL);
  bigint a, b, result;

  for (i = 0; i < NTESTS; i++) {
    bigint_urandom(&seed, &a, BENCH_SIZE);
    bigint_urandom(&seed, &b, BENCH_SIZE);

    for (j = 0; j < NWARMUP; j++) {
      bigint_mul(&result, &a, &b);
    }

    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++) {
      bigint_mul(&result, &a, &b);
    }
    t1 = get_cyclecounter();
    cycles_mul[i] = t1 - t0;
  }

  qsort(cycles_mul, NTESTS, sizeof(uint64_t), cmp_uint64_t);
  print_median("mul", cycles_mul);
  printf("\n");
  print_percentile_legend();
  print_percentiles("mul", cycles_mul);

  return 0;
}

int main(void) {
  printf("BENCH_SIZE = (1 << %d) = %d\n", __builtin_ctz(BENCH_SIZE), BENCH_SIZE);
  enable_cyclecounter();
  bench_mpz();
  bench_mul();
  disable_cyclecounter();

  return 0;
}
