#ifndef BIGINT_H
#define BIGINT_H

#include <stdint.h>

#define BIGINT_BITS (1 << 23)
#define LIMB_BITS 32
#define BIGINT_LIMBS ((BIGINT_BITS + LIMB_BITS - 1) / LIMB_BITS)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t limbs[BIGINT_LIMBS];
} bigint;

void bigint_urandom(uint64_t *seed, bigint *r, uint32_t bits);
void bigint_mul(bigint *res, const bigint *a, const bigint *b);

#ifdef __cplusplus
}
#endif

#endif // BIGINT_H
