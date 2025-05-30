#ifndef BIGINT_H
#define BIGINT_H

#include <stdint.h>

#define BIGINT_BITS     1024
#define LIMB_BITS       30
#define BIGINT_LIMBS    (BIGINT_BITS / LIMB_BITS)

typedef struct {
    uint64_t limbs[BIGINT_LIMBS];
} bigint;

void bigint_set_zero(bigint *r);
void bigint_copy(bigint *dest, const bigint *src);
void bigint_urandom(uint64_t *seed, bigint *r);
void bigint_mul(bigint *res, const bigint *a, const bigint *b);
/* void bigint_print(const bigint *x); */

#endif
