#include <bits/allocator.h>

#include <cstdint>

#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <compare>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

namespace simd {
using i128 = __m128i;
using i256 = __m256i;
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u64 __attribute__((vector_size(32)));

u32x8 load_u32x8(u32 *ptr) { return (u32x8)(_mm256_load_si256((i256 *)ptr)); }
u32x8 loadu_u32x8(u32 *ptr) { return (u32x8)(_mm256_loadu_si256((i256 *)ptr)); }
void store_u32x8(u32 *ptr, u32x8 val) {
  _mm256_store_si256((i256 *)ptr, (i256)(val));
}
void storeu_u32x8(u32 *ptr, u32x8 val) {
  _mm256_storeu_si256((i256 *)ptr, (i256)(val));
}

u64x4 load_u64x4(u64 *ptr) { return (u64x4)(_mm256_load_si256((i256 *)ptr)); }
u64x4 loadu_u64x4(u64 *ptr) { return (u64x4)(_mm256_loadu_si256((i256 *)ptr)); }
void store_u64x4(u64 *ptr, u64x4 val) {
  _mm256_store_si256((i256 *)ptr, (i256)(val));
}
void storeu_u64x4(u64 *ptr, u64x4 val) {
  _mm256_storeu_si256((i256 *)ptr, (i256)(val));
}

u32x8 set1_u32x8(u32 val) { return (u32x8)(_mm256_set1_epi32(val)); }
u64x4 set1_u64x4(u64 val) { return (u64x4)(_mm256_set1_epi64x(val)); }

u32x8 setr_u32x8(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5, u32 a6,
                 u32 a7) {
  return (u32x8)(_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7));
}
u64x4 setr_u64x4(u64 a0, u64 a1, u64 a2, u64 a3) {
  return (u64x4)(_mm256_setr_epi64x(a0, a1, a2, a3));
}

template <int imm8> u32x8 shuffle_u32x8(u32x8 val) {
  return (u32x8)(_mm256_shuffle_epi32((i256)(val), imm8));
}
u32x8 permute_u32x8(u32x8 val, u32x8 p) {
  return (u32x8)(_mm256_permutevar8x32_epi32((i256)(val), (i256)(p)));
}

template <int imm8> u32x8 permute_u32x8_epi128(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_permute2x128_si256((i256)(a), (i256)(b), imm8));
}

template <int imm8> u32x8 blend_u32x8(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_blend_epi32((i256)(a), (i256)(b), imm8));
}

template <int imm8> u32x8 shift_left_u32x8_epi128(u32x8 val) {
  return (u32x8)(_mm256_bslli_epi128((i256)(val), imm8));
}
template <int imm8> u32x8 shift_right_u32x8_epi128(u32x8 val) {
  return (u32x8)(_mm256_bsrli_epi128((i256)(val), imm8));
}

u32x8 shift_left_u32x8_epi64(u32x8 val, int imm8) {
  return (u32x8)(_mm256_slli_epi64((i256)(val), imm8));
}
u32x8 shift_right_u32x8_epi64(u32x8 val, int imm8) {
  return (u32x8)(_mm256_srli_epi64((i256)(val), imm8));
}

u32x8 min_u32x8(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_min_epu32((i256)(a), (i256)(b)));
}
u32x8 mul64_u32x8(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_mul_epu32((i256)(a), (i256)(b)));
}

u32x8 add_u32x8(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_add_epi32((i256)(a), (i256)(b)));
}
u64x4 add_u64x4(u64x4 a, u64x4 b) {
  return (u64x4)(_mm256_add_epi32((i256)(a), (i256)(b)));
}

u32x8 sub_u32x8(u32x8 a, u32x8 b) {
  return (u32x8)(_mm256_sub_epi32((i256)(a), (i256)(b)));
}
}; // namespace simd
using namespace simd;

struct Montgomery {
  uint32_t mod, mod2, n_inv, r, r2;

  Montgomery() = default;
  Montgomery(uint32_t mod) : mod(mod) {
    assert(mod % 2);
    assert(mod < (1 << 30));
    n_inv = -mod & 3;
    for (int i = 0; i < 4; i++) {
      n_inv *= 2u + n_inv * mod;
    }
    assert(n_inv * mod == -1u);

    mod2 = 2 * mod;
    r = static_cast<uint32_t>((1ULL << 32) % mod);
    r2 = static_cast<uint32_t>(r * uint64_t(r) % mod);
  }

  uint32_t shrink(uint32_t val) const { return std::min(val, val - mod); }
  uint32_t shrink2(uint32_t val) const { return std::min(val, val - mod2); }
  uint32_t shrink_n(uint32_t val) const { return std::min(val, val + mod); }
  uint32_t shrink2_n(uint32_t val) const { return std::min(val, val + mod2); }

  template <bool strict = false> uint32_t reduce(uint64_t val) const {
    uint32_t res = (val + uint32_t(val) * n_inv * uint64_t(mod)) >> 32;
    if constexpr (strict)
      res = shrink(res);
    return res;
  }

  template <bool strict = false> uint32_t mul(uint32_t a, uint32_t b) const {
    uint64_t val = uint64_t(a) * b;
    uint32_t res = (val + uint32_t(val) * n_inv * uint64_t(mod)) >> 32;
    if constexpr (strict)
      res = shrink(res);
    return res;
  }

  template <bool input_in_space = false, bool in_space_res = true>
  uint32_t power(uint32_t b, uint32_t e) const {
    if constexpr (!input_in_space)
      b = mul<true>(b, r2);
    uint32_t res = (in_space_res ? r : 1);
    for (; e > 0; e >>= 1) {
      if (e & 1)
        res = mul(res, b);
      b = mul(b, b);
    }

    res = shrink(res);
    return res;
  }

  template <bool input_in_space = false, bool in_space_res = true>
  uint32_t inv(uint32_t a) const {
    return power<input_in_space, in_space_res>(a, mod - 2);
  }
};

struct Montgomery_simd {
  alignas(32) u32x8 mod;
  alignas(32) u32x8 mod2;
  alignas(32) u32x8 n_inv;
  alignas(32) u32x8 r;
  alignas(32) u32x8 r2;

  Montgomery_simd() = default;
  Montgomery_simd(u32 md) {
    Montgomery mt(md);
    mod = set1_u32x8(mt.mod);
    mod2 = set1_u32x8(mt.mod2);
    n_inv = set1_u32x8(mt.n_inv);
    r = set1_u32x8(mt.r);
    r2 = set1_u32x8(mt.r2);
  }

  u32x8 shrink(u32x8 val) const { return min_u32x8(val, val - mod); }
  u32x8 shrink2(u32x8 val) const { return min_u32x8(val, val - mod2); }
  u32x8 shrink_n(u32x8 val) const { return min_u32x8(val, val + mod); }
  u32x8 shrink2_n(u32x8 val) const { return min_u32x8(val, val + mod2); }

  template <bool strict = false> u64x4 reduce(u64x4 val) const {
    val = (u64x4)shift_right_u32x8_epi64(
        u32x8(val + (u64x4)mul64_u32x8(mul64_u32x8((u32x8)val, n_inv), mod)),
        32);
    if constexpr (strict) {
      val = (u64x4)shrink((u32x8)val);
    }
    return val;
  }

  template <bool strict = false> u32x8 reduce(u64x4 x0246, u64x4 x1357) const {
    u32x8 x0246_ninv = mul64_u32x8((u32x8)x0246, n_inv);
    u32x8 x1357_ninv = mul64_u32x8((u32x8)x1357, n_inv);
    u32x8 res = blend_u32x8<0b10'10'10'10>(
        shift_right_u32x8_epi128<4>(
            u32x8((u64x4)x0246 + (u64x4)mul64_u32x8(x0246_ninv, mod))),
        u32x8((u64x4)x1357 + (u64x4)mul64_u32x8(x1357_ninv, mod)));
    if constexpr (strict)
      res = shrink(res);
    return res;
  }

  template <bool strict = false, bool eq_b = false>
  u32x8 mul(u32x8 a, u32x8 b) const {
    u32x8 x0246 = mul64_u32x8(a, b);
    u32x8 b_sh = b;
    if constexpr (!eq_b) {
      b_sh = shift_right_u32x8_epi128<4>(b);
    }
    u32x8 x1357 = mul64_u32x8(shift_right_u32x8_epi128<4>(a), b_sh);

    return reduce<strict>((u64x4)x0246, (u64x4)x1357);
  }

  template <bool strict = false> u64x4 mul_to_hi(u64x4 a, u64x4 b) const {
    u32x8 val = mul64_u32x8((u32x8)a, (u32x8)b);
    u32x8 val_ninv = mul64_u32x8(val, n_inv);
    u32x8 res = u32x8(u64x4(val) + u64x4(mul64_u32x8(val_ninv, mod)));
    if constexpr (strict)
      res = shrink(res);
    return (u64x4)res;
  }

  template <bool strict = false> u64x4 mul(u64x4 a, u64x4 b) const {
    u32x8 val = mul64_u32x8((u32x8)a, (u32x8)b);
    return reduce<strict>((u64x4)val);
  }
};

namespace super_fast_NTT {
#pragma GCC target("avx2,bmi")

using u32 = uint32_t;
using u64 = uint64_t;

struct Montgomery {
  u32 mod;   // mod
  u32 mod2;  // 2 * mod
  u32 n_inv; // n_inv * mod == -1 (mod 2^32)
  u32 r;     // 2^32 % mod
  u32 r2;    // (2^32)^2 % mod

  Montgomery() = default;
  Montgomery(u32 mod) : mod(mod) {
    assert(mod % 2 == 1);
    assert(mod < (1 << 30));
    mod2 = 2 * mod;
    n_inv = 1;
    for (int i = 0; i < 5; i++) {
      n_inv *= 2 + n_inv * mod;
    }
    r = (u64(1) << 32) % mod;
    r2 = u64(r) * r % mod;
  }

  u32 shrink(u32 val) const { return std::min(val, val - mod); }
  u32 shrink2(u32 val) const { return std::min(val, val - mod2); }

  template <bool strict = true> u32 reduce(u64 val) const {
    u32 res = val + u32(val) * n_inv * u64(mod) >> 32;
    if constexpr (strict)
      res = shrink(res);
    return res;
  }

  template <bool strict = true> u32 mul(u32 a, u32 b) const {
    return reduce<strict>(u64(a) * b);
  }

  template <bool input_in_space = false, bool output_in_space = false>
  u32 power(u32 b, u32 e) const {
    if (!input_in_space)
      b = mul<false>(b, r2);
    u32 r = output_in_space ? this->r : 1;
    for (; e > 0; e >>= 1) {
      if (e & 1)
        r = mul<false>(r, b);
      b = mul<false>(b, b);
    }
    return shrink(r);
  }

  template <bool input_in_space = false, bool in_space_res = true>
  uint32_t inv(uint32_t a) const {
    return power<input_in_space, in_space_res>(a, mod - 2);
  }
};

using i256 = __m256i;
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u64 __attribute__((vector_size(32)));

u32x8 load_u32x8(const u32 *ptr) {
  return (u32x8)_mm256_load_si256((const i256 *)ptr);
}
void store_u32x8(u32 *ptr, u32x8 vec) {
  _mm256_store_si256((i256 *)ptr, (i256)vec);
}

struct Montgomery_simd {
  u32x8 mod;   // mod
  u32x8 mod2;  // 2 * mod
  u32x8 n_inv; // n_inv * mod == -1 (mod 2^32)
  u32x8 r;     // 2^32 % mod
  u32x8 r2;    // (2^32)^2 % mod

  Montgomery_simd() = default;
  Montgomery_simd(u32 mod) {
    Montgomery mt(mod);
    this->mod = (u32x8)_mm256_set1_epi32(mt.mod);
    this->mod2 = (u32x8)_mm256_set1_epi32(mt.mod2);
    this->n_inv = (u32x8)_mm256_set1_epi32(mt.n_inv);
    this->r = (u32x8)_mm256_set1_epi32(mt.r);
    this->r2 = (u32x8)_mm256_set1_epi32(mt.r2);
  }

  u32x8 shrink(u32x8 vec) const {
    return (u32x8)_mm256_min_epu32((i256)vec,
                                   _mm256_sub_epi32((i256)vec, (i256)mod));
  }
  u32x8 shrink2(u32x8 vec) const {
    return (u32x8)_mm256_min_epu32((i256)vec,
                                   _mm256_sub_epi32((i256)vec, (i256)mod2));
  }
  u32x8 shrink_n(u32x8 vec) const {
    return (u32x8)_mm256_min_epu32((i256)vec,
                                   _mm256_add_epi32((i256)vec, (i256)mod));
  }
  u32x8 shrink2_n(u32x8 vec) const {
    return (u32x8)_mm256_min_epu32((i256)vec,
                                   _mm256_add_epi32((i256)vec, (i256)mod2));
  }

  template <bool strict = true> u32x8 reduce(u64x4 x0246, u64x4 x1357) const {
    u64x4 x0246_ninv = (u64x4)_mm256_mul_epu32((i256)x0246, (i256)n_inv);
    u64x4 x1357_ninv = (u64x4)_mm256_mul_epu32((i256)x1357, (i256)n_inv);
    u64x4 x0246_res = (u64x4)_mm256_add_epi64(
        (i256)x0246, _mm256_mul_epu32((i256)x0246_ninv, (i256)mod));
    u64x4 x1357_res = (u64x4)_mm256_add_epi64(
        (i256)x1357, _mm256_mul_epu32((i256)x1357_ninv, (i256)mod));
    u32x8 res = (u32x8)_mm256_or_si256(_mm256_bsrli_epi128((i256)x0246_res, 4),
                                       (i256)x1357_res);
    if (strict)
      res = shrink(res);
    return res;
  }

  template <bool strict = true, bool b_use_only_even = false>
  u32x8 mul_u32x8(u32x8 a, u32x8 b) const {
    u32x8 a_sh = (u32x8)_mm256_bsrli_epi128((i256)a, 4);
    u32x8 b_sh = b_use_only_even ? b : (u32x8)_mm256_bsrli_epi128((i256)b, 4);
    u64x4 x0246 = (u64x4)_mm256_mul_epu32((i256)a, (i256)b);
    u64x4 x1357 = (u64x4)_mm256_mul_epu32((i256)a_sh, (i256)b_sh);
    return reduce<strict>(x0246, x1357);
  }

  template <bool strict = true> u64x4 mul_u64x4(u64x4 a, u64x4 b) const {
    u64x4 pr = (u64x4)_mm256_mul_epu32((i256)a, (i256)b);
    u64x4 pr2 = (u64x4)_mm256_mul_epu32(_mm256_mul_epu32((i256)pr, (i256)n_inv),
                                        (i256)mod);
    u64x4 res =
        (u64x4)_mm256_bsrli_epi128(_mm256_add_epi64((i256)pr, (i256)pr2), 4);
    if (strict)
      res = (u64x4)shrink((u32x8)res);
    return res;
  }
};

class NTT {
public:
  u32 mod, pr_root;

  Montgomery mt;
  Montgomery_simd mts;

private:
  static constexpr int LG = 32; // more than enough for u32

  u32 w[4], wr[4];
  u32 wd[LG], wrd[LG];

  u64x4 wt_init, wrt_init;
  u64x4 wd_x4[LG], wrd_x4[LG];

  u64x4 wl_init;
  u64x4 wld_x4[LG];

  static u32 find_pr_root(u32 mod, const Montgomery &mt) {
    std::vector<u32> factors;
    u32 n = mod - 1;
    for (u32 i = 2; u64(i) * i <= n; i++) {
      if (n % i == 0) {
        factors.push_back(i);
        do {
          n /= i;
        } while (n % i == 0);
      }
    }
    if (n > 1) {
      factors.push_back(n);
    }
    for (u32 i = 2; i < mod; i++) {
      if (std::all_of(factors.begin(), factors.end(), [&](u32 f) {
            return mt.power<false, false>(i, (mod - 1) / f) != 1;
          })) {
        return i;
      }
    }
    assert(false && "primitive root not found");
  }

public:
  NTT() = default;
  NTT(u32 mod) : mod(mod), mt(mod), mts(mod) {
    const Montgomery mt = this->mt;
    const Montgomery_simd mts = this->mts;

    pr_root = find_pr_root(mod, mt);

    int lg = __builtin_ctz(mod - 1);
    assert(lg <= LG);

    memset(w, 0, sizeof(w));
    memset(wr, 0, sizeof(wr));
    memset(wd_x4, 0, sizeof(wd_x4));
    memset(wrd_x4, 0, sizeof(wrd_x4));
    memset(wld_x4, 0, sizeof(wld_x4));

    std::vector<u32> vec(lg + 1), vecr(lg + 1);
    vec[lg] = mt.power<false, true>(pr_root, mod - 1 >> lg);
    vecr[lg] = mt.power<true, true>(vec[lg], mod - 2);
    for (int i = lg - 1; i >= 0; i--) {
      vec[i] = mt.mul<true>(vec[i + 1], vec[i + 1]);
      vecr[i] = mt.mul<true>(vecr[i + 1], vecr[i + 1]);
    }

    w[0] = wr[0] = mt.r;
    if (lg >= 2) {
      w[1] = vec[2], wr[1] = vecr[2];
      if (lg >= 3) {
        w[2] = vec[3], wr[2] = vecr[3];
        w[3] = mt.mul<true>(w[1], w[2]);
        wr[3] = mt.mul<true>(wr[1], wr[2]);
      }
    }
    wt_init = (u64x4)_mm256_setr_epi64x(w[0], w[0], w[0], w[1]);
    wrt_init = (u64x4)_mm256_setr_epi64x(wr[0], wr[0], wr[0], wr[1]);

    wl_init = (u64x4)_mm256_setr_epi64x(w[0], w[1], w[2], w[3]);

    u32 prf = mt.r, prf_r = mt.r;
    for (int i = 0; i < lg - 2; i++) {
      u32 f = mt.mul<true>(prf, vec[i + 3]),
          fr = mt.mul<true>(prf_r, vecr[i + 3]);
      prf = mt.mul<true>(prf, vecr[i + 3]),
      prf_r = mt.mul<true>(prf_r, vec[i + 3]);
      u32 f2 = mt.mul<true>(f, f), f2r = mt.mul<true>(fr, fr);

      wd_x4[i] = (u64x4)_mm256_setr_epi64x(f2, f, f2, f);
      wrd_x4[i] = (u64x4)_mm256_setr_epi64x(f2r, fr, f2r, fr);
    }

    prf = mt.r;
    for (int i = 0; i < lg - 3; i++) {
      u32 f = mt.mul<true>(prf, vec[i + 4]);
      prf = mt.mul<true>(prf, vecr[i + 4]);
      wld_x4[i] = (u64x4)_mm256_set1_epi64x(f);
    }
  }

private:
  static constexpr int L0 = 3;
  int get_low_lg(int lg) const { return lg % 2 == L0 % 2 ? L0 : L0 + 1; }

  //    public:
  //     bool lg_available(int lg) {
  //         return L0 <= lg && lg <= __builtin_ctz(mod - 1) + get_low_lg(lg);
  //     }

private:
  template <bool transposed, bool trivial = false>
  static void butterfly_x2(u32 *ptr_a, u32 *ptr_b, u32x8 w,
                           const Montgomery_simd &mts) {
    u32x8 a = load_u32x8(ptr_a), b = load_u32x8(ptr_b);
    u32x8 a2, b2;
    if (!transposed) {
      a = mts.shrink2(a),
      b = trivial ? mts.shrink2(b) : mts.mul_u32x8<false, true>(b, w);
      a2 = a + b, b2 = a + mts.mod2 - b;
    } else {
      a2 = mts.shrink2(a + b),
      b2 = trivial ? mts.shrink2_n(a - b)
                   : mts.mul_u32x8<false, true>(a + mts.mod2 - b, w);
    }
    store_u32x8(ptr_a, a2), store_u32x8(ptr_b, b2);
  }

  template <bool transposed, bool trivial = false>
  static void butterfly_x4(u32 *ptr_a, u32 *ptr_b, u32 *ptr_c, u32 *ptr_d,
                           u32x8 w1, u32x8 w2, u32x8 w3,
                           const Montgomery_simd &mts) {
    u32x8 a = load_u32x8(ptr_a), b = load_u32x8(ptr_b), c = load_u32x8(ptr_c),
          d = load_u32x8(ptr_d);
    if (!transposed) {
      butterfly_x2<false, trivial>((u32 *)&a, (u32 *)&c, w1, mts);
      butterfly_x2<false, trivial>((u32 *)&b, (u32 *)&d, w1, mts);
      butterfly_x2<false, trivial>((u32 *)&a, (u32 *)&b, w2, mts);
      butterfly_x2<false, false>((u32 *)&c, (u32 *)&d, w3, mts);
    } else {
      butterfly_x2<true, trivial>((u32 *)&a, (u32 *)&b, w2, mts);
      butterfly_x2<true, false>((u32 *)&c, (u32 *)&d, w3, mts);
      butterfly_x2<true, trivial>((u32 *)&a, (u32 *)&c, w1, mts);
      butterfly_x2<true, trivial>((u32 *)&b, (u32 *)&d, w1, mts);
    }
    store_u32x8(ptr_a, a), store_u32x8(ptr_b, b), store_u32x8(ptr_c, c),
        store_u32x8(ptr_d, d);
  }

  template <bool inverse, bool trivial = false>
  void transform_aux(int k, int i, u32 *data, u64x4 &wi,
                     const Montgomery_simd &mts) const {
    u32x8 w1 = (u32x8)_mm256_shuffle_epi32((i256)wi, 0b00'00'00'00);
    u32x8 w2 = (u32x8)_mm256_permute4x64_epi64(
        (i256)wi, 0b01'01'01'01); // only even indices will be used
    u32x8 w3 = (u32x8)_mm256_permute4x64_epi64(
        (i256)wi, 0b11'11'11'11); // only even indices will be used
    for (int j = 0; j < (1 << k); j += 8) {
      butterfly_x4<inverse, trivial>(
          data + i + (1 << k) * 0 + j, data + i + (1 << k) * 1 + j,
          data + i + (1 << k) * 2 + j, data + i + (1 << k) * 3 + j, w1, w2, w3,
          mts);
    }
    wi = mts.mul_u64x4<true>(
        wi, (inverse ? wrd_x4 : wd_x4)[__builtin_ctz(~i >> k + 2)]);
  }

public:
  // input in [0, 4 * mod)
  // output in [0, 4 * mod)
  // data must be 32-byte aligned
  void transform_forward(int lg, u32 *data) const {
    const Montgomery_simd mts = this->mts;
    const int L = get_low_lg(lg);

    // for (int k = lg - 2; k >= L; k -= 2) {
    //     u64x4 wi = wt_init;
    //     transform_aux<false, true>(k, 0, data, wi, mts);
    //     for (int i = (1 << k + 2); i < (1 << lg); i += (1 << k + 2)) {
    //         transform_aux<false>(k, i, data, wi, mts);
    //     }
    // }

    if (L < lg) {
      const int lc = (lg - L) / 2;
      u64x4 wi_data[LG / 2];
      std::fill(wi_data, wi_data + lc, wt_init);

      for (int k = lg - 2; k >= L; k -= 2) {
        transform_aux<false, true>(k, 0, data, wi_data[k - L >> 1], mts);
      }
      for (int i = 1; i < (1 << lc * 2 - 2); i++) {
        int s = __builtin_ctz(i) >> 1;
        for (int k = s; k >= 0; k--) {
          transform_aux<false>(2 * k + L, i * (1 << L + 2), data, wi_data[k],
                               mts);
        }
      }
    }
  }

  // input in [0, 2 * mod)
  // output in [0, mod)
  // data must be 32-byte aligned
  template <bool mul_by_sc = false>
  void transform_inverse(int lg, u32 *data,
                         /* as normal number */ u32 sc = u32()) const {
    const Montgomery_simd mts = this->mts;
    const int L = get_low_lg(lg);

    // for (int k = L; k + 2 <= lg; k += 2) {
    //     u64x4 wi = wrt_init;
    //     transform_aux<true, true>(k, 0, data, wi, mts);
    //     for (int i = (1 << k + 2); i < (1 << lg); i += (1 << k + 2)) {
    //         transform_aux<true>(k, i, data, wi, mts);
    //     }
    // }

    if (L < lg) {
      const int lc = (lg - L) / 2;
      u64x4 wi_data[LG / 2];
      std::fill(wi_data, wi_data + lc, wrt_init);

      for (int i = 0; i < (1 << lc * 2 - 2); i++) {
        int s = __builtin_ctz(~i) >> 1;
        if (i + 1 == (1 << 2 * s)) {
          s--;
        }
        for (int k = 0; k <= s; k++) {
          transform_aux<true>(2 * k + L, (i + 1 - (1 << 2 * k)) * (1 << L + 2),
                              data, wi_data[k], mts);
        }
        if (i + 1 == (1 << 2 * (s + 1))) {
          s++;
          transform_aux<true, true>(2 * s + L,
                                    (i + 1 - (1 << 2 * s)) * (1 << L + 2), data,
                                    wi_data[s], mts);
        }
      }
    }

    const Montgomery mt = this->mt;
    u32 f = mt.power<false, true>(mod + 1 >> 1, lg - L);
    if constexpr (mul_by_sc)
      f = mt.mul<true>(f, mt.mul<false>(mt.r2, sc));
    u32x8 f_x8 = (u32x8)_mm256_set1_epi32(f);
    for (int i = 0; i < (1 << lg); i += 8) {
      store_u32x8(data + i,
                  mts.mul_u32x8<true, true>(load_u32x8(data + i), f_x8));
    }
  }

private:
  // input in [0, 4 * mod)
  // output in [0, 2 * mod)
  // multiplies mod (x^2^L - w)
  template <int L, int K, bool remove_montgomery_reduction_factor = true>
  /* !!! O3 is crucial here !!! */ __attribute__((optimize("O3"))) static void
  aux_mul_mod_x2L(const u32 *a, const u32 *b, u32 *c,
                  const std::array<u32x8, K> &ar_w,
                  const Montgomery_simd &mts) {
    static_assert(L >= 3);
    // static_assert(L == L0 || L == L0 + 1);

    constexpr int n = 1 << L;
    alignas(64) u32 aux_a[K][n];
    alignas(64) u64 aux_b[K][n * 2];
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < n; i += 8) {
        u32x8 ai = load_u32x8(a + n * k + i);
        if constexpr (remove_montgomery_reduction_factor) {
          ai = mts.mul_u32x8<true, true>(ai, mts.r2);
        } else {
          ai = mts.shrink(mts.shrink2(ai));
        }
        store_u32x8(aux_a[k] + i, ai);

        u32x8 bi = load_u32x8(b + n * k + i);
        u32x8 bi_0 = mts.shrink(mts.shrink2(bi));
        u32x8 bi_w = mts.mul_u32x8<true, true>(bi, ar_w[k]);

        store_u32x8((u32 *)(aux_b[k] + i + 0),
                    (u32x8)_mm256_permutevar8x32_epi32(
                        (i256)bi_w, _mm256_setr_epi64x(0, 1, 2, 3)));
        store_u32x8((u32 *)(aux_b[k] + i + 4),
                    (u32x8)_mm256_permutevar8x32_epi32(
                        (i256)bi_w, _mm256_setr_epi64x(4, 5, 6, 7)));
        store_u32x8((u32 *)(aux_b[k] + n + i + 0),
                    (u32x8)_mm256_permutevar8x32_epi32(
                        (i256)bi_0, _mm256_setr_epi64x(0, 1, 2, 3)));
        store_u32x8((u32 *)(aux_b[k] + n + i + 4),
                    (u32x8)_mm256_permutevar8x32_epi32(
                        (i256)bi_0, _mm256_setr_epi64x(4, 5, 6, 7)));
      }
    }

    u64x4 aux_ans[K][n / 4];
    memset(aux_ans, 0, sizeof(aux_ans));
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        u64x4 ai = (u64x4)_mm256_set1_epi32(aux_a[k][i]);
        for (int j = 0; j < n; j += 4) {
          u64x4 bi = (u64x4)_mm256_loadu_si256((i256 *)(aux_b[k] + n - i + j));
          aux_ans[k][j / 4] +=
              /* 64-bit addition */ (u64x4)_mm256_mul_epu32((i256)ai, (i256)bi);
        }
      }
      if (i >= 8 && (i & 7) == 7) {
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < n; j += 4) {
            aux_ans[k][j / 4] = (u64x4)mts.shrink2((u32x8)aux_ans[k][j / 4]);
          }
        }
      }
    }

    for (int k = 0; k < K; k++) {
      for (int i = 0; i < n; i += 8) {
        u64x4 c0 = aux_ans[k][i / 4], c1 = aux_ans[k][i / 4 + 1];
        u32x8 res = (u32x8)_mm256_permutevar8x32_epi32(
            (i256)mts.reduce<false>(c0, c1),
            _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
        store_u32x8(c + k * n + i, mts.shrink2(res));
      }
    }
  }

  template <int L, bool remove_montgomery_reduction_factor = true>
  void aux_mul_mod_full(int lg, const u32 *a, const u32 *b, u32 *c) const {
    constexpr int sz = 1 << L;
    const Montgomery_simd mts = this->mts;
    int cnt = 1 << lg - L;
    if (cnt == 1) {
      aux_mul_mod_x2L<L, 1, remove_montgomery_reduction_factor>(a, b, c,
                                                                {mts.r}, mts);
      return;
    }
    if (cnt <= 8) {
      for (int i = 0; i < cnt; i += 2) {
        u32x8 wi = (u32x8)_mm256_set1_epi32(w[i / 2]);
        aux_mul_mod_x2L<L, 2, remove_montgomery_reduction_factor>(
            a + i * sz, b + i * sz, c + i * sz, {wi, (mts.mod - wi)}, mts);
      }
      return;
    }
    u64x4 wi = wl_init;
    for (int i = 0; i < cnt; i += 8) {
      u32x8 w_ar[4] = {
          (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b00'00'00'00),
          (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b01'01'01'01),
          (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b10'10'10'10),
          (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b11'11'11'11),
      };
      if constexpr (L == L0) {
        for (int j = 0; j < 8; j += 4) {
          aux_mul_mod_x2L<L, 4, remove_montgomery_reduction_factor>(
              a + (i + j) * sz, b + (i + j) * sz, c + (i + j) * sz,
              {w_ar[j / 2], mts.mod - w_ar[j / 2], w_ar[j / 2 + 1],
               mts.mod - w_ar[j / 2 + 1]},
              mts);
        }
      } else {
        for (int j = 0; j < 8; j += 2) {
          aux_mul_mod_x2L<L, 2, remove_montgomery_reduction_factor>(
              a + (i + j) * sz, b + (i + j) * sz, c + (i + j) * sz,
              {w_ar[j / 2], mts.mod - w_ar[j / 2]}, mts);
        }
      }
      wi = mts.mul_u64x4<true>(wi, wld_x4[__builtin_ctz(~i >> 3)]);
    }
  }

public:
  template <bool remove_montgomery_reduction_factor = true>
  void aux_dot_mod(int lg, const u32 *a, const u32 *b, u32 *c) const {
    int L = get_low_lg(lg);
    if (L == L0) {
      aux_mul_mod_full<L0, remove_montgomery_reduction_factor>(lg, a, b, c);
    } else {
      aux_mul_mod_full<L0 + 1, remove_montgomery_reduction_factor>(lg, a, b, c);
    }
  }

  // lg must be greater than or equal to 3
  // a, b must be 32-byte aligned
  void convolve_cyclic(int lg, u32 *a, u32 *b) const {
    transform_forward(lg, a);
    transform_forward(lg, b);
    aux_dot_mod(lg, a, b, a);
    transform_inverse(lg, a);
  }

  alignas(32) inline static u32 buf1[1 << 20], buf2[1 << 20];
  template <bool are_a_b_extended = false, bool is_square = false>
  void inplace_convolve(std::vector<uint32_t> &lhs_poly,
                        std::vector<uint32_t> &rhs_poly) const {
    int sz = 0;
    for (;
         (1 << sz) < (are_a_b_extended ? lhs_poly.size()
                                       : lhs_poly.size() + rhs_poly.size() - 1);
         sz++)
      ;
    assert(sz >= 3);

    memcpy(buf1, lhs_poly.data(), 4 << sz);
    memcpy(buf2, rhs_poly.data(), 4 << sz);
    convolve_cyclic(sz, buf1, buf2);
    memcpy(lhs_poly.data(), buf1, 4 << sz);
  }
};
} // namespace super_fast_NTT

using super_fast_NTT::NTT;

class BigInteger {
  inline static uint64_t kBase = 1ull << 32;

private:
  inline static const NTT ntt_mod1 = NTT(998244353);
  inline static const NTT ntt_mod2 = NTT(897581057);
  inline static const NTT ntt_mod3 = NTT(880803841);

  void normalize_helper() {
    if (data_.size() == 0) {
      data_ = {0};
      return;
    }

    while (data_.size() >= 2 && data_.back() == 0) {
      data_.pop_back();
    }
  }

  void normalize() { normalize_helper(); }

  void normalize() const {
    return; // nothing
  }

public:
  std::vector<uint32_t> data_;
  BigInteger() : data_({0}) {}
  explicit BigInteger(std::vector<uint32_t> data) : data_(data) {
    if (data.empty()) {
      data = {0};
    }
    normalize_helper();
  }

private:
  void smart_multiplication(const BigInteger &other) {
    uint64_t from_less_digits = 0;
    uint32_t result_size =
        static_cast<uint32_t>(data_.size() + other.data_.size() - 1);
    uint32_t sz = 3;
    for (; (1u << sz) < result_size; sz++) {
    }

    const auto mt1 = ntt_mod1.mt, mt2 = ntt_mod2.mt, mt3 = ntt_mod3.mt;
    std::vector<uint32_t> convolved_mod1 = data_, mod1_temp_data = other.data_;
    convolved_mod1.resize(1 << sz), mod1_temp_data.resize(1 << sz);
    std::vector<uint32_t> convolved_mod2 = convolved_mod1,
                          mod2_temp_data = mod1_temp_data,
                          convolved_mod3 = convolved_mod1,
                          mod3_temp_data = mod1_temp_data;
    ntt_mod1.inplace_convolve<1, /*is_square=*/false>(convolved_mod1,
                                                      mod1_temp_data);
    ntt_mod2.inplace_convolve<1, /*is_square=*/false>(convolved_mod2,
                                                      mod2_temp_data);
    ntt_mod3.inplace_convolve<1, /*is_square=*/false>(convolved_mod3,
                                                      mod3_temp_data);

    static uint32_t garner_magic_const_1 = mt2.inv<false, true>(mt1.mod);
    static uint32_t garner_magic_const_2 =
        mt3.inv<false, true>(static_cast<uint32_t>(
            static_cast<uint64_t>(mt1.mod) * mt2.mod % mt3.mod));
    static uint32_t garner_magic_const_3 = mt3.inv<false, true>(mt2.mod);

    convolved_mod1.resize((1 << sz));
    for (uint32_t i = 0; i < (1u << sz); i++) {
      uint32_t mod1_result_residue = mt1.shrink(convolved_mod1[i]);
      uint32_t mod2_result_residue =
          mt2.mul<true>(2 * mt2.mod + convolved_mod2[i] - mod1_result_residue,
                        garner_magic_const_1);
      uint32_t mod3_result_residue = mt3.shrink(
          mt3.mul<true>(2 * mt3.mod + convolved_mod3[i] - mod1_result_residue,
                        garner_magic_const_2) +
          mt3.mul<true>(2 * mt3.mod - mod2_result_residue,
                        garner_magic_const_3));

      uint64_t lower = static_cast<uint64_t>(mt1.mod) * mt2.mod % kBase,
               upper = static_cast<uint64_t>(mt1.mod) * mt2.mod / kBase;
      upper = upper * mod3_result_residue + lower * mod3_result_residue / kBase;
      lower = lower * mod3_result_residue % kBase;
      lower += static_cast<uint64_t>(mod2_result_residue) * mt1.mod;
      lower += mod1_result_residue + from_less_digits;
      upper += lower / kBase;
      lower %= kBase;

      convolved_mod1[i] = static_cast<uint32_t>(lower);
      from_less_digits = upper;
    }

    while (from_less_digits) {
      convolved_mod1.push_back(static_cast<uint32_t>(from_less_digits % kBase));
      from_less_digits /= kBase;
    }

    data_ = convolved_mod1;
    normalize();
  }

public:
  BigInteger &operator*=(BigInteger other) {
    smart_multiplication(other);
    return *this;
  }
};
