#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>

#define NOINLINE __attribute__((noinline))
// #define NOINLINE

using u32 = uint32_t;
using u64 = uint64_t;

// there is no std::countr_zero on raspberry pi :(
static constexpr inline u32 countr_zero(u32 x) { return __builtin_ctz(x); }

template <u32 mod> struct Montgomery {
  static constexpr int W = 32, L = 5;
  static constexpr u32 xinv = [] {
    u32 q = 1;
    for (int j = 0; j < L; j++)
      q *= 2 - q * mod;
    q = -q;
    return q;
  }();
  static_assert(u32(-xinv * mod) == 1);
  static constexpr u32 R1 = static_cast<u32>((1ULL << W) % mod);
  static constexpr u32 R2 = static_cast<u32>(1ULL * R1 * R1 % mod);
  static_assert(mod < std::numeric_limits<u32>::max() / 4);

  static constexpr u32 redc(u64 T) {
    u64 m = u32(T) * xinv;
    T += m * mod;
    T >>= W;
    return u32(T >= mod ? T - mod : T);
  }
  static constexpr u32 redc(u32 a, u32 b) {
    assert(a < mod && b < mod);
    return redc(u64(a) * b);
  }
  static constexpr u32 from(u32 x) {
    assert(x < mod);
    return redc(x, R2);
  }
  static constexpr u32 get(u32 a) { return redc(a); }
  static constexpr u32 one() { return R1; }

  static constexpr u32 add(u32 a, u32 b) {
    return a + b >= mod ? a + b - mod : a + b;
  }
  static constexpr u32 sub(u32 a, u32 b) { return a < b ? a - b + mod : a - b; }
  static constexpr u32 mul(u32 a, u32 b) {
    return static_cast<u32>(u64(a) * b % mod);
    /* return get(redc(from(a), from(b))); */
  }
  static constexpr u32 pow(u32 e, u32 p) {
    u32 r = 1;
    while (p) {
      if (p & 1)
        r = mul(r, e);
      e = mul(e, e);
      p >>= 1;
    }
    return r;
  }
  static constexpr u32 inv(u32 x) { return pow(x, mod - 2); }
  // a * b % mod == get(redc(from(a), from(b)))

  // +---------------------------------------------------------------+
  // |                           SIMD                                |
  // +---------------------------------------------------------------+

  static uint32x4_t redc_64x4(uint64x2_t T_lo, uint64x2_t T_hi) {
    // Extract low 32 bits from T for multiplication with xinv
    uint32x4_t T_low = vcombine_u32(vmovn_u64(T_lo), vmovn_u64(T_hi));

    // Compute m = T_low * xinv (mod 2^32)
    uint32x4_t m = vmulq_u32(T_low, vdupq_n_u32(xinv));

    // Multiply m with mod
    // Add m * mod to T
    T_lo = vmlal_u32(T_lo, vdup_n_u32(mod), vget_low_u32(m));
    T_hi = vmlal_u32(T_hi, vdup_n_u32(mod), vget_high_u32(m));

    // Shift right by W (32 bits) - take high 32 bits
    uint32x4_t result =
        vcombine_u32(vshrn_n_u64(T_lo, 32), vshrn_n_u64(T_hi, 32));

    // Conditional subtraction: if result >= mod then result -= mod
    result = vminq_u32(result, vsubq_u32(result, vdupq_n_u32(mod)));

    return result;
  }

  static uint32x4_t redc_32x4(uint32x4_t a, uint32x4_t b) {
    // Multiply a * b using widening multiplication
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint32x2_t b_hi = vget_high_u32(b);

    uint64x2_t ab_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t ab_hi = vmull_u32(a_hi, b_hi);

    return redc_64x4(ab_lo, ab_hi);
  }

  static uint32x4_t from_32x4(uint32x4_t T) {
    return redc_32x4(T, vdupq_n_u32(R2));
  }

  static uint32x4_t get_32x4(uint32x4_t T) {
    return redc_32x4(T, vdupq_n_u32(1));
  }

  template <bool strict = true>
  static uint32x4_t add_32x4(uint32x4_t a, uint32x4_t b) {
    uint32x4_t sum = vaddq_u32(a, b);
    if constexpr (strict) {
      return vminq_u32(sum, vsubq_u32(sum, vdupq_n_u32(mod)));
    } else {
      return sum;
    }
  }

  static uint32x4_t shrink(uint32x4_t x) {
    return vminq_u32(x, vsubq_u32(x, vdupq_n_u32(mod)));
  }

  static uint32x4_t shrink2(uint32x4_t x) {
    return vminq_u32(x, vsubq_u32(x, vdupq_n_u32(mod * 2)));
  }

  template <bool strict = true>
  static uint32x4_t sub_32x4(uint32x4_t a, uint32x4_t b) {
    uint32x4_t diff = vsubq_u32(a, b);
    if constexpr (strict) {
      return vminq_u32(diff, vaddq_u32(diff, vdupq_n_u32(mod)));
    } else {
      return vaddq_u32(diff, vdupq_n_u32(mod));
    }
  }

  struct MulConstContext {
    uint32_t b, b_prime;
    MulConstContext(u32 t_b) : b(t_b), b_prime(-t_b * xinv) {}
  };

  template <bool strict = true>
  static uint32x4_t redc_32x4_by_context(uint32x4_t a,
                                         const MulConstContext &ctx) {
    // use signed reduction
    int32x4_t a_signed = vreinterpretq_s32_u32(a);
    uint32x4_t low = vmulq_u32(a, vdupq_n_u32(ctx.b_prime));
    int32x4_t low_signed = vreinterpretq_s32_u32(low);
    int32x4_t diff = vsubq_s32(vqrdmulhq_s32(a_signed, vdupq_n_s32(ctx.b)),
                               vqrdmulhq_s32(low_signed, vdupq_n_s32(mod)));
    uint32x4_t result = (uint32x4_t)vshrq_n_s32(diff, 1);
    if constexpr (strict) {
      return vminq_u32(result, vaddq_u32(result, vdupq_n_u32(mod)));
    } else {
      return vaddq_u32(result, vdupq_n_u32(mod));
    }
  }
};

template <u32 mod> static constexpr u32 find_primitive_root() {
  using Mont = Montgomery<mod>;
  u32 factors[30] = {}, k = 0;
  u32 n = mod - 1;
  for (u32 i = 2; u64(i) * i <= n; i++)
    if (n % i == 0) {
      factors[k++] = i;
      while (n % i == 0)
        n /= i;
    }
  if (n > 1) {
    factors[k++] = n;
  }
  for (u32 i = 2; i < mod; i++) {
    bool ok = true;
    for (u32 j = 0; j < k; j++)
      if (Mont::pow(i, (mod - 1) / factors[j]) == 1) {
        ok = false;
      }
    if (ok) {
      return i;
    }
  }
  assert(false && "primitive root not found");
}

template <u32 mod, u32 G = find_primitive_root<mod>()> struct NTT {
  static constexpr int simd_size = 4;

  using Mont = Montgomery<mod>;
  using MulConstContext = Mont::MulConstContext;
  static constexpr int rank2 = countr_zero(mod - 1);
  std::array<u32, rank2 + 1> root;  // root[i]^(2^i) == 1
  std::array<u32, rank2 + 1> iroot; // root[i] * iroot[i] == 1

  // root, iroot 陣列不在 montgomery domain
  // 但以下的陣列都在 montgomery domain 裡面
  std::array<u32, std::max(0, rank2 - 2 + 1)> rate2;
  std::array<u32, std::max(0, rank2 - 2 + 1)> irate2;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> rate3_simd;
  std::array<u32, std::max(0, rank2 - 3 + 1)> irate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> irate3_simd;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3_4; // rate3[i]^4

  std::array<u32, std::max(0, rank2 - 4 + 1)> rate4;
  std::array<u32, std::max(0, rank2 - 4 + 1)> irate4;
  std::array<u32, std::max(0, rank2 - 4 + 1)> rate4_8; // rate4[i]^8

  constexpr NTT() {
    root[rank2] = Mont::pow(G, (mod - 1) >> rank2);
    iroot[rank2] = Mont::inv(root[rank2]);
    for (int i = rank2 - 1; i >= 0; i--) {
      root[i] = Mont::mul(root[i + 1], root[i + 1]);
      iroot[i] = Mont::mul(iroot[i + 1], iroot[i + 1]);
      assert(Mont::mul(root[i], iroot[i]) == 1);
    }

    // prepare rate2 for radix2-fft
    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = Mont::mul(root[i + 2], prod);
        irate2[i] = Mont::mul(iroot[i + 2], iprod);
        prod = Mont::mul(prod, iroot[i + 2]);
        iprod = Mont::mul(iprod, root[i + 2]);
        assert(Mont::mul(rate2[i], irate2[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = Mont::from(rate2[i]);
        irate2[i] = Mont::from(irate2[i]);
      }
    }

    // prepare rate3 for radix4-fft
    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = Mont::mul(root[i + 3], prod);
        irate3[i] = Mont::mul(iroot[i + 3], iprod);
        prod = Mont::mul(prod, iroot[i + 3]);
        iprod = Mont::mul(iprod, root[i + 3]);
        assert(Mont::mul(rate3[i], irate3[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = Mont::from(rate3[i]);
        irate3[i] = Mont::from(irate3[i]);
      }
    }

    // prepare rate3's power for radix4-fft and pointwise mul
    {
      for (int i = 0; i <= rank2 - 3; i++) {
        u32 prod = Mont::one(), power[4];
        for (int j = 0; j < 4; j++) {
          power[j] = prod;
          prod = Mont::redc(prod, rate3[i]);
        }
        rate3_4[i] = prod;
        rate3_simd[i] = vld1q_u32(power);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        u32 prod = Mont::one(), power[4];
        for (int j = 0; j < 4; j++) {
          power[j] = prod;
          prod = Mont::redc(prod, irate3[i]);
        }
        irate3_simd[i] = vld1q_u32(power);
      }
    }

    // prepare rate4
    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 4; i++) {
        rate4[i] = Mont::mul(root[i + 4], prod);
        irate4[i] = Mont::mul(iroot[i + 4], iprod);
        prod = Mont::mul(prod, iroot[i + 4]);
        iprod = Mont::mul(iprod, root[i + 4]);
        assert(Mont::mul(rate4[i], irate4[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 4; i++) {
        rate4[i] = Mont::from(rate4[i]);
        irate4[i] = Mont::from(irate4[i]);
      }
    }
    {
      for (int i = 0; i <= rank2 - 4; i++) {
        u32 prod = Mont::one();
        for (int j = 0; j < 8; j++) {
          prod = Mont::redc(prod, rate4[i]);
        }
        rate4_8[i] = prod;
      }
    }
  }

  void radix4_forward(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    u32 rot = Mont::one(), imag = Mont::from(root[2]);
    for (int s = 0; s < (1 << len); s++) {
      u32 rot2 = Mont::redc(rot, rot);
      u32 rot3 = Mont::redc(rot2, rot);

      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 a0 = F[i + offset];
        u32 a1 = Mont::redc(F[i + offset + p], rot);
        u32 a2 = Mont::redc(F[i + offset + 2 * p], rot2);
        u32 a3 = Mont::redc(F[i + offset + 3 * p], rot3);
        u32 a1na3imag = Mont::redc(Mont::sub(a1, a3), imag);

        F[i + offset] = Mont::add(Mont::add(a0, a2), Mont::add(a1, a3));
        F[i + offset + p] = Mont::sub(Mont::add(a0, a2), Mont::add(a1, a3));
        F[i + offset + 2 * p] = Mont::add(Mont::sub(a0, a2), a1na3imag);
        F[i + offset + 3 * p] = Mont::sub(Mont::sub(a0, a2), a1na3imag);
      }
      if (s + 1 != (1 << len))
        rot = Mont::redc(rot, rate3[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_forward_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t rot_simd = vdupq_n_u32(Mont::one());
    MulConstContext imag{Mont::from(root[2])};
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      MulConstContext rot{vgetq_lane_u32(rot_simd, 1)};
      MulConstContext rot2{vgetq_lane_u32(rot_simd, 2)};
      MulConstContext rot3{vgetq_lane_u32(rot_simd, 3)};

      for (int i = 0; i < p; i += simd_size) {
        uint32x4_t a0 = vld1q_u32(&F[i + offset]);
        uint32x4_t a1 = vld1q_u32(&F[i + offset + p]);
        uint32x4_t a2 = vld1q_u32(&F[i + offset + 2 * p]);
        uint32x4_t a3 = vld1q_u32(&F[i + offset + 3 * p]);

        a1 = Mont::redc_32x4_by_context(a1, rot);
        a2 = Mont::redc_32x4_by_context(a2, rot2);
        a3 = Mont::redc_32x4_by_context(a3, rot3);
        uint32x4_t a1na3imag =
            Mont::redc_32x4_by_context(Mont::sub_32x4(a1, a3), imag);

        auto a0pa2 = Mont::add_32x4(a0, a2);
        auto a1pa3 = Mont::add_32x4(a1, a3);
        auto a0na2 = Mont::sub_32x4(a0, a2);
        vst1q_u32(&F[i + offset], Mont::add_32x4(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + p], Mont::sub_32x4(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + 2 * p], Mont::add_32x4(a0na2, a1na3imag));
        vst1q_u32(&F[i + offset + 3 * p], Mont::sub_32x4(a0na2, a1na3imag));
      }
      if (s + 1 != (1 << len))
        rot_simd =
            Mont::redc_32x4(rot_simd, rate3_simd[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_inverse(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    u32 irot = Mont::one(), iimag = Mont::from(iroot[2]);
    for (int s = 0; s < (1 << len); s++) {
      u32 irot2 = Mont::redc(irot, irot);
      u32 irot3 = Mont::redc(irot2, irot);
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 a0 = F[i + offset];
        u32 a1 = F[i + offset + p];
        u32 a2 = F[i + offset + 2 * p];
        u32 a3 = F[i + offset + 3 * p];

        u32 a2na3iimag = Mont::redc(Mont::sub(a2, a3), iimag);

        F[i + offset] = Mont::add(Mont::add(a0, a1), Mont::add(a2, a3));
        F[i + offset + p] =
            Mont::redc(Mont::add(Mont::sub(a0, a1), a2na3iimag), irot);
        F[i + offset + 2 * p] =
            Mont::redc(Mont::sub(Mont::add(a0, a1), Mont::add(a2, a3)), irot2);
        F[i + offset + 3 * p] =
            Mont::redc(Mont::sub(Mont::sub(a0, a1), a2na3iimag), irot3);
      }
      if (s + 1 != (1 << len))
        irot = Mont::redc(irot, irate3[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_inverse_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t irot_simd = vdupq_n_u32(Mont::one());
    MulConstContext iimag{Mont::from(iroot[2])};
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      MulConstContext irot{vgetq_lane_u32(irot_simd, 1)};
      MulConstContext irot2{vgetq_lane_u32(irot_simd, 2)};
      MulConstContext irot3{vgetq_lane_u32(irot_simd, 3)};

      for (int i = 0; i < p; i += simd_size) {
        uint32x4_t a0 = vld1q_u32(&F[i + offset]);
        uint32x4_t a1 = vld1q_u32(&F[i + offset + p]);
        uint32x4_t a2 = vld1q_u32(&F[i + offset + 2 * p]);
        uint32x4_t a3 = vld1q_u32(&F[i + offset + 3 * p]);
        uint32x4_t a2na3iimag =
            Mont::redc_32x4_by_context(Mont::sub_32x4(a2, a3), iimag);

        auto a0pa1 = Mont::add_32x4(a0, a1);
        auto a2pa3 = Mont::add_32x4(a2, a3);
        auto a0na1 = Mont::sub_32x4(a0, a1);
        vst1q_u32(&F[i + offset], Mont::add_32x4(a0pa1, a2pa3));
        vst1q_u32(&F[i + offset + p],
                  Mont::redc_32x4_by_context(Mont::add_32x4(a0na1, a2na3iimag),
                                             irot));
        vst1q_u32(
            &F[i + offset + 2 * p],
            Mont::redc_32x4_by_context(Mont::sub_32x4(a0pa1, a2pa3), irot2));
        vst1q_u32(&F[i + offset + 3 * p],
                  Mont::redc_32x4_by_context(Mont::sub_32x4(a0na1, a2na3iimag),
                                             irot3));
      }
      if (s + 1 != (1 << len))
        irot_simd =
            Mont::redc_32x4(irot_simd, irate3_simd[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_forward(int len, int h, u32 F[]) {
    /* assert(false && "radix2 transform is disabled"); */
    const int p = 1 << (h - len - 1);
    u32 rot = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 l = F[i + offset];
        u32 r = Mont::redc(F[i + offset + p], rot);
        F[i + offset] = Mont::add(l, r);
        F[i + offset + p] = Mont::sub(l, r);
      }
      if (s + 1 != (1 << len))
        rot = Mont::redc(rot, rate2[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_forward_simd(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 1);
    assert(p >= 4 && p % 4 == 0);
    u32 rot = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      MulConstContext rot_ctx{rot};
      for (int i = 0; i < p; i += 4) {
        uint32x4_t l = vld1q_u32(&F[i + offset]);
        uint32x4_t r = vld1q_u32(&F[i + offset + p]);
        r = Mont::redc_32x4_by_context(r, rot);
        vst1q_u32(&F[i + offset], Mont::add_32x4(l, r));
        vst1q_u32(&F[i + offset + p], Mont::sub_32x4(l, r));
      }
      if (s + 1 != (1 << len))
        rot = Mont::redc(rot, rate2[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_inverse(int len, int h, u32 F[]) {
    /* assert(false && "radix2 transform is disabled"); */
    const int p = 1 << (h - len - 1);
    u32 irot = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 l = F[i + offset];
        u32 r = F[i + offset + p];
        F[i + offset] = Mont::add(l, r);
        F[i + offset + p] = Mont::redc(Mont::sub(l, r), irot);
      }
      if (s + 1 != (1 << len))
        irot = Mont::redc(irot, irate2[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_inverse_simd(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 1);
    assert(p >= 4 && p % 4 == 0);
    u32 irot = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      MulConstContext irot_ctx{irot};
      for (int i = 0; i < p; i += 4) {
        uint32x4_t l = vld1q_u32(&F[i + offset]);
        uint32x4_t r = vld1q_u32(&F[i + offset + p]);
        vst1q_u32(&F[i + offset], Mont::add_32x4(l, r));
        vst1q_u32(&F[i + offset + p], Mont::redc_32x4_by_context(Mont::sub_32x4(l, r), irot));
      }
      if (s + 1 != (1 << len))
        irot = Mont::redc(irot, irate2[countr_zero(~(u32)(s))]);
    }
  }

  // assume F is NOT in montgomery domain
  template <int num_skip_round>
  NOINLINE void transform_forward(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = countr_zero((u32)n);

    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], Mont::from_32x4(a0));
    }

    int len = 0; // a[i, i+(n>>len), i+2*(n>>len), ..] is transformed
    while (len < h - num_skip_round) {
      if (len + 2 <= h - num_skip_round) {
        radix4_forward_simd(len, h, F);
        len += 2;
      } else {
        radix2_forward_simd(len, h, F);
        len += 1;
      }
    }
  }

  // assume F is in montgomery domain
  template <int num_skip_round>
  NOINLINE void transform_inverse(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = countr_zero((u32)n);

    int len = h - num_skip_round;
    while (len > 0) {
      if (len >= 2) {
        len -= 2;
        radix4_inverse_simd(len, h, F);
      } else {
        len -= 1;
        radix2_inverse_simd(len, h, F);
      }
    }

    // const u32 invn = modinv(n);
    // for (int i = 0; i < n; i++) {
    //   F[i] = Mont::mul(Mont::get(F[i]), invn);
    // }
    const u32 multiplier = (1u << (32 - h + num_skip_round)) % mod;
    MulConstContext multiplier_simd{multiplier};
    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], Mont::get_32x4(
                           Mont::redc_32x4_by_context(a0, multiplier_simd)));
    }
  }

  NOINLINE void pointwise_product_modx8nw(u32 a[], u32 b[], int n) {
    const int h = countr_zero((u32)n);
    const int len = h - 3;

    u32 rot8 = Mont::one();
    for (int s = 0; s < (1 << len); s++) {

      int offset = s << (h - len);

      u32 aux_b[16] = {};
      for (int y = 0; y < 8; y++)
        aux_b[y + 8] = b[offset + y];
      MulConstContext rot8_ctx{rot8};
      for (int y = 0; y < 8; y += 4) {
        auto wb = Mont::redc_32x4_by_context(vld1q_u32(&aux_b[y + 8]), rot8_ctx);
        vst1q_u32(&aux_b[y], wb);
      }


      uint64x2_t res[4] = {};
#pragma unroll
      for (int x = 0; x < 8; x++) {
        auto ax = vdup_n_u32(a[offset + x]);
#pragma unroll
        for (int z = 0; z < 8; z += 2) {
          auto by = vld1_u32(&aux_b[z + 8 - x]);
          res[z / 2] = vaddq_u64(res[z / 2], vmull_u32(ax, by));
        }
      }
      for (int z = 0; z < 8; z += 2) {
        res[z / 2] = (uint64x2_t)Mont::shrink2((uint32x4_t)res[z / 2]);
      }
      for (int z = 0; z < 8; z += 4) {
        auto az = Mont::redc_64x4(res[z / 2], res[z / 2 + 1]);
        vst1q_u32(&a[offset + z], az);
      }
      // 此時 res 的值域大小是 8 * mod^2，遠小於 2^{64}
      // 所以上面可以在 u64 裡面直接加
      // Montgomery 需要 R * mod 以下，所以需要 shrink 一次

      if (s + 1 != (1 << len))
        rot8 = Mont::redc(rot8, rate4_8[std::countr_zero(~(u32)(s))]);
    }
  }

  // calculate pointwise product mod x^4 - w^4
  // assume a, b are in montgomery domain
  NOINLINE void pointwise_product_modx4nw(u32 a[], u32 b[], int n) {
    const int h = countr_zero((u32)n);
    const int len = h - 2;

    u32 rot4 = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);

      uint32x4_t vb = vld1q_u32(&b[offset]);
      uint32x4_t w_vb = Mont::redc_32x4(vb, vdupq_n_u32(rot4));

      uint64x2_t vres_lo = vdupq_n_u64(0);
      uint64x2_t vres_hi = vdupq_n_u64(0);

      uint32x4_t vb_rot;
      uint32x2_t ax;
#define MUL_HELPER(x)                                                          \
  vb_rot = (x == 0 ? vb : vextq_u32(w_vb, vb, (4 - x)));                       \
  ax = vdup_n_u32(a[offset + x]);                                              \
  vres_lo = vaddq_u64(vres_lo, vmull_u32(ax, vget_low_u32(vb_rot)));           \
  vres_hi = vaddq_u64(vres_hi, vmull_u32(ax, vget_high_u32(vb_rot)));

      MUL_HELPER(0)
      MUL_HELPER(1)
      MUL_HELPER(2)
      MUL_HELPER(3)
#undef MUL_HELPER

      // 此時 vres 的值域大小是 4 * mod^2，遠小於 2^{64}
      // 所以上面可以在 u64 裡面直接加
      // 同時 Montgomery 只需要 R * mod 以下，所以是安全的
      uint32x4_t vres = Mont::redc_64x4(vres_lo, vres_hi);

      vst1q_u32(&a[offset], vres);

      if (s + 1 != (1 << len))
        rot4 = Mont::redc(rot4, rate3_4[countr_zero(~(u32)(s))]);
    }
  }

  // calculate pointwise product mod x^2 - w^2
  // assume a, b are in montgomery domain
  void pointwise_product_modx2nw(u32 a[], u32 b[], int n) {
    const int h = countr_zero((u32)n);
    const int len = h - 1;

    u32 rot = Mont::one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);

      u32 a0 = a[offset], a1 = a[offset + 1];
      u32 b0 = b[offset], b1 = b[offset + 1];
      u32 rot2 = Mont::redc(rot, rot);
      a[offset] =
          Mont::add(Mont::redc(a0, b0), Mont::redc(Mont::redc(a1, b1), rot2));
      a[offset + 1] = Mont::add(Mont::redc(a0, b1), Mont::redc(a1, b0));

      if (s + 1 != (1 << len))
        rot = Mont::redc(rot, rate2[countr_zero(~(u32)(s))]);
    }
  }

  static constexpr int maxn = 1 << 20;
  u32 buf1[maxn], buf2[maxn];
  void convolve(const u32 *a, const u32 *b, int n) {
    assert(n <= maxn);
    memcpy(buf1, a, n * sizeof(u32));
    memcpy(buf2, b, n * sizeof(u32));
    transform_forward<3>(buf1, n);
    transform_forward<3>(buf2, n);
    pointwise_product_modx8nw(buf1, buf2, n);
    transform_inverse<3>(buf1, n);
  }
};
