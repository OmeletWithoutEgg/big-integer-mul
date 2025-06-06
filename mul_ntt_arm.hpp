#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

// there is no std::countr_zero on raspberry pi :(
static constexpr inline u32 countr_zero(u32 x) { return __builtin_ctz(x); }

struct Montgomery {
  constexpr static int W = 32, L = 5;
  u32 mod, R1, R2, xinv;
  constexpr Montgomery(u32 t_mod) : mod(t_mod) {
    mod = t_mod;
    assert(mod & 1);
    xinv = 1;
    for (int j = 0; j < L; j++)
      xinv *= 2 - xinv * mod;
    assert(xinv * mod == 1);
    xinv = -xinv;
    const u64 R = (u64(1) << W) % mod;
    R1 = u32(R);
    R2 = u32(R * R % mod);
  }
  u32 redc(u64 T) const {
    u64 m = u32(T) * xinv;
    T += m * mod;
    T >>= W;
    return u32(T >= mod ? T - mod : T);
  }
  u32 redc(u32 a, u32 b) const {
    assert(a < mod && b < mod);
    return redc(u64(a) * b);
  }
  u32 from(u32 x) const {
    assert(x < mod);
    return redc(x, R2);
  }
  u32 get(u32 a) const { return redc(a); }
  u32 one() const { return R1; }

  u32 add(u32 a, u32 b) const { return a + b >= mod ? a + b - mod : a + b; }
  u32 sub(u32 a, u32 b) const { return a < b ? a - b + mod : a - b; }
  u32 mul(u32 a, u32 b) const {
    // return static_cast<u32>(u64(a) * b % mod);
    return get(redc(from(a), from(b)));
  }
  u32 pow(u32 e, u32 p) const {
    u32 r = 1;
    while (p) {
      if (p & 1)
        r = mul(r, e);
      e = mul(e, e);
      p >>= 1;
    }
    return r;
  }
  u32 inv(u32 x) const { return pow(x, mod - 2); }
  // a * b % mod == get(redc(from(a), from(b)))

  // +---------------------------------------------------------------+
  // |                           SIMD                                |
  // +---------------------------------------------------------------+

  uint32x4_t redc_64x4(uint64x2_t T_lo, uint64x2_t T_hi) const {
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

  uint32x4_t redc_32x4(uint32x4_t a, uint32x4_t b) const {
    // Multiply a * b using widening multiplication
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint32x2_t b_hi = vget_high_u32(b);

    uint64x2_t ab_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t ab_hi = vmull_u32(a_hi, b_hi);

    return redc_64x4(ab_lo, ab_hi);
  }

  uint32x4_t from_32x4(uint32x4_t T) const {
    return redc_32x4(T, vdupq_n_u32(R2));
  }

  uint32x4_t get_32x4(uint32x4_t T) const {
    return redc_32x4(T, vdupq_n_u32(1));
  }

  uint32x4_t add_32x4(uint32x4_t a, uint32x4_t b) const {
    uint32x4_t sum = vaddq_u32(a, b);
    return vminq_u32(sum, vsubq_u32(sum, vdupq_n_u32(mod)));
  }
  uint32x4_t sub_32x4(uint32x4_t a, uint32x4_t b) const {
    uint32x4_t diff = vsubq_u32(a, b);
    return vminq_u32(diff, vaddq_u32(diff, vdupq_n_u32(mod)));
  }
};

template <u32 mod, u32 G> struct NTT {

  static constexpr int simd_size = 4;

  Montgomery mont;
  static constexpr int rank2 = countr_zero(mod - 1);
  static_assert((1 << rank2) >= BIGINT_LIMBS);
  std::array<u32, rank2 + 1> root;  // root[i]^(2^i) == 1
  std::array<u32, rank2 + 1> iroot; // root[i] * iroot[i] == 1

  // root, iroot 陣列不在 montgomery domain，但以下的陣列都在 montgomery domain
  // 裡面
  std::array<u32, std::max(0, rank2 - 2 + 1)> rate2;
  std::array<u32, std::max(0, rank2 - 2 + 1)> irate2;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> rate3_simd;
  std::array<u32, std::max(0, rank2 - 3 + 1)> irate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> irate3_simd;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3_4; // rate3[i]^4

  NTT() : mont(mod) {
    root[rank2] = mont.pow(G, (mod - 1) >> rank2);
    iroot[rank2] = mont.inv(root[rank2]);
    for (int i = rank2 - 1; i >= 0; i--) {
      root[i] = mont.mul(root[i + 1], root[i + 1]);
      iroot[i] = mont.mul(iroot[i + 1], iroot[i + 1]);
      assert(mont.mul(root[i], iroot[i]) == 1);
    }

    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = mont.mul(root[i + 2], prod);
        irate2[i] = mont.mul(iroot[i + 2], iprod);
        prod = mont.mul(prod, iroot[i + 2]);
        iprod = mont.mul(iprod, root[i + 2]);
        assert(mont.mul(rate2[i], irate2[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = mont.from(rate2[i]);
        irate2[i] = mont.from(irate2[i]);
      }
    }
    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = mont.mul(root[i + 3], prod);
        irate3[i] = mont.mul(iroot[i + 3], iprod);
        prod = mont.mul(prod, iroot[i + 3]);
        iprod = mont.mul(iprod, root[i + 3]);
        assert(mont.mul(rate3[i], irate3[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = mont.from(rate3[i]);
        irate3[i] = mont.from(irate3[i]);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        u32 prod = mont.one(), power[4];
        for (int j = 0; j < simd_size; j++) {
          power[j] = prod;
          prod = mont.redc(prod, rate3[i]);
        }
        rate3_4[i] = prod;
        rate3_simd[i] = vld1q_u32(power);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        u32 prod = mont.one(), power[4];
        for (int j = 0; j < simd_size; j++) {
          power[j] = prod;
          prod = mont.redc(prod, irate3[i]);
        }
        irate3_simd[i] = vld1q_u32(power);
      }
    }
  }

  void radix4_forward(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    u32 rot = mont.one(), imag = mont.from(root[2]);
    for (int s = 0; s < (1 << len); s++) {
      u32 rot2 = mont.redc(rot, rot);
      u32 rot3 = mont.redc(rot2, rot);

      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 a0 = F[i + offset];
        u32 a1 = mont.redc(F[i + offset + p], rot);
        u32 a2 = mont.redc(F[i + offset + 2 * p], rot2);
        u32 a3 = mont.redc(F[i + offset + 3 * p], rot3);
        u32 a1na3imag = mont.redc(mont.sub(a1, a3), imag);

        F[i + offset] = mont.add(mont.add(a0, a2), mont.add(a1, a3));
        F[i + offset + p] = mont.sub(mont.add(a0, a2), mont.add(a1, a3));
        F[i + offset + 2 * p] = mont.add(mont.sub(a0, a2), a1na3imag);
        F[i + offset + 3 * p] = mont.sub(mont.sub(a0, a2), a1na3imag);
      }
      if (s + 1 != (1 << len))
        rot = mont.redc(rot, rate3[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_forward_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t rot_simd = vdupq_n_u32(mont.one()),
               imag = vdupq_n_u32(mont.from(root[2]));
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i += simd_size) {
        uint32x4_t rot = vdupq_laneq_u32(rot_simd, 1);
        uint32x4_t rot2 = vdupq_laneq_u32(rot_simd, 2);
        uint32x4_t rot3 = vdupq_laneq_u32(rot_simd, 3);

        uint32x4_t a0 = vld1q_u32(&F[i + offset]);
        uint32x4_t a1 = mont.redc_32x4(vld1q_u32(&F[i + offset + p]), rot);
        uint32x4_t a2 = mont.redc_32x4(vld1q_u32(&F[i + offset + 2 * p]), rot2);
        uint32x4_t a3 = mont.redc_32x4(vld1q_u32(&F[i + offset + 3 * p]), rot3);
        uint32x4_t a1na3imag = mont.redc_32x4(mont.sub_32x4(a1, a3), imag);

        auto a0pa2 = mont.add_32x4(a0, a2);
        auto a1pa3 = mont.add_32x4(a1, a3);
        auto a0na2 = mont.sub_32x4(a0, a2);
        vst1q_u32(&F[i + offset], mont.add_32x4(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + p], mont.sub_32x4(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + 2 * p], mont.add_32x4(a0na2, a1na3imag));
        vst1q_u32(&F[i + offset + 3 * p], mont.sub_32x4(a0na2, a1na3imag));
      }
      if (s + 1 != (1 << len))
        rot_simd = mont.redc_32x4(rot_simd, rate3_simd[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_inverse(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    u32 irot = mont.one(), iimag = mont.from(iroot[2]);
    for (int s = 0; s < (1 << len); s++) {
      u32 irot2 = mont.redc(irot, irot);
      u32 irot3 = mont.redc(irot2, irot);
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 a0 = F[i + offset];
        u32 a1 = F[i + offset + p];
        u32 a2 = F[i + offset + 2 * p];
        u32 a3 = F[i + offset + 3 * p];

        u32 a2na3iimag = mont.redc(mont.sub(a2, a3), iimag);

        F[i + offset] = mont.add(mont.add(a0, a1), mont.add(a2, a3));
        F[i + offset + p] =
            mont.redc(mont.add(mont.sub(a0, a1), a2na3iimag), irot);
        F[i + offset + 2 * p] =
            mont.redc(mont.sub(mont.add(a0, a1), mont.add(a2, a3)), irot2);
        F[i + offset + 3 * p] =
            mont.redc(mont.sub(mont.sub(a0, a1), a2na3iimag), irot3);
      }
      if (s + 1 != (1 << len))
        irot = mont.redc(irot, irate3[countr_zero(~(u32)(s))]);
    }
  }

  void radix4_inverse_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t irot_simd = vdupq_n_u32(mont.one()),
               iimag = vdupq_n_u32(mont.from(iroot[2]));
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i += simd_size) {
        uint32x4_t irot = vdupq_laneq_u32(irot_simd, 1);
        uint32x4_t irot2 = vdupq_laneq_u32(irot_simd, 2);
        uint32x4_t irot3 = vdupq_laneq_u32(irot_simd, 3);

        uint32x4_t a0 = vld1q_u32(&F[i + offset]);
        uint32x4_t a1 = vld1q_u32(&F[i + offset + p]);
        uint32x4_t a2 = vld1q_u32(&F[i + offset + 2 * p]);
        uint32x4_t a3 = vld1q_u32(&F[i + offset + 3 * p]);
        uint32x4_t a2na3iimag = mont.redc_32x4(mont.sub_32x4(a2, a3), iimag);

        auto a0pa1 = mont.add_32x4(a0, a1);
        auto a2pa3 = mont.add_32x4(a2, a3);
        auto a0na1 = mont.sub_32x4(a0, a1);
        vst1q_u32(&F[i + offset], mont.add_32x4(a0pa1, a2pa3));
        vst1q_u32(&F[i + offset + p],
                  mont.redc_32x4(mont.add_32x4(a0na1, a2na3iimag), irot));
        vst1q_u32(&F[i + offset + 2 * p],
                  mont.redc_32x4(mont.sub_32x4(a0pa1, a2pa3), irot2));
        vst1q_u32(&F[i + offset + 3 * p],
                  mont.redc_32x4(mont.sub_32x4(a0na1, a2na3iimag), irot3));
      }
      if (s + 1 != (1 << len))
        irot_simd =
            mont.redc_32x4(irot_simd, irate3_simd[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_forward(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 1);
    u32 rot = mont.one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 l = F[i + offset];
        u32 r = mont.redc(F[i + offset + p], rot);
        F[i + offset] = mont.add(l, r);
        F[i + offset + p] = mont.sub(l, r);
      }
      if (s + 1 != (1 << len))
        rot = mont.redc(rot, rate2[countr_zero(~(u32)(s))]);
    }
  }

  void radix2_inverse(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 1);
    u32 irot = mont.one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i++) {
        u32 l = F[i + offset];
        u32 r = F[i + offset + p];
        F[i + offset] = mont.add(l, r);
        F[i + offset + p] = mont.redc(mont.sub(l, r), irot);
      }
      if (s + 1 != (1 << len))
        irot = mont.redc(irot, irate2[countr_zero(~(u32)(s))]);
    }
  }

  // assume F is NOT in montgomery domain
  template <int num_skip_round> void transform_forward(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = countr_zero((u32)n);

    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], mont.from_32x4(a0));
    }

    int len = 0; // a[i, i+(n>>len), i+2*(n>>len), ..] is transformed
    while (len < h - num_skip_round) {
      if (len + 2 < h - num_skip_round) {
        radix4_forward_simd(len, h, F);
        len += 2;
      } else {
        radix2_forward(len, h, F);
        len += 1;
      }
    }
  }

  // assume F is in montgomery domain
  template <int num_skip_round> void transform_inverse(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = countr_zero((u32)n);

    int len = h - num_skip_round;
    while (len > 0) {
      if (len >= 2 && h - len >= 2) {
        len -= 2;
        radix4_inverse_simd(len, h, F);
      } else {
        len -= 1;
        radix2_inverse(len, h, F);
      }
    }

    // const u32 invn = modinv(n);
    // for (int i = 0; i < n; i++) {
    //   F[i] = mont.mul(mont.get(F[i]), invn);
    // }
    const u32 multiplier = (1u << (32 - h + num_skip_round)) % mod;
    // +2 because of the factor four when doing pointwise_product
    const uint32x4_t multiplier_simd = vdupq_n_u32(multiplier);
    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], mont.get_32x4(mont.redc_32x4(a0, multiplier_simd)));
    }
  }

  // calculate pointwise product mod x^4 - w^4
  // assume a, b are in montgomery domain
  void pointwise_product_modx4nw(u32 a[], u32 b[], int n) {
    const int h = countr_zero((u32)n);
    const int len = h - 2;

    u32 rot4 = mont.one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);

      uint32x4_t va = vld1q_u32(&a[offset]);
      uint32x4_t vb = vld1q_u32(&b[offset]);
      uint32x4_t w_vb = mont.redc_32x4(vb, vdupq_n_u32(rot4));

      uint32x4_t vres = vdupq_n_u32(0);

      uint32x4_t vb_rot, ax;
#define MUL_HELPER(x)                                                          \
  vb_rot = (x == 0 ? vb : vextq_u32(w_vb, vb, (4 - x)));                       \
  ax = vdupq_laneq_u32(va, x);                                                 \
  vres = mont.add_32x4(vres, mont.redc_32x4(ax, vb_rot));

      MUL_HELPER(0)
      MUL_HELPER(1)
      MUL_HELPER(2)
      MUL_HELPER(3)
#undef MUL_HELPER

      vst1q_u32(&a[offset], vres);

      if (s + 1 != (1 << len))
        rot4 = mont.redc(rot4, rate3_4[countr_zero(~(u32)(s))]);
    }
  }

  // calculate pointwise product mod x^2 - w^2
  // assume a, b are in montgomery domain
  void pointwise_product_modx2nw(u32 a[], u32 b[], int n) {
    const int h = countr_zero((u32)n);
    const int len = h - 1;

    u32 rot = mont.one();
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);

      u32 a0 = a[offset], a1 = a[offset + 1];
      u32 b0 = b[offset], b1 = b[offset + 1];
      u32 rot2 = mont.redc(rot, rot);
      a[offset] =
          mont.add(mont.redc(a0, b0), mont.redc(mont.redc(a1, b1), rot2));
      a[offset + 1] = mont.add(mont.redc(a0, b1), mont.redc(a1, b0));

      if (s + 1 != (1 << len))
        rot = mont.redc(rot, rate2[countr_zero(~(u32)(s))]);
    }
  }

  void inplace_convolve(u32 *__restrict__ a, u32 *__restrict__ b, int n) {
    const u32 mod2 = mod * 2;
    for (int i = 0; i < n; i++) {
      if (a[i] >= mod2)
        a[i] -= mod2;
      if (a[i] >= mod2)
        a[i] -= mod2;
      if (a[i] >= mod)
        a[i] -= mod;
    }
    for (int i = 0; i < n; i++) {
      if (b[i] >= mod2)
        b[i] -= mod2;
      if (b[i] >= mod2)
        b[i] -= mod2;
      if (b[i] >= mod)
        b[i] -= mod;
    }
    transform_forward<1>(a, n);
    transform_forward<1>(b, n);
    pointwise_product_modx2nw(a, b, n);
    transform_inverse<1>(a, n);
  }
};
