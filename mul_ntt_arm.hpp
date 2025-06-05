#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

using u32 = uint32_t;
using u64 = uint64_t;

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
};
// a * b % mod == get(redc(from(a), from(b)))

struct MontgomeryNeon {
  uint32x4_t mod, R1, R2, xinv;

  MontgomeryNeon(const Montgomery &mont) {
    mod = vdupq_n_u32(mont.mod);
    R1 = vdupq_n_u32(mont.R1);
    R2 = vdupq_n_u32(mont.R2);
    xinv = vdupq_n_u32(mont.xinv);
  }

  uint32x4_t redc(uint64x2_t T_lo, uint64x2_t T_hi) const {
    // Extract low 32 bits from T for multiplication with xinv
    uint32x4_t T_low = vcombine_u32(vmovn_u64(T_lo), vmovn_u64(T_hi));

    // Compute m = T_low * xinv (mod 2^32)
    uint32x4_t m = vmulq_u32(T_low, xinv);

    // Multiply m with mod
    uint64x2_t m_mod_lo = vmull_u32(vget_low_u32(mod), vget_low_u32(m));
    uint64x2_t m_mod_hi = vmull_u32(vget_high_u32(mod), vget_high_u32(m));

    // Add m * mod to T
    T_lo = vaddq_u64(T_lo, m_mod_lo);
    T_hi = vaddq_u64(T_hi, m_mod_hi);

    // Shift right by W (32 bits) - take high 32 bits
    uint32x4_t result =
        vcombine_u32(vshrn_n_u64(T_lo, 32), vshrn_n_u64(T_hi, 32));

    // Conditional subtraction: if result >= mod then result -= mod
    result = vminq_u32(result, vsubq_u32(result, mod));

    return result;
  }

  uint32x4_t redc(uint32x4_t a, uint32x4_t b) const {
    // Multiply a * b using widening multiplication
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t a_hi = vget_high_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint32x2_t b_hi = vget_high_u32(b);

    uint64x2_t ab_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t ab_hi = vmull_u32(a_hi, b_hi);

    return redc(ab_lo, ab_hi);
  }

  uint32x4_t from(uint32x4_t x) const { return redc(x, R2); }

  uint32x4_t get(uint32x4_t a) const {
    uint32x4_t one_vec = vdupq_n_u32(1);
    return redc(a, one_vec);
  }

  uint32x4_t one() const { return R1; }

  uint32x4_t add(uint32x4_t a, uint32x4_t b) const {
    uint32x4_t sum = vaddq_u32(a, b);
    return vminq_u32(sum, vsubq_u32(sum, mod));
  }
  uint32x4_t sub(uint32x4_t a, uint32x4_t b) const {
    uint32x4_t diff = vsubq_u32(a, b);
    return vminq_u32(diff, vaddq_u32(diff, mod));
  }
};

template <u32 mod, u32 G, int maxn> struct NTT {
  static_assert(maxn == (maxn & -maxn));

  static constexpr int simd_size = 4;

  Montgomery mont;
  MontgomeryNeon mont_simd;
  static constexpr int rank2 = __builtin_ctz(mod - 1);
  std::array<u32, rank2 + 1> root;  // root[i]^(2^i) == 1
  std::array<u32, rank2 + 1> iroot; // root[i] * iroot[i] == 1

  // root, iroot 陣列不在 montgomery domain，但以下的陣列都在 montgomery domain 裡面
  std::array<u32, std::max(0, rank2 - 2 + 1)> rate2;
  std::array<u32, std::max(0, rank2 - 2 + 1)> irate2;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> rate3_simd;
  std::array<u32, std::max(0, rank2 - 3 + 1)> irate3;
  std::array<uint32x4_t, std::max(0, rank2 - 3 + 1)> irate3_simd;

  NTT() : mont(mod), mont_simd(mont) {
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
        u32 power[4];
        power[0] = mont.one();
        for (int j = 1; j < simd_size; j++) {
          power[j] = mont.redc(power[j - 1], rate3[i]);
        }
        rate3_simd[i] = vld1q_u32(power);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        u32 power[4];
        power[0] = mont.one();
        for (int j = 1; j < simd_size; j++) {
          power[j] = mont.redc(power[j - 1], irate3[i]);
        }
        irate3_simd[i] = vld1q_u32(power);
      }
    }
  }

  void radix4_forward(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
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
        rot = mont.redc(rot, rate3[__builtin_ctz(~(u32)(s))]);
    }
  }

  void radix4_forward_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t rot_simd = mont_simd.one(),
               imag = vdupq_n_u32(mont.from(root[2]));
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      for (int i = 0; i < p; i += simd_size) {
        uint32x4_t rot = vdupq_laneq_u32(rot_simd, 1);
        uint32x4_t rot2 = vdupq_laneq_u32(rot_simd, 2);
        uint32x4_t rot3 = vdupq_laneq_u32(rot_simd, 3);

        uint32x4_t a0 = vld1q_u32(&F[i + offset]);
        uint32x4_t a1 = mont_simd.redc(vld1q_u32(&F[i + offset + p]), rot);
        uint32x4_t a2 = mont_simd.redc(vld1q_u32(&F[i + offset + 2 * p]), rot2);
        uint32x4_t a3 = mont_simd.redc(vld1q_u32(&F[i + offset + 3 * p]), rot3);
        uint32x4_t a1na3imag = mont_simd.redc(mont_simd.sub(a1, a3), imag);

        auto a0pa2 = mont_simd.add(a0, a2);
        auto a1pa3 = mont_simd.add(a1, a3);
        auto a0na2 = mont_simd.sub(a0, a2);
        vst1q_u32(&F[i + offset], mont_simd.add(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + p], mont_simd.sub(a0pa2, a1pa3));
        vst1q_u32(&F[i + offset + 2 * p], mont_simd.add(a0na2, a1na3imag));
        vst1q_u32(&F[i + offset + 3 * p], mont_simd.sub(a0na2, a1na3imag));
      }
      if (s + 1 != (1 << len))
        rot_simd =
            mont_simd.redc(rot_simd, rate3_simd[__builtin_ctz(~(u32)(s))]);
    }
  }

  void radix4_inverse(int len, int h, u32 F[]) {
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
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
        F[i + offset + p] = mont.redc(mont.add(mont.sub(a0, a1), a2na3iimag), irot);
        F[i + offset + 2 * p] = mont.redc(mont.sub(mont.add(a0, a1), mont.add(a2, a3)), irot2);
        F[i + offset + 3 * p] = mont.redc(mont.sub(mont.sub(a0, a1), a2na3iimag), irot3);
      }
      if (s + 1 != (1 << len))
        irot = mont.redc(irot, irate3[__builtin_ctz(~(u32)(s))]);
    }
  }

  void radix4_inverse_simd(int len, int h, u32 F[]) {
    // XXX
    const int p = 1 << (h - len - 2);
    assert(p >= 4 && p % 4 == 0);
    uint32x4_t irot_simd = mont_simd.one(),
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
        uint32x4_t a2na3iimag = mont_simd.redc(mont_simd.sub(a2, a3), iimag);

        auto a0pa1 = mont_simd.add(a0, a1);
        auto a2pa3 = mont_simd.add(a2, a3);
        auto a0na1 = mont_simd.sub(a0, a1);
        vst1q_u32(&F[i + offset], mont_simd.add(a0pa1, a2pa3));
        vst1q_u32(&F[i + offset + p],
                  mont_simd.redc(mont_simd.add(a0na1, a2na3iimag), irot));
        vst1q_u32(&F[i + offset + 2 * p],
                  mont_simd.redc(mont_simd.sub(a0pa1, a2pa3), irot2));
        vst1q_u32(&F[i + offset + 3 * p],
                  mont_simd.redc(mont_simd.sub(a0na1, a2na3iimag), irot3));
      }
      if (s + 1 != (1 << len))
        irot_simd =
            mont_simd.redc(irot_simd, irate3_simd[__builtin_ctz(~(u32)(s))]);
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
        rot = mont.redc(rot, rate2[__builtin_ctz(~(u32)(s))]);
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
        irot = mont.redc(irot, irate2[__builtin_ctz(~(u32)(s))]);
    }
  }

  // assume F is NOT in montgomery domain
  void transform_forward(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = __builtin_ctz((u32)n);
    assert(h % 2 == 0);

    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], mont_simd.from(a0));
    }

    int len = 0; // a[i, i+(n>>len), i+2*(n>>len), ..] is transformed
    while (len < h - 2) {
      radix4_forward_simd(len, h, F);
      len += 2;
    }
  }

  // assume F is in montgomery domain
  void transform_inverse(u32 F[], int n) {
    assert(n == (n & -n));
    assert(n % simd_size == 0);
    const int h = __builtin_ctz((u32)n);
    assert(h % 2 == 0);

    int len = h - 2;
    while (len > 0) {
      len -= 2;
      radix4_inverse_simd(len, h, F);
    }

    // const u32 invn = modinv(n);
    // for (int i = 0; i < n; i++) {
    //   F[i] = mont.mul(mont.get(F[i]), invn);
    // }
    const u32 multiplier = (1u << (32 - h)) % mod;
    const uint32x4_t multiplier_simd = vdupq_n_u32(multiplier);
    for (int i = 0; i < n; i += simd_size) {
      uint32x4_t a0 = vld1q_u32(&F[i]);
      vst1q_u32(&F[i], mont_simd.get(mont_simd.redc(a0, multiplier_simd)));
    }
  }

  // calculate pointwise product mod x^4 - w^k
  // assume a, b are in montgomery domain
  void pointwise_product(u32 a[], u32 b[], int n) {
    const int h = __builtin_ctz((u32)n);
    assert(h % 2 == 0);

    const int len = h - 2;
    // p = 1

    auto twiddle_simd = [&](auto &F, const auto &factor) {
      uint32x4_t rot_simd = mont_simd.one(),
                 imag = vdupq_n_u32(mont.from(root[2]));
      for (int s = 0; s < (1 << len); s++) {
        int offset = s << (h - len);
        uint32x4_t a0 = vld1q_u32(&F[offset]);
        vst1q_u32(&F[offset], mont_simd.redc(a0, rot_simd));
        if (s + 1 != (1 << len))
          rot_simd = mont_simd.redc(rot_simd, factor[__builtin_ctz(~(u32)(s))]);
      }
    };
    twiddle_simd(a, rate3_simd);
    twiddle_simd(b, rate3_simd);

    const uint32x4_t four_simd = vdupq_n_u32(mont.from(4));
    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);

      uint32x4_t va = vld1q_u32(&a[offset]); // 不 rotate
      uint32x4_t vb = vld1q_u32(&b[offset]); // 我們對它 rotate

      uint32x4_t vres = vdupq_n_u32(0);

      for (int x = 0; x < 4; x++) {
        // 將 b 向左循環位移 x
        uint32x4_t vb_rot = vextq_u32(vb, vb, (4 - x) % 4);

        // 對應 a[x]，broadcast 成 vector
        uint32x4_t ax = vdupq_laneq_u32(va, x);

        // 做 a[x] * b[(y+x)%4]（實際就是 b_rot）
        uint64x2_t low = vmull_u32(vget_low_u32(ax), vget_low_u32(vb_rot));
        uint64x2_t high = vmull_u32(vget_high_u32(ax), vget_high_u32(vb_rot));

        // 做 redc（要嘛 vectorized 要嘛轉 scalar）
        uint32x4_t prod = mont_simd.redc(low, high);

        // 累加
        vres = mont_simd.add(vres, prod);
      }

      vst1q_u32(&a[offset], mont_simd.redc(vres, four_simd));
    }
    twiddle_simd(a, irate3_simd);
  }
};
