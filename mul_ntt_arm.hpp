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
    const u64 R = (u64(1) << W) % mod;
    R1 = u32(R);
    R2 = u32(R * R % mod);
  }
  u32 redc(u64 T) const {
    u64 m = -u32(T) * xinv;
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
};
// a * b % mod == get(redc(from(a), from(b)))

template <u32 mod, u32 G, int maxn> struct NTT {
  static_assert(maxn == (maxn & -maxn));

  Montgomery mont;
  static constexpr int rank2 = std::countr_zero(mod - 1);
  std::array<u32, rank2 + 1> root;  // root[i]^(2^i) == 1
  std::array<u32, rank2 + 1> iroot; // root[i] * iroot[i] == 1

  std::array<u32, std::max(0, rank2 - 2 + 1)> rate2;
  std::array<u32, std::max(0, rank2 - 2 + 1)> irate2;

  std::array<u32, std::max(0, rank2 - 3 + 1)> rate3;
  std::array<u32, std::max(0, rank2 - 3 + 1)> irate3;

  u32 add(u32 a, u32 b) { return a + b >= mod ? a + b - mod : a + b; }
  u32 sub(u32 a, u32 b) { return a < b ? a - b + mod : a - b; }
  u32 mul(u32 a, u32 b) {
    return static_cast<u32>(u64(a) * b % mod);
    // return mont.get(mont.redc(mont.from(a), mont.from(b)));
  }
  u32 modpow(u32 e, u32 p) {
    u32 r = 1;
    while (p) {
      if (p & 1)
        r = mul(r, e);
      e = mul(e, e);
      p >>= 1;
    }
    return r;
  }
  u32 modinv(u32 x) { return modpow(x, mod - 2); }

  constexpr NTT() : mont(mod) {
    root[rank2] = modpow(G, (mod - 1) >> rank2);
    iroot[rank2] = modinv(root[rank2]);
    for (int i = rank2 - 1; i >= 0; i--) {
      root[i] = mul(root[i + 1], root[i + 1]);
      iroot[i] = mul(iroot[i + 1], iroot[i + 1]);
      assert(mul(root[i], iroot[i]) == 1);
    }

    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = mul(root[i + 2], prod);
        irate2[i] = mul(iroot[i + 2], iprod);
        prod = mul(prod, iroot[i + 2]);
        iprod = mul(iprod, root[i + 2]);
        assert(mul(rate2[i], irate2[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 2; i++) {
        rate2[i] = mont.from(rate2[i]);
        irate2[i] = mont.from(irate2[i]);
      }
    }
    {
      u32 prod = 1, iprod = 1;
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = mul(root[i + 3], prod);
        irate3[i] = mul(iroot[i + 3], iprod);
        prod = mul(prod, iroot[i + 3]);
        iprod = mul(iprod, root[i + 3]);
        assert(mul(rate3[i], irate3[i]) == 1);
      }
      for (int i = 0; i <= rank2 - 3; i++) {
        rate3[i] = mont.from(rate3[i]);
        irate3[i] = mont.from(irate3[i]);
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
        u32 a1na3imag = mont.redc(sub(a1, a3), imag);

        F[i + offset] = add(add(a0, a2), add(a1, a3));
        F[i + offset + p] = sub(add(a0, a2), add(a1, a3));
        F[i + offset + 2 * p] = add(sub(a0, a2), a1na3imag);
        F[i + offset + 3 * p] = sub(sub(a0, a2), a1na3imag);
      }
      if (s + 1 != (1 << len))
        rot = mont.redc(rot, rate3[std::countr_zero(~(u32)(s))]);
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

        u32 a2na3iimag = mont.redc(sub(a2, a3), iimag);

        F[i + offset] = add(add(a0, a1), add(a2, a3));
        F[i + offset + p] = mont.redc(add(sub(a0, a1), a2na3iimag), irot);
        F[i + offset + 2 * p] = mont.redc(sub(add(a0, a1), add(a2, a3)), irot2);
        F[i + offset + 3 * p] = mont.redc(sub(sub(a0, a1), a2na3iimag), irot3);
      }
      if (s + 1 != (1 << len))
        irot = mont.redc(irot, irate3[std::countr_zero(~(u32)(s))]);
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
        F[i + offset] = add(l, r);
        F[i + offset + p] = sub(l, r);
      }
      if (s + 1 != (1 << len))
        rot = mont.redc(rot, rate2[std::countr_zero(~(u32)(s))]);
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
        F[i + offset] = add(l, r);
        F[i + offset + p] = mont.redc(sub(l, r), irot);
      }
      if (s + 1 != (1 << len))
        irot = mont.redc(irot, irate2[std::countr_zero(~(u32)(s))]);
    }
  }

  // assume F is NOT in montgomery domain
  void transform_forward(u32 F[], int n) {
    for (int i = 0; i < n; i++)
      F[i] = mont.from(F[i]);

    const int h = std::countr_zero((u32)n);
    assert(h % 2 == 0);
    int len = 0; // a[i, i+(n>>len), i+2*(n>>len), ..] is transformed
    while (len < h - 2) {
      radix4_forward(len, h, F);
      len += 2;
    }
  }

  // assume F is in montgomery domain
  void transform_inverse(u32 F[], int n) {
    const int h = std::countr_zero((u32)n);
    assert(h % 2 == 0);

    int len = h - 2;
    while (len > 0) {
      len -= 2;
      radix4_inverse(len, h, F);
    }

    // const u32 invn = modinv(n);
    // for (int i = 0; i < n; i++) {
    //   F[i] = mul(mont.get(F[i]), invn);
    // }
    const u32 multiplier = (1u << (32 - h)) % mod;
    for (int i = 0; i < n; i++) {
      F[i] = mont.redc(mont.redc(F[i], multiplier));
    }
  }

  // calculate pointwise product mod x^4 - w^k
  // assume a, b are in montgomery domain
  void pointwise_product(u32 a[], u32 b[], int n) {
    const int h = std::countr_zero((u32)n);
    assert(h % 2 == 0);

    const int len = h - 2;
    // p = 1

    // radix4_forward(len, h, a);
    // radix4_forward(len, h, b);
    // for (int i = 0; i < n; i++) {
    //   a[i] = mont.redc(a[i], b[i]);
    // }
    // radix4_inverse(len, h, a);
    // return;

    auto twiddle = [&](auto &F) {
      u32 rot = mont.one();
      for (int s = 0; s < (1 << len); s++) {
        u32 rot2 = mont.redc(rot, rot);
        u32 rot3 = mont.redc(rot2, rot);

        int offset = s << (h - len);
        F[offset] = F[offset];
        F[offset + 1] = mont.redc(F[offset + 1], rot);
        F[offset + 2] = mont.redc(F[offset + 2], rot2);
        F[offset + 3] = mont.redc(F[offset + 3], rot3);
        std::swap(F[offset + 1], F[offset + 3]);
        if (s + 1 != (1 << len))
          rot = mont.redc(rot, rate3[std::countr_zero(~(u32)(s))]);
      }
    };
    twiddle(a);
    twiddle(b);

    for (int s = 0; s < (1 << len); s++) {
      int offset = s << (h - len);
      u32 res[4] = {0, 0, 0, 0};
      for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++) {
          res[(x + y) % 4] = add(res[(x + y) % 4], mont.redc(a[offset + x], b[offset + y]));
        }
      for (int x = 0; x < 4; x++)
        a[offset + x] = mont.redc(res[x], mont.from(4));
    }

    auto inverse_twiddle = [&](auto &F) {
      u32 irot = mont.one();
      for (int s = 0; s < (1 << len); s++) {
        u32 irot2 = mont.redc(irot, irot);
        u32 irot3 = mont.redc(irot2, irot);

        int offset = s << (h - len);
        std::swap(F[offset + 1], F[offset + 3]);
        F[offset] = F[offset];
        F[offset + 1] = mont.redc(F[offset + 1], irot);
        F[offset + 2] = mont.redc(F[offset + 2], irot2);
        F[offset + 3] = mont.redc(F[offset + 3], irot3);
        if (s + 1 != (1 << len))
          irot = mont.redc(irot, irate3[std::countr_zero(~(u32)(s))]);
      }
    };
    inverse_twiddle(a);
  }
};
