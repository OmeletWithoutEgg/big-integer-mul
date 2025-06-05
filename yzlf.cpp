#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bigint.h"

#include <bits/stdc++.h>
namespace __yzlf {
using u32 = unsigned int;
using i64 = long long;
using u64 = unsigned long long;
using u128 = __uint128_t;
using idt = std::size_t;
template <class T> inline T *cpy(T *f, const T *g, idt n) {
  return (T *)memcpy(f, g, n * sizeof(T));
}
template <class T> inline T *clr(T *f, idt n) {
  return (T *)memset(f, 0, n * sizeof(T));
}
using f64 = double;
using cpx = std::complex<f64>;
constexpr idt bcl(idt x) { return x < 2 ? 1 : idt(2) << std::__lg(x - 1); }
namespace __fft {
using f64 = double;
using ldb = long double;
struct cpx {
  f64 x, y;
  cpx() = default;
  cpx(f64 xx, f64 yy = 0.) : x(xx), y(yy) {}
  cpx operator+(cpx b) const { return {x + b.x, y + b.y}; }
  cpx operator-(cpx b) const { return {x - b.x, y - b.y}; }
  cpx operator*(cpx b) const { return {x * b.x - y * b.y, x * b.y + y * b.x}; }
  // a*conj(b)
  friend cpx mulT(cpx a, cpx b) {
    return {a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y};
  }
  //(a-b)*i
  friend cpx subI(cpx a, cpx b) { return {b.y - a.y, a.x - b.x}; }
  // mod (x^2-1)
  friend cpx mulY(cpx a, cpx b) {
    return {a.x * b.x + a.y * b.y, a.x * b.y + a.y * b.x};
  }
  cpx operator-() { return {-x, -y}; }
};
inline cpx Wn(ldb a) { return {f64(std::cos(a)), f64(std::sin(a))}; }
inline cpx conj(cpx x) { return {x.x, -x.y}; }
struct ffter {
  std::vector<cpx> wn;
  ffter() : wn{1.} {}
  void reserve(idt l) {
    idt sz = wn.size();
    if (l > sz * 2) {
      int t = std::__lg(l), t2 = t >> 1;
      idt l2 = idt(1) << t2;
      std::vector<cpx> bas(l2 << 1);
      const auto p0 = std::acos(-1.l) / l2, p1 = p0 / l2;
      for (idt i = 0, j = (l2 * 3) >> 1, p = 0; i < l2;
           p -= l2 - (j >> __builtin_ctzll(++i))) {
        bas[i] = Wn(i64(p) * p0), bas[i | l2] = Wn(i64(p) * p1);
      }
      wn.resize(l >> 1);
      for (idt i = sz; i < (l >> 1); ++i) {
        wn[i] = bas[i & (l2 - 1)] * bas[l2 | (i >> t2)];
      }
    }
  }
  void dif(cpx *f, idt n) {
    idt L = n >> 1;
    if (__builtin_ctzll(n) & 1) {
      for (idt j = 0; j < L; ++j) {
        cpx x = f[j], y = f[j + L];
        f[j] = x + y, f[j + L] = x - y;
      }
      L >>= 1;
    }
    L >>= 1;
    for (idt l = L << 2; L; l = L, L >>= 2) {
      for (idt j = 0; j < L; ++j) {
        cpx f0 = f[j], f1 = f[j + L], f2 = f[j + L * 2], f3 = f[j + L * 3];
        cpx g0 = f0 + f2, g1 = f1 + f3, g2 = f0 - f2, g3 = subI(f1, f3);
        f[j] = g0 + g1, f[j + L] = g0 - g1, f[j + L * 2] = g2 + g3,
        f[j + L * 3] = g2 - g3;
      }
      for (idt i = l, k = 1; i < n; i += l, ++k) {
        auto r1 = wn[k * 2], r2 = wn[k], r3 = r1 * r2;
        for (idt j = i; j < i + L; ++j) {
          cpx f0 = f[j], f1 = f[j + L] * r1, f2 = f[j + L * 2] * r2,
              f3 = f[j + L * 3] * r3;
          cpx g0 = f0 + f2, g1 = f1 + f3, g2 = f0 - f2, g3 = subI(f1, f3);
          f[j] = g0 + g1, f[j + L] = g0 - g1, f[j + L * 2] = g2 + g3,
          f[j + L * 3] = g2 - g3;
        }
      }
    }
  }
  void dit(cpx *f, idt n) {
    idt L = 1;
    for (idt l = L << 2; L < (n >> 1); L = l, l <<= 2) {
      for (idt j = 0; j < L; ++j) {
        cpx f0 = f[j], f1 = f[j + L], f2 = f[j + L * 2], f3 = f[j + L * 3];
        cpx g0 = f0 + f1, g1 = f0 - f1, g2 = f2 + f3, g3 = subI(f3, f2);
        f[j] = g0 + g2, f[j + L] = g1 + g3, f[j + L * 2] = g0 - g2,
        f[j + L * 3] = g1 - g3;
      }
      for (idt i = l, k = 1; i < n; i += l, ++k) {
        auto r1 = wn[k * 2], r2 = wn[k], r3 = r1 * r2;
        for (idt j = i; j < i + L; ++j) {
          cpx f0 = f[j], f1 = f[j + L], f2 = f[j + L * 2], f3 = f[j + L * 3];
          cpx g0 = f0 + f1, g1 = f0 - f1, g2 = f2 + f3, g3 = subI(f3, f2);
          f[j] = g0 + g2, f[j + L] = mulT(g1 + g3, r1),
          f[j + L * 2] = mulT(g0 - g2, r2), f[j + L * 3] = mulT(g1 - g3, r3);
        }
      }
    }
    if (L != n) {
      for (idt j = 0; j < L; ++j) {
        cpx x = f[j], y = f[j + L];
        f[j] = x + y, f[j + L] = x - y;
      }
    }
  }
  void __fconv(cpx *F, cpx *G, idt lm) {
    reserve(lm), dif(F, lm), dif(G, lm);
    f64 fx = 1. / lm, fx2 = 0.25 * fx;
    F[0] = mulY(F[0], G[0]) * fx, F[1] = F[1] * G[1] * fx;
    for (idt k = 2, m = 3; k < lm; k <<= 1, m <<= 1) {
      for (idt i = k, j = i + k - 1; i < m; ++i, --j) {
        cpx oi = F[i] + conj(F[j]), hi = F[i] - conj(F[j]);
        cpx Oi = G[i] + conj(G[j]), Hi = G[i] - conj(G[j]);
        cpx r0 = oi * Oi - hi * Hi * ((i & 1) ? -wn[i >> 1] : wn[i >> 1]),
            r1 = Oi * hi + oi * Hi;
        F[i] = (r0 + r1) * fx2, F[j] = conj(r0 - r1) * fx2;
      }
    }
    dit(F, lm);
  }
  void fconv(f64 *f, f64 *g, idt lm) { __fconv((cpx *)f, (cpx *)g, lm / 2); }
} fft;
} // namespace __fft
void split_b2(u64 *_f, f64 *g, idt n, int k) {
  auto f = (u32 *)_f;
  u64 tmp = f[0], msk = (u64(1) << k) - 1;
  idt i = 1, j = 0;
  int w = 32;
  while (i < (n << 1)) {
    g[j++] = i64(tmp & msk), tmp >>= k, w -= k;
    if (w < 32) {
      tmp |= u64(f[i++]) << w, w += 32;
    }
  }
  while (w > 0) {
    g[j++] = i64(tmp & msk), tmp >>= k, w -= k;
  }
}
void merge_b2(u64 *f, f64 *g, idt n, int k) {
  idt i = 0, j = 0;
  int w = 0;
  __uint128_t tmp = 0;
  while (i < n) {
    while (w < 64) {
      tmp += __uint128_t(u64(i64(g[j++] + 0.5))) << w, w += k;
    }
    if (w >= 64) {
      f[i++] = u64(tmp), tmp >>= 64, w -= 64;
    }
  }
}
inline auto stou_8_b16(const char *s) {
  const u64 t = (*(u64 *)s);
  const u64 t0 = t & 0x4040404040404040, t1 = t - 0x3030303030303030;
  const u64 u = t1 - (t0 >> 6) * ('A' - '0' - 10);
  const u64 u0 = ((u << 4) | (u >> 8)) & 0xff00ff00ff00ff;
  const u64 u1 = ((u0 << 8) | (u0 >> 16)) & 0xffff0000ffff;
  return u32((u1 << 16) | (u1 >> 32));
}
inline auto utos_8_b16(char *s, u32 x) {
  const u64 u = ((x >> 16) | (u64(x) << 32)) & 0xffff0000ffff;
  const u64 u0 = ((u >> 8) | (u << 16)) & 0xff00ff00ff00ff;
  const u64 u1 = ((u0 >> 4) | (u0 << 8)) & 0xf0f0f0f0f0f0f0f;
  const u64 z = u1 + 0x3636363636363636;
  const u64 z0 = z & 0x4040404040404040, z1 = u1 + 0x3030303030303030;
  *(u64 *)s = z1 + (z0 >> 6) * ('A' - '0' - 10);
}
inline auto stou_16_B16(const char *s) {
  return u64(stou_8_b16(s)) << 32 | stou_8_b16(s + 8);
}
inline auto stou_B16(const char *s, idt len) {
  static constexpr auto tb = [] {
    std::array<char, 128> t = {};
    for (char c = '0'; c <= '9'; ++c) {
      t[c] = c - '0';
    }
    for (char c = 'A'; c <= 'Z'; ++c) {
      t[c] = c - 'A' + 10;
    }
    return t;
  }();
  u64 res = 0;
  for (idt i = 0; i < len; ++i) {
    res = res << 4 | tb[s[i]];
  }
  return res;
}
inline auto utos_16_b16(char *s, u64 x) {
  utos_8_b16(s, x >> 32), utos_8_b16(s + 8, x);
}
inline auto utos_B16(char *s, u64 x) {
  static constexpr auto tb = [] {
    std::array<char, 16> t = {};
    for (int i = 0; i < 10; ++i) {
      t[i] = '0' + i;
    }
    for (int i = 10; i < 16; ++i) {
      t[i] = 'A' + i - 10;
    }
    return t;
  }();
  idt i = 16 - (__builtin_clzll(x) >> 2);
  for (idt j = 0; j < i; ++j) {
    s[j] = tb[(x >> ((i - j - 1) * 4)) & 15];
  }
  return i;
}
struct _buint {
  u64 *a;
  idt sz, cp;
  struct _uinit {
    idt l;
  };
  void unsv_res(idt n) & {
    if (n > cp) {
      delete[] a, a = new u64[n], cp = n;
    }
  }
  u64 &operator[](idt p) { return a[p]; }
  u64 operator[](idt p) const { return a[p]; }

public:
  _buint() : a(nullptr), sz(0), cp(0) {}
  ~_buint() { delete[] a; }
  _buint(u64 x) : a(new u64[2]), sz(x > 0), cp(2) { a[0] = x; }
  operator bool() const { return sz != 0; }
  _buint(const _buint &x)
      : a(cpy(new u64[x.sz], x.a, x.sz)), sz(x.sz), cp(x.sz) {}
  _buint(_buint &&x) : a(x.a), sz(x.sz), cp(x.cp) { x.a = nullptr; }
  _buint &operator=(const _buint &x) & {
    unsv_res(x.sz), cpy(a, x.a, sz = x.sz);
    return *this;
  }
  _buint &operator=(_buint &&x) & {
    delete[] a, a = x.a, sz = x.sz, cp = x.cp, x.a = nullptr;
    return *this;
  }
  friend auto operator<=>(const _buint &a, const _buint &b) {
    if (a.sz != b.sz) {
      return a.sz <=> b.sz;
    }
    for (idt i = a.sz - 1; ~i; --i) {
      if (a[i] != b[i]) {
        return a[i] <=> b[i];
      }
    }
    return 0 <=> 0;
  }
  friend auto operator==(const _buint &a, const _buint &b) {
    return (a <=> b) == 0;
  }
  _buint(_uinit x) : a(new u64[x.l]), cp(x.l) {}
  void shrk() & {
    while (sz > 0 && a[sz - 1] == 0) {
      --sz;
    }
  }
  auto _radd(const _buint &b) const {
    _buint c(_uinit{sz + 1});
    idt i = 0;
    bool ca = false;
    for (; i < b.sz; ++i) {
      ca = __builtin_uaddll_overflow(a[i], ca, &c[i]);
      ca |= __builtin_uaddll_overflow(c[i], b[i], &c[i]);
    }
    for (; i < sz && ca; ++i) {
      ca = __builtin_uaddll_overflow(a[i], ca, &c[i]);
    }
    if (ca) {
      c.sz = sz + 1, c[sz] = 1;
    } else {
      c.sz = sz, cpy(c.a + i, a + i, sz - i);
    }
    return c;
  }
  _buint _rsub(const _buint &b) const {
    _buint c(_uinit{sz});
    idt i = 0;
    bool ca = false;
    for (; i < b.sz; ++i) {
      ca = __builtin_usubll_overflow(a[i], ca, &c[i]);
      ca |= __builtin_usubll_overflow(c[i], b[i], &c[i]);
    }
    for (; ca; ++i) {
      ca = __builtin_usubll_overflow(a[i], ca, &c[i]);
    }
    if (i == sz) {
      c.sz = sz, c.shrk();
    } else {
      c.sz = sz, cpy(c.a + i, a + i, sz - i);
    }
    return c;
  }
  _buint _rmul_bf(const _buint &b) const {
    _buint c(_uinit{sz + b.sz});
    clr(c.a, sz);
    for (idt i = 0; i < b.sz; ++i) {
      u64 t1 = 0;
      for (idt j = 0; j < sz; ++j) {
        auto tmp = b[i] * u128(a[j]);
        tmp += c[i + j], tmp += t1;
        auto lo = u64(tmp), hi = u64(tmp >> 64);
        c[i + j] = lo, t1 = hi;
      }
      c[i + sz] = t1;
    }
    c.sz = sz + b.sz, c.shrk();
    return c;
  }
  _buint _rmul_fft(const _buint &b) const {
    idt u = sz + b.sz;
    static constexpr auto gkk = [] {
      std::array<idt, 11> ar = {};
      for (idt i = 0; i < 10; ++i) {
        ar[i] = idt(19 - i) << (2 * i + 4);
      }
      ar[10] = idt(-1);
      return ar;
    }();
    int kk = 0;
    while (u > gkk[kk]) {
      ++kk;
    }
    kk = 19 - kk;
    idt lm = idt(2) << (__builtin_clzll(((u * 64) / kk) + 1) ^ 63);
    std::vector<f64> F(lm), G(lm);
    split_b2(a, F.data(), sz, kk);
    split_b2(b.a, G.data(), b.sz, kk);
    __fft::fft.fconv(F.data(), G.data(), lm);
    _buint c(_uinit{u});
    merge_b2(c.a, F.data(), u, kk);
    c.sz = u, c.shrk();
    return c;
  }
  _buint _rmul(const _buint &b) const {
    if (!b) {
      return {};
    }
    if (b.sz <= 50) {
      return _rmul_bf(b);
    }
    return _rmul_fft(b);
  }
  friend void fr_str(_buint &x, std::string_view s) {
    idt n = s.size(), i = 0;
    x.unsv_res((n + 15) >> 4);
    for (; n > 15; n -= 16) {
      x[i++] = stou_16_B16(s.data() + n - 16);
    }
    if (n) {
      x[i++] = stou_B16(s.data(), n);
    }
    x.sz = i, x.shrk();
  }
  friend void to_str(const _buint &x, std::string &s) {
    s.assign(x.sz * 16, '0');
    if (!x) {
      s = "0";
      return;
    }
    idt i = utos_B16(s.data(), x[x.sz - 1]);
    for (idt j = x.sz - 2; ~j; --j) {
      utos_16_b16(s.data() + i, x[j]), i += 16;
    }
    s.resize(i);
  }

public:
  friend _buint operator+(const _buint &a, const _buint &b) {
    return a.sz < b.sz ? b._radd(a) : a._radd(b);
  }
  friend _buint operator-(const _buint &a, const _buint &b) {
    assert(a >= b);
    return a._rsub(b);
  }
  friend _buint operator*(const _buint &a, const _buint &b) {
    return a.sz < b.sz ? b._rmul(a) : a._rmul(b);
  }
};
static void fr_bigint(_buint &x, const bigint *rhs) {
  static_assert(LIMB_BITS == 64);
  x.unsv_res(BIGINT_BITS);
  for (idt i = 0; i < BIGINT_LIMBS; i++)
    x[i] = rhs->limbs[i];
  x.sz = BIGINT_LIMBS;
  x.shrk();
}
static void to_bigint(const _buint &x, bigint *rhs) {
  static_assert(LIMB_BITS == 64);
  for (idt i = 0; i < x.sz; i++)
    rhs->limbs[i] = x[i];
  for (idt i = x.sz; i < BIGINT_LIMBS; i++)
    rhs->limbs[i] = 0;
}
} // namespace __yzlf

extern "C" {

void bigint_set_zero(bigint *r) { memset(r->limbs, 0, sizeof(r->limbs)); }

void bigint_copy(bigint *dest, const bigint *src) {
  memcpy(dest->limbs, src->limbs, sizeof(src->limbs));
}

static uint64_t xorshift64(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

void bigint_urandom(uint64_t *seed, bigint *r, uint32_t bits) {
  assert(bits % LIMB_BITS == 0);
  assert(bits * 2 <= BIGINT_BITS);
  for (size_t i = 0; i < bits / LIMB_BITS; i++) {
    /* r->limbs[i] = (xorshift64(seed) & ((1ull << LIMB_BITS) - 1)); */
    r->limbs[i] = xorshift64(seed);
  }
  for (size_t i = bits / LIMB_BITS; i < BIGINT_LIMBS; i++) {
    r->limbs[i] = 0;
  }
}

// --- bigint 乘法 (base 2^LIMB_BITS) ---
void bigint_mul(bigint *res, const bigint *a, const bigint *b) {
  __yzlf::_buint A, B;
  __yzlf::fr_bigint(A, a);
  __yzlf::fr_bigint(B, b);
  A = A * B;
  __yzlf::to_bigint(A, res);
}
}
