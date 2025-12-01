#include <intrin.h>
#include <cstdio>

static bool has_sse42() {
  int r[4]; __cpuid(r, 1);
  return (r[2] & (1 << 20)) != 0; // ECX.SSE4_2
}

static bool has_avx() {
  int r[4]; __cpuid(r, 1);
  bool osxsave = (r[2] & (1 << 27)) != 0; // OSXSAVE
  bool avxbit  = (r[2] & (1 << 28)) != 0; // AVX
  if (!(osxsave && avxbit)) return false;
  unsigned long long xcr0 = _xgetbv(0);
  return (xcr0 & 0x6) == 0x6; // XMM (bit1) and YMM (bit2) state enabled
}

static bool has_avx2() {
  int r[4]; __cpuid(r, 7);   // leaf 7, subleaf 0
  return (r[1] & (1 << 5)) != 0; // EBX.AVX2
}

int main() {
  std::printf("SSE4.2: %s\n", has_sse42() ? "YES" : "NO");
  std::printf("AVX:    %s\n", has_avx()   ? "YES" : "NO");
  std::printf("AVX2:   %s\n", has_avx2()  ? "YES" : "NO");
}
