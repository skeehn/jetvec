#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#include <stdint.h>

float jv_dot_f32_avx2 (const float* a, const float* b, uint32_t n);
float jv_l2sq_f32_avx2(const float* a, const float* b, uint32_t n);

static inline float hsum256_ps(__m256 v) {
  __m128 vlow  = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  __m128 vsum  = _mm_add_ps(vlow, vhigh);
  vsum = _mm_hadd_ps(vsum, vsum);
  vsum = _mm_hadd_ps(vsum, vsum);
  return _mm_cvtss_f32(vsum);
}

float jv_dot_f32_avx2(const float* a, const float* b, uint32_t n) {
  uint32_t i = 0;
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(va, vb, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
  }
  float sum = hsum256_ps(acc);
  for (; i < n; ++i) sum += a[i]*b[i];
  return sum;
}

float jv_l2sq_f32_avx2(const float* a, const float* b, uint32_t n) {
  uint32_t i = 0;
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vd = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(vd, vd, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(vd, vd));
#endif
  }
  float sum = hsum256_ps(acc);
  for (; i < n; ++i) { float d = a[i]-b[i]; sum += d*d; }
  return sum;
}
#endif
