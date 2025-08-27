#ifndef JV_DISTANCE_H
#define JV_DISTANCE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float (*jv_dot_f32_fn)(const float*, const float*, uint32_t);
typedef float (*jv_l2sq_f32_fn)(const float*, const float*, uint32_t);

// Indirect calls (set at runtime by jv_runtime_init)
extern jv_dot_f32_fn  jv_dot_f32;
extern jv_l2sq_f32_fn jv_l2sq_f32;

// Scalar fallbacks (always available)
float jv_dot_f32_scalar (const float* a, const float* b, uint32_t n);
float jv_l2sq_f32_scalar(const float* a, const float* b, uint32_t n);

// Optional AVX2 variants (linked if compiled)
float jv_dot_f32_avx2  (const float* a, const float* b, uint32_t n);
float jv_l2sq_f32_avx2 (const float* a, const float* b, uint32_t n);

#ifdef __cplusplus
}
#endif
#endif
