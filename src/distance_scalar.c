#include "distance.h"

float jv_dot_f32_scalar(const float* a, const float* b, uint32_t n) {
  float s = 0.0f;
  for (uint32_t i=0; i<n; ++i) s += a[i]*b[i];
  return s;
}

float jv_l2sq_f32_scalar(const float* a, const float* b, uint32_t n) {
  float s = 0.0f;
  for (uint32_t i=0; i<n; ++i) {
    float d = a[i]-b[i];
    s += d*d;
  }
  return s;
}
