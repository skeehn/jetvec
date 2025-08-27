#include "distance.h"

// Function pointers default to scalar; set in jv_runtime_init (api.c)
jv_dot_f32_fn  jv_dot_f32  = jv_dot_f32_scalar;
jv_l2sq_f32_fn jv_l2sq_f32 = jv_l2sq_f32_scalar;
