#include "api.h"
#include "distance.h"
#include <stdlib.h>
#include <string.h>

struct jv_index {
  uint32_t dim;
  jv_metric_t metric;
  // TODO: HNSW/IVF/PQ state
};

void jv_runtime_init(void) {
  // Set scalar defaults
  jv_dot_f32  = jv_dot_f32_scalar;
  jv_l2sq_f32 = jv_l2sq_f32_scalar;

  // Runtime CPU dispatch (x86 for now)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) {
    extern float jv_dot_f32_avx2  (const float*, const float*, uint32_t);
    extern float jv_l2sq_f32_avx2 (const float*, const float*, uint32_t);
    jv_dot_f32  = jv_dot_f32_avx2;
    jv_l2sq_f32 = jv_l2sq_f32_avx2;
  }
#endif
#endif
}

jv_index_t* jv_create(void) {
  jv_runtime_init();
  jv_index_t* idx = (jv_index_t*)calloc(1, sizeof(jv_index_t));
  return idx;
}

void jv_free(jv_index_t* idx) {
  if (!idx) return;
  // TODO: free internal sections
  free(idx);
}

int jv_build(jv_index_t* idx, const float* vectors, uint64_t n, jv_build_params_t p) {
  if (!idx || !vectors || p.dim==0) return -1;
  idx->dim = p.dim;
  idx->metric = p.metric;
  // TODO: implement HNSW/IVF/PQ build paths
  (void)n; (void)p;
  return 0;
}

int jv_save(jv_index_t* idx, const char* path) {
  (void)idx; (void)path;
  // TODO: implement on-disk writer (header + sections + CRC)
  return 0;
}

jv_index_t* jv_load(const char* path) {
  (void)path;
  // TODO: map file, validate header, set pointers
  jv_index_t* idx = jv_create();
  return idx;
}

int jv_search(jv_index_t* idx, const float* query, uint32_t k, uint32_t efSearch, jv_hit_t* out_hits) {
  (void)idx; (void)query; (void)k; (void)efSearch; (void)out_hits;
  // TODO: implement HNSW/IVF/PQ search pipeline
  return 0;
}
