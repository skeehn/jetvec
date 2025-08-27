#ifndef JV_API_H
#define JV_API_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { JV_METRIC_L2=0, JV_METRIC_COSINE=1 } jv_metric_t;

typedef struct {
  uint32_t dim;
  jv_metric_t metric;
  uint32_t hnsw_M;
  uint32_t hnsw_efc;
  uint32_t ivf_nlist;
  uint32_t pq_m;
} jv_build_params_t;

typedef struct jv_index jv_index_t;

typedef struct { uint32_t id; float dist; } jv_hit_t;

// Lifecycle
jv_index_t* jv_create(void);
void        jv_free(jv_index_t*);

// Build / IO
int         jv_build(jv_index_t*, const float* vectors, uint64_t n, jv_build_params_t);
int         jv_save(jv_index_t*, const char* path);
jv_index_t* jv_load(const char* path);

// Search
int         jv_search(jv_index_t*, const float* query, uint32_t k, uint32_t efSearch, jv_hit_t* out_hits);

// Init SIMD dispatch (called automatically by jv_create)
void        jv_runtime_init(void);

#ifdef __cplusplus
}
#endif
#endif
