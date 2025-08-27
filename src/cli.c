#include "api.h"
#include "distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void bench_distance(uint32_t n, uint32_t dim) {
  float* a = (float*)malloc((size_t)n*dim*sizeof(float));
  float* q = (float*)malloc((size_t)dim*sizeof(float));
  if (!a || !q) { fprintf(stderr,"alloc failed\n"); exit(1); }
  srand(42);
  for (uint32_t i=0;i<n*dim;++i) a[i] = (float)rand()/RAND_MAX;
  for (uint32_t i=0;i<dim;++i)    q[i] = (float)rand()/RAND_MAX;

  volatile float sink = 0.f;
  for (int w=0; w<3; ++w) for (uint32_t i=0;i<n;++i) sink += jv_dot_f32(a + i*dim, q, dim);

  clock_t t0 = clock();
  for (uint32_t i=0;i<n;++i) sink += jv_dot_f32(a + i*dim, q, dim);
  clock_t t1 = clock();
  double ms = 1000.0*(t1 - t0)/CLOCKS_PER_SEC;
  printf("[bench] jv_dot_f32: N=%u dim=%u time=%.2fms (%.2f us/query)\n",
         n, dim, ms, (ms*1000.0)/n);

  t0 = clock();
  for (uint32_t i=0;i<n;++i) sink += jv_l2sq_f32(a + i*dim, q, dim);
  t1 = clock();
  ms = 1000.0*(t1 - t0)/CLOCKS_PER_SEC;
  printf("[bench] jv_l2sq_f32: N=%u dim=%u time=%.2fms (%.2f us/query)\n",
         n, dim, ms, (ms*1000.0)/n);

  free(a); free(q);
  (void)sink;
}

int main(int argc, char** argv) {
  jv_runtime_init();

  if (argc <= 1) {
    printf("jetvec CLI\n");
    printf("  --bench [N dim]     : microbench distance kernels (default 100000 768)\n");
    printf("  build/search/stats  : coming soon\n");
    return 0;
  }

  if (strcmp(argv[1], "--bench")==0) {
    uint32_t N = 100000, D = 768;
    if (argc >= 4) { N = (uint32_t)atoi(argv[2]); D = (uint32_t)atoi(argv[3]); }
    bench_distance(N, D);
    return 0;
  }

  fprintf(stderr, "Unknown command.\n");
  return 1;
}
