#include "distance.h"
#include "api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static float dot_naive(const float* a, const float* b, unsigned n){
  float s=0.f; for(unsigned i=0;i<n;++i) s+=a[i]*b[i]; return s;
}
static float l2s_naive(const float* a, const float* b, unsigned n){
  float s=0.f; for(unsigned i=0;i<n;++i){ float d=a[i]-b[i]; s+=d*d; } return s;
}

int main(void){
  jv_runtime_init();
  const unsigned n=1023; // uneven to force tail path
  float* a=(float*)malloc(n*sizeof(float));
  float* b=(float*)malloc(n*sizeof(float));
  for(unsigned i=0;i<n;++i){ a[i]=(float)i*0.001f; b[i]=(float)(i%7)*0.01f; }

  float d0 = dot_naive(a,b,n);
  float d1 = jv_dot_f32(a,b,n);
  assert(fabsf(d0 - d1) < 1e-3f);

  float l0 = l2s_naive(a,b,n);
  float l1 = jv_l2sq_f32(a,b,n);
  assert(fabsf(l0 - l1) < 1e-3f);

  printf("distance ok\n");
  free(a); free(b);
  return 0;
}
