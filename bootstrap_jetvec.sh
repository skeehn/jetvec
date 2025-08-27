#!/usr/bin/env bash
set -euo pipefail

PROJECT="jetvec"
ROOT_DIR="${PWD}/${PROJECT}"

mkdir -p "$ROOT_DIR"/{src,include,tests,cmake,ffi,bench,.github/workflows}

# =========================
# README (full plan text)
# =========================
cat > "$ROOT_DIR/README.md" << 'EOF'
# JetVec — Tiny ANN Vector Index in C

A tiny, WASM-ready vector index with HNSW + optional IVF-PQ, SIMD hot loops, a memory-mapped on-disk format, and a clean C API. Designed to be the working-set retrieval engine for memory-aware agent systems (e.g., Cilow).

---

## Outcomes (concrete targets)
- **p95 search < 3 ms** on 100k vectors (768d, float32), single core.
- **Index build < 3 min** for 1M vectors (HNSW) with parallel build.
- **Memory use**: ≤ 60% of float32 baseline via IVF-PQ (m×nbits codebooks).
- **Binary size**: ≤ 2MB; no C++ runtime.
- **Bindings**: Python & Node to follow.
- **License**: MIT.

---

## Architecture
**Core modules**
- `distance.[ch]` — L2, dot, cosine; scalar + AVX2 (x86), NEON (future). Runtime CPU dispatch.
- `mmio.[ch]` — cross-platform memory-mapped IO; little-endian on disk; versioned header.
- `hnsw.[ch]` — graph ANN (levels, M, efConstruction/efSearch, neighbor heuristic).
- `ivf.[ch]` — k-means coarse quantizer, inverted lists, residuals.
- `pq.[ch]` — Product Quantization (trainer/encoder/LUT distance).
- `store.[ch]` — on-disk sections (HNSW/IVF/PQ/EMB/Stats), CRC64 checks.
- `api.[ch]` — minimal external API; opaque handles; build/search/save/load.
- `thread.[ch]` — thread pool (pthreads/Win32) for build/search.
- `cli.c` — `jetvec build|search|stats|bench`.

**On-disk format (mmap-friendly)**
```
[JVECHDR v1] magic, version, dim, metric, nvec, sections...
[EMB]   float32/float16 baseline (optional if IVF-PQ only)
[HNSW]  layers, neighbor lists, offsets
[IVF]   centroids (float32), list offsets
[PQ]    codebooks (m × k × d'), codes (byte-packed)
[STATS] build params, checksums, wallclock
```

---

## Algorithms

**Cosine / L2 kernels (SIMD)**
- Cosine as dot if vectors are L2-normalized offline.
- L2: accumulate `(x−y)^2` with SIMD; scalar tail loop.
- Runtime CPUID → pick AVX2/NEON vs scalar.

**HNSW**
- Typical params: `M≈16`, `efConstruction≈200`, `efSearch≈64–256`.
- Build: greedy search down levels; neighbor-diversity heuristic.
- Search: best-first with candidate heap; visited-bitset; hot structs kept compact.

**IVF-PQ**
- k-means coarse quantizer (`nlist` centroids).
- Residuals: `r = x − c(x)`.
- PQ: split residual dims into `m` subspaces; `k=256` codewords each (8-bit codes).
- Scan: IVF prefilter → PQ LUT distances → optional exact refine with SIMD.

**Recall vs latency**
- Small (≤500k): HNSW alone is great.
- Larger: IVF narrows candidates → PQ scan → refine top-R with exact SIMD.

---

## Public C API (sketch)
```c
typedef struct jv_index jv_index_t;
typedef enum { JV_METRIC_L2=0, JV_METRIC_COSINE=1 } jv_metric_t;

typedef struct {
  uint32_t dim;
  jv_metric_t metric;
  uint32_t hnsw_M;       // 0 => disable HNSW (future exact-only index)
  uint32_t hnsw_efc;     // efConstruction
  uint32_t ivf_nlist;    // 0 => disable IVF
  uint32_t pq_m;         // 0 => disable PQ
} jv_build_params_t;

jv_index_t* jv_create(void);
void        jv_free(jv_index_t*);
int         jv_build(jv_index_t*, const float* vecs, uint64_t n, jv_build_params_t);
int         jv_save(jv_index_t*, const char* path);
jv_index_t* jv_load(const char* path);
typedef struct { uint32_t id; float dist; } jv_hit_t;
int         jv_search(jv_index_t*, const float* query, uint32_t k, uint32_t efSearch, jv_hit_t* out);
```

---

## CLI UX

```bash
# Build
jetvec build --in vecs.f32 --dim 768 --metric cosine \
  --hnsw-M 16 --efc 200 --ivf-nlist 4096 --pq-m 32 \
  --out cilow.jvec

# Search
jetvec search --index cilow.jvec --q q.f32 --k 10 --efs 128

# Bench (microbench kernels)
jv_cli --bench 100000 768
```

---

## Test, Bench, and Acceptance

* **Correctness:** exact vs ANN recall@{1,10,100}, nDCG, tie handling.
* **Perf:** p50/p95/p99 latency with 1/4/8 threads; throughput QPS.
* **Memory:** index size, RSS, mmap residency; PQ compression ratio.
* **Stability:** fuzz random inputs, corrupt headers, partial writes.
* **Portability:** x86-64 Linux/macOS, ARM64 (Apple Silicon), Windows.

---

## Stretch goals (post v0)

* **OPQ** rotation for better PQ recall.
* **Vector dtypes**: float16, int8 (scale/zero-point).
* **Graph-aware re-ranking** hook (merge ANN with temporal/causal graph scores).
* **WASM** build + demo: browser-side RAG with 100k vectors in <50MB.
* **Streaming add + background compaction** (IVF list merges).
* **Disk-first mode**: mmap only codes + centroids for >10M scale.

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
./bin/jv_cli --bench 100000 768
ctest --output-on-failure
```

MIT License. PRs welcome.
EOF

# =========================
# CMake
# =========================
cat > "$ROOT_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(jetvec C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(JETVEC_BUILD_CLI "Build CLI" ON)
option(JETVEC_BUILD_TESTS "Build tests" ON)

# Output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Detect compiler
include(CheckCCompilerFlag)

# Sources
set(JETVEC_SRC
  src/api.c
  src/distance.c
  src/distance_scalar.c
  src/mmio.c
  src/util_crc64.c
)

# Try to add AVX2 object if supported (GCC/Clang)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  check_c_compiler_flag("-mavx2" HAS_MAVX2)
  if(HAS_MAVX2)
    add_library(distance_avx2 OBJECT src/distance_avx2.c)
    target_compile_options(distance_avx2 PRIVATE -mavx2)
    list(APPEND JETVEC_SRC $<TARGET_OBJECTS:distance_avx2>)
    target_compile_definitions(distance_avx2 PRIVATE JETVEC_COMPILED_WITH_AVX2=1)
  endif()
endif()

add_library(jetvec STATIC ${JETVEC_SRC})
target_include_directories(jetvec PUBLIC include)
if(MSVC)
  target_compile_options(jetvec PRIVATE /O2)
else()
  target_compile_options(jetvec PRIVATE -O3 -fno-math-errno -ffast-math)
endif()

if(JETVEC_BUILD_CLI)
  add_executable(jv_cli src/cli.c)
  target_link_libraries(jv_cli PRIVATE jetvec)
endif()

if(JETVEC_BUILD_TESTS)
  enable_testing()
  add_executable(test_distance tests/test_distance.c)
  target_link_libraries(test_distance PRIVATE jetvec)
  add_test(NAME distance COMMAND test_distance)
endif()
EOF

# =========================
# Public headers
# =========================
cat > "$ROOT_DIR/include/api.h" << 'EOF'
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
EOF

cat > "$ROOT_DIR/include/distance.h" << 'EOF'
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
EOF

cat > "$ROOT_DIR/include/mmio.h" << 'EOF'
#ifndef JV_MMIO_H
#define JV_MMIO_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
  void*   addr;
  size_t  size;
#ifdef _WIN32
  void*   hFile;
  void*   hMap;
#endif
} jv_mmap_t;

int  jv_mmap_read (const char* path, jv_mmap_t* out);
int  jv_mmap_rw   (const char* path, size_t size, jv_mmap_t* out); // create/resize then map
int  jv_munmap    (jv_mmap_t* m);
int  jv_msync     (jv_mmap_t* m);

#endif
EOF

cat > "$ROOT_DIR/include/util_crc64.h" << 'EOF'
#ifndef JV_UTIL_CRC64_H
#define JV_UTIL_CRC64_H
#include <stddef.h>
#include <stdint.h>
uint64_t jv_crc64(const void* data, size_t len);
#endif
EOF

# =========================
# Sources
# =========================
cat > "$ROOT_DIR/src/api.c" << 'EOF'
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
EOF

cat > "$ROOT_DIR/src/distance.c" << 'EOF'
#include "distance.h"

// Function pointers default to scalar; set in jv_runtime_init (api.c)
jv_dot_f32_fn  jv_dot_f32  = jv_dot_f32_scalar;
jv_l2sq_f32_fn jv_l2sq_f32 = jv_l2sq_f32_scalar;
EOF

cat > "$ROOT_DIR/src/distance_scalar.c" << 'EOF'
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
EOF

cat > "$ROOT_DIR/src/distance_avx2.c" << 'EOF'
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
EOF

cat > "$ROOT_DIR/src/mmio.c" << 'EOF'
#include "mmio.h"
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
int jv_mmap_read(const char* path, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hf == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return -1; }
  HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!hm) { CloseHandle(hf); return -1; }
  void* addr = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
  if (!addr) { CloseHandle(hm); CloseHandle(hf); return -1; }
  out->addr = addr; out->size = (size_t)sz.QuadPart; out->hFile = hf; out->hMap = hm; return 0;
}
int jv_mmap_rw(const char* path, size_t size, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  HANDLE hf = CreateFileA(path, GENERIC_READ|GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hf == INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; sz.QuadPart = (LONGLONG)size;
  if (!SetFilePointerEx(hf, sz, NULL, FILE_BEGIN) || !SetEndOfFile(hf)) { CloseHandle(hf); return -1; }
  HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READWRITE, 0, 0, NULL);
  if (!hm) { CloseHandle(hf); return -1; }
  void* addr = MapViewOfFile(hm, FILE_MAP_WRITE|FILE_MAP_READ, 0, 0, size);
  if (!addr) { CloseHandle(hm); CloseHandle(hf); return -1; }
  out->addr = addr; out->size = size; out->hFile = hf; out->hMap = hm; return 0;
}
int jv_munmap(jv_mmap_t* m) {
  if (!m || !m->addr) return 0;
  UnmapViewOfFile(m->addr);
  if (m->hMap) CloseHandle((HANDLE)m->hMap);
  if (m->hFile) CloseHandle((HANDLE)m->hFile);
  memset(m,0,sizeof(*m));
  return 0;
}
int jv_msync(jv_mmap_t* m) {
  if (!m || !m->addr) return -1;
  return FlushViewOfFile(m->addr, 0) ? 0 : -1;
}
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int jv_mmap_read(const char* path, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st; if (fstat(fd, &st) < 0) { close(fd); return -1; }
  void* addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) { close(fd); return -1; }
  out->addr = addr; out->size = (size_t)st.st_size;
  close(fd);
  return 0;
}
int jv_mmap_rw(const char* path, size_t size, jv_mmap_t* out) {
  memset(out, 0, sizeof(*out));
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  if (ftruncate(fd, (off_t)size) < 0) { close(fd); return -1; }
  void* addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) { close(fd); return -1; }
  out->addr = addr; out->size = size;
  close(fd);
  return 0;
}
int jv_munmap(jv_mmap_t* m) {
  if (!m || !m->addr) return 0;
  munmap(m->addr, m->size);
  memset(m,0,sizeof(*m));
  return 0;
}
int jv_msync(jv_mmap_t* m) {
  if (!m || !m->addr) return -1;
  return msync(m->addr, m->size, MS_SYNC);
}
#endif
EOF

cat > "$ROOT_DIR/src/util_crc64.c" << 'EOF'
#include "util_crc64.h"
#define POLY 0x42F0E1EBA9EA3693ULL
static uint64_t table[256];
static int      initd = 0;
static void init(void){
  if(initd) return;
  for (unsigned i=0; i<256; ++i){
    uint64_t crc = (uint64_t)i << 56;
    for (int j=0; j<8; ++j)
      crc = (crc & 0x8000000000000000ULL) ? (crc << 1) ^ POLY : (crc << 1);
    table[i]=crc;
  }
  initd=1;
}
uint64_t jv_crc64(const void* data, size_t len){
  init();
  const unsigned char* p = (const unsigned char*)data;
  uint64_t crc = 0ULL;
  for (size_t i=0;i<len;++i){
    unsigned idx = (unsigned)((crc >> 56) ^ p[i]) & 0xFFU;
    crc = table[idx] ^ (crc << 8);
  }
  return crc;
}
EOF

# =========================
# CLI (microbench)
# =========================
cat > "$ROOT_DIR/src/cli.c" << 'EOF'
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
EOF

# =========================
# Tests
# =========================
cat > "$ROOT_DIR/tests/test_distance.c" << 'EOF'
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
EOF

# =========================
# GitHub Actions (optional)
# =========================
cat > "$ROOT_DIR/.github/workflows/ci.yml" << 'EOF'
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Configure
      run: mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
    - name: Build
      run: cmake --build build -j
    - name: Test
      run: cd build && ctest --output-on-failure
EOF

echo "✅ Bootstrapped ${PROJECT} at ${ROOT_DIR}"
echo "Next:"
echo "  cd ${PROJECT} && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j"
echo "  ./bin/jv_cli --bench 100000 768"