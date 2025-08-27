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
