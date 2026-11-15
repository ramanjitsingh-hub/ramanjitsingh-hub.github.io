// Investigation 03: the anatomy of a graphics card.
// Small kernels that each isolate one hardware effect, so a simple model can be tested against it.
//
//   anat info
//   anat chase  --bytes B [--stride 128]          pointer-chase latency, one thread
//   anat bw     --warps W --ilp K                 streaming read bandwidth at controlled occupancy
//   anat stride --s S                             strided 4-byte reads, effective bandwidth
//   anat ai     --prec f32|f64 --flops F          F FMAs per loaded element (roofline sweep)
//   anat div    --active K --paths P              FP64 work under warp divergence
//   anat bank   --s S                             shared-memory reads with word stride S
//   anat bankw  --width 4|8|16 --s S              shared-memory reads of wider elements
// milestone 2:
//   anat fp64   --warps W --chains 1|8            FP64 FMA issue rate at W warps per SM
//   anat bw     ... --array B --passes P --pipe 1 software-pipelined reads, any array size
//   anat chase  ... --jitter 1                    one line per stride block at a random offset (TLB probe)
//   anat stride ... --span B --halfoff 32         fixed span; odd warps shifted by 32 elements
//   anat chase|bw ... --cg 1                      loads bypass L1 (ld.global.cg), so the L2 is measured
//
// Every mode prints one JSON line. Times are medians of --reps launches (default 5).
// SM clock during a run is measured from clock64() on block 0 over the launch's event time.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int SMS = 0;
struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  void start() { CK(cudaEventRecord(a)); }
  float stop() { CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }

// ---------------------------------------------------------------- clock
// SM clock during a kernel: clock64() cycles over %globaltimer nanoseconds, both read by thread 0 of
// block 0 at its start and end. Independent of how many blocks are resident at once.
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define T_BEGIN long long c0_ = clock64(); unsigned long long g0_ = gtimer();
#define T_END(cyc) if (blockIdx.x == 0 && threadIdx.x == 0) { cyc[0] = clock64() - c0_; cyc[1] = (long long)(gtimer() - g0_); }
__global__ void k_spin(long long cycles, long long* out) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
  if (threadIdx.x == 0 && blockIdx.x == 0) *out = clock64() - t0;
}
static double spin_clock_ghz(long long* d) {
  Timer t; k_spin<<<1, 32>>>(10000000, d); CK(cudaDeviceSynchronize());   // warm
  t.start(); k_spin<<<1, 32>>>(400000000, d); float ms = t.stop();
  long long c; CK(cudaMemcpy(&c, d, 8, cudaMemcpyDeviceToHost));
  return c / (ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------- 1. pointer chase
// next[] holds a random cyclic permutation over cache lines (Sattolo), so every load depends on the
// previous one and no two consecutive loads are neighbours in memory.
__global__ void k_scatter(unsigned* next, const unsigned* perm, const unsigned* addr, size_t lines) {
  size_t l = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (l < lines) next[addr[l]] = addr[perm[l]];
}
// CG = true loads with ld.global.cg, which caches in L2 only, so L1 cannot hold the chain.
template <bool CG>
__device__ __forceinline__ unsigned ldn(const unsigned* p) { return CG ? __ldcg(p) : *p; }
template <bool CG>
__global__ void k_chase(unsigned* next, unsigned start, int warm, int iters, long long* cyc, unsigned* sink) {
  unsigned j = start;
  for (int i = 0; i < warm; i++) j = ldn<CG>(next + j);
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) j = ldn<CG>(next + j);
  long long t1 = clock64();
  cyc[0] = t1 - t0; sink[0] = j;
}

// ---------------------------------------------------------------- 2. streaming read at controlled occupancy
template <bool CG>
__device__ __forceinline__ float4 ld4(const float4* p) { return CG ? __ldcg(p) : *p; }
template <int K, bool CG>
__global__ void k_read(const float4* __restrict__ a, size_t n, int passes, float* out, long long* cyc) {
  T_BEGIN
  size_t gs = (size_t)gridDim.x * blockDim.x, i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (int pass = 0; pass < passes; pass++)
  for (size_t i = i0; i + (K - 1) * gs < n; i += K * gs) {
    float4 v[K];
#pragma unroll
    for (int k = 0; k < K; k++) v[k] = ld4<CG>(a + i + k * gs);      // K independent 16-byte loads in flight per thread
#pragma unroll
    for (int k = 0; k < K; k++) acc += v[k].x + v[k].y + v[k].z + v[k].w;
  }
  if (acc == -1.f) out[0] = acc;
  T_END(cyc)
}
// Software-pipelined: the next round of K loads is issued before the current round is summed,
// so the adds and loop overhead overlap with memory latency instead of adding to it.
template <int K, bool CG>
__global__ void k_readp(const float4* __restrict__ a, size_t n, int passes, float* out, long long* cyc) {
  T_BEGIN
  size_t gs = (size_t)gridDim.x * blockDim.x, i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (int pass = 0; pass < passes; pass++) {
    size_t i = i0;
    if (i + (K - 1) * gs >= n) continue;
    float4 cur[K], nxt[K];
#pragma unroll
    for (int k = 0; k < K; k++) cur[k] = ld4<CG>(a + i + k * gs);
    while (true) {
      size_t j = i + K * gs; bool more = j + (K - 1) * gs < n;
      if (more) {
#pragma unroll
        for (int k = 0; k < K; k++) nxt[k] = ld4<CG>(a + j + k * gs);
      }
#pragma unroll
      for (int k = 0; k < K; k++) acc += cur[k].x + cur[k].y + cur[k].z + cur[k].w;
      if (!more) break;
#pragma unroll
      for (int k = 0; k < K; k++) cur[k] = nxt[k];
      i = j;
    }
  }
  if (acc == -1.f) out[0] = acc;
  T_END(cyc)
}

// ---------------------------------------------------------------- 3. strided reads
__global__ void k_stride(const float* __restrict__ a, size_t m, int S, int off, float* out) {
  size_t gs = (size_t)gridDim.x * blockDim.x, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (; i < m; i += gs) acc += a[i * (size_t)S + (((i >> 5) & 1) ? off : 0)];   // off shifts every other warp
  if (acc == -1.f) out[0] = acc;
}

// ---------------------------------------------------------------- 4. arithmetic intensity sweep
// Each thread loads one 16-byte vector and applies F FMAs to each component, spread over 8
// independent accumulators. FLOP per DRAM byte = 2F / sizeof(T).
template <class T, class V, int F>
__global__ void k_ai(const V* __restrict__ a, size_t n, T* out, long long* cyc) {
  T_BEGIN
  constexpr int NC = sizeof(V) / sizeof(T);
  T acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++) acc[j] = T(j);
  size_t gs = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gs) {
    V v = a[i];
    const T* c = reinterpret_cast<const T*>(&v);
#pragma unroll
    for (int q = 0; q < NC; q++)
#pragma unroll
      for (int f = 0; f < F; f++) { int j = (q * F + f) & 7; acc[j] = fma(acc[j], c[q], T(1.0001)); }
  }
  T s = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) s += acc[j];
  if (s == T(-1)) out[0] = s;
  T_END(cyc)
}

// ---------------------------------------------------------------- 5. divergence
// Path p runs chain<p>, a separate non-inlined function, so the compiler cannot merge the paths.
template <int C>
__device__ __noinline__ double chain(double y, double x, int it) {
  for (int i = 0; i < it; i++) y = fma(y, x, 1e-9 * (C + 1));
  return y;
}
#define CASE(p) case p: y = chain<p>(y, x, it); break;
__global__ void k_div(int it, int active, int paths, double* out, long long* cyc) {
  T_BEGIN
  int lane = threadIdx.x & 31;
  double x = 0.999 + 1e-6 * lane, y = 1.0;
  if (lane < active) {
    switch (lane % paths) {
      CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15)
      CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31)
    }
  }
  if (y == -1.0) out[0] = y;
  T_END(cyc)
}

// ---------------------------------------------------------------- 5b. FP64 issue rate (milestone 2)
// One block of 32W threads per SM. C = 1: one dependent chain per thread. C = 8: eight independent
// chains, so a single warp can have several FMAs ready at once.
template <int C>
__global__ void k_fp64(int it, double* out, long long* cyc) {
  T_BEGIN
  double x = 0.999 + 1e-6 * threadIdx.x, a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = 1.0 + j;
  for (int i = 0; i < it; i += 8) {
#pragma unroll
    for (int k = 0; k < 8; k++) { const int j = C == 1 ? 0 : k; a[j] = fma(a[j], x, 1e-9); }
  }
  double s = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) s += a[j];
  if (s == -1.0) out[0] = s;
  T_END(cyc)
}

// ---------------------------------------------------------------- 6. shared-memory banks
// Lane l reads word l*S (plus multiples of 256, which keep the bank). 32 banks of 4 bytes, so the
// predicted conflict degree is gcd(S, 32), and 1 for S = 0 (broadcast).
__global__ void k_bank(int S, int iters, float* out, long long* cyc) {
  __shared__ float sm[2048];
  for (int t = threadIdx.x; t < 2048; t += blockDim.x) sm[t] = (float)t;
  __syncthreads();
  T_BEGIN
  int lane = threadIdx.x & 31, base = (lane * S) & 2047;
  volatile float* v = sm;
  float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < 8; k++) a[k] += v[(base + 256 * k) & 2047];   // 256 is a multiple of 32: same bank
  }
  float s = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) s += a[k];
  if (s == -1.f) out[0] = s;
  T_END(cyc)
}

// Same test with wider elements, to separate bank conflicts from instruction issue. T is float,
// float2 or float4; lane l reads element l*S. Loads are explicit ld.volatile.shared, because with a
// plain C++ load the compiler hoists the few distinct addresses out of the loop (an earlier version
// of this kernel measured an impossible 178 bytes per cycle that way).
__device__ __forceinline__ unsigned saddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float lds(const float* p) { float r; asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=f"(r) : "r"(saddr(p))); return r; }
__device__ __forceinline__ float lds(const float2* p) { float a, b; asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];" : "=f"(a), "=f"(b) : "r"(saddr(p))); return a + b; }
__device__ __forceinline__ float lds(const float4* p) { float a, b, c, d; asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "r"(saddr(p))); return (a + b) + (c + d); }
template <class T>
__global__ void k_bankw(int S, int iters, float* out, long long* cyc) {
  constexpr int NE = 8192 / sizeof(T);
  __shared__ T sm[NE];
  float* f = reinterpret_cast<float*>(sm);
  for (int t = threadIdx.x; t < 2048; t += blockDim.x) f[t] = (float)t;
  __syncthreads();
  T_BEGIN
  int lane = threadIdx.x & 31, base = (lane * S) & (NE - 1);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < 8; k++) acc[k] += lds(&sm[(base + k * (NE / 8)) & (NE - 1)]);   // offsets of 1 KB keep every bank
  }
  float s = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) s += acc[k];
  if (s == -1.f) out[0] = s;
  T_END(cyc)
}

// ---------------------------------------------------------------- host
struct Args {
  std::string mode; long long bytes = 1 << 20; int stride = 128, warps = 48, ilp = 1, s = 1, flops = 1, active = 32, paths = 1, reps = 5;
  std::string prec = "f32"; int width = 4;
  long long array = 256ll << 20, span = 0; int passes = 1, pipe = 0, jitter = 0, halfoff = 0, chains = 1, cg = 0;
};
static Args parse(int argc, char** argv) {
  Args o; if (argc > 1) o.mode = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i], b = argv[i + 1];
    if (a == "--bytes") o.bytes = atoll(b.c_str()); else if (a == "--stride") o.stride = atoi(b.c_str());
    else if (a == "--warps") o.warps = atoi(b.c_str()); else if (a == "--ilp") o.ilp = atoi(b.c_str());
    else if (a == "--s") o.s = atoi(b.c_str()); else if (a == "--flops") o.flops = atoi(b.c_str());
    else if (a == "--active") o.active = atoi(b.c_str()); else if (a == "--paths") o.paths = atoi(b.c_str());
    else if (a == "--width") o.width = atoi(b.c_str());
    else if (a == "--array") o.array = atoll(b.c_str()); else if (a == "--span") o.span = atoll(b.c_str());
    else if (a == "--passes") o.passes = atoi(b.c_str()); else if (a == "--pipe") o.pipe = atoi(b.c_str());
    else if (a == "--jitter") o.jitter = atoi(b.c_str()); else if (a == "--halfoff") o.halfoff = atoi(b.c_str());
    else if (a == "--chains") o.chains = atoi(b.c_str()); else if (a == "--cg") o.cg = atoi(b.c_str());
    else if (a == "--prec") o.prec = b; else if (a == "--reps") o.reps = atoi(b.c_str());
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); exit(2); }
  }
  return o;
}

static int g_cg = 0;
template <int K>
static void launch_read(int grid, int block, const float4* a, size_t n, int passes, int pipe, float* out, long long* cyc) {
  if (g_cg) { if (pipe) k_readp<K, true><<<grid, block>>>(a, n, passes, out, cyc); else k_read<K, true><<<grid, block>>>(a, n, passes, out, cyc); }
  else { if (pipe) k_readp<K, false><<<grid, block>>>(a, n, passes, out, cyc); else k_read<K, false><<<grid, block>>>(a, n, passes, out, cyc); }
}

template <class T, class V, int F>
static void launch_ai(int grid, int block, const void* a, size_t n, void* out, long long* cyc) { k_ai<T, V, F><<<grid, block>>>((const V*)a, n, (T*)out, cyc); }
typedef void (*AiFn)(int, int, const void*, size_t, void*, long long*);
template <class T, class V>
static AiFn pick_ai(int F) {
  switch (F) {
    case 1: return launch_ai<T, V, 1>; case 2: return launch_ai<T, V, 2>; case 4: return launch_ai<T, V, 4>; case 8: return launch_ai<T, V, 8>;
    case 16: return launch_ai<T, V, 16>; case 32: return launch_ai<T, V, 32>; case 64: return launch_ai<T, V, 64>; case 128: return launch_ai<T, V, 128>;
    case 256: return launch_ai<T, V, 256>; case 512: return launch_ai<T, V, 512>;
  }
  fprintf(stderr, "flops must be a power of two from 1 to 512\n"); exit(2);
}

int main(int argc, char** argv) {
  Args o = parse(argc, argv); g_cg = o.cg;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); SMS = p.multiProcessorCount;
  long long* d_cyc; CK(cudaMalloc(&d_cyc, 64));
  float* d_out; CK(cudaMalloc(&d_out, 64));
  Timer tm;
  auto clock_of = [&](double) { long long c[2]; CK(cudaMemcpy(c, d_cyc, 16, cudaMemcpyDeviceToHost)); return c[1] > 0 ? (double)c[0] / c[1] : 0.0; };

  if (o.mode == "info") {
    int memclk = 0, bus = 0; CK(cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0)); CK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0));
    printf("{\"mode\":\"info\",\"name\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\",\"l2_bytes\":%d,\"smem_per_sm\":%zu,\"max_threads_per_sm\":%d,"
           "\"mem_clock_khz\":%d,\"bus_bits\":%d,\"spin_clock_GHz\":%.4f}\n",
           p.name, SMS, p.major, p.minor, p.l2CacheSize, p.sharedMemPerMultiprocessor, p.maxThreadsPerMultiProcessor, memclk, bus, spin_clock_ghz(d_cyc));
    return 0;
  }

  if (o.mode == "chase") {
    size_t n = (size_t)o.bytes / 4, E = o.stride / 4, lines = n / E;
    std::vector<unsigned> perm(lines), addr(lines);
    for (size_t l = 0; l < lines; l++) perm[l] = (unsigned)l;
    std::mt19937_64 g(42);
    for (size_t l = lines - 1; l > 0; l--) { std::uniform_int_distribution<size_t> u(0, l - 1); std::swap(perm[l], perm[u(g)]); }   // Sattolo: one cycle
    // node l sits at the start of stride block l, or with --jitter at a random 128-byte line inside it,
    // so that large power-of-two strides do not all map to the same L2 set
    std::mt19937_64 gj(7);
    size_t lpb = std::max<size_t>(1, (size_t)o.stride / 128);
    for (size_t l = 0; l < lines; l++) addr[l] = (unsigned)(l * E + (o.jitter ? (gj() % lpb) * 32 : 0));
    unsigned *d, *sink, *dp, *da; CK(cudaMalloc(&d, n * 4)); CK(cudaMalloc(&sink, 4)); CK(cudaMemset(d, 0, n * 4));
    CK(cudaMalloc(&dp, lines * 4)); CK(cudaMalloc(&da, lines * 4));
    CK(cudaMemcpy(dp, perm.data(), lines * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(da, addr.data(), lines * 4, cudaMemcpyHostToDevice));
    k_scatter<<<(unsigned)((lines + 255) / 256), 256>>>(d, dp, da, lines); CK(cudaDeviceSynchronize());
    int iters = 1 << 15, warm = (int)std::min<size_t>(lines * 2, 1 << 20);
    std::vector<double> cyc;
    double clk = spin_clock_ghz(d_cyc);
    for (int r = 0; r < o.reps; r++) {
      if (o.cg) k_chase<true><<<1, 1>>>(d, addr[0], warm, iters, d_cyc, sink); else k_chase<false><<<1, 1>>>(d, addr[0], warm, iters, d_cyc, sink);
      CK(cudaDeviceSynchronize());
      long long c; CK(cudaMemcpy(&c, d_cyc, 8, cudaMemcpyDeviceToHost)); cyc.push_back((double)c / iters);
    }
    double m = median(cyc);
    printf("{\"mode\":\"chase\",\"bytes\":%lld,\"stride\":%d,\"jitter\":%d,\"cg\":%d,\"lines\":%zu,\"cycles\":%.2f,\"clock_GHz\":%.4f,\"ns\":%.2f}\n", o.bytes, o.stride, o.jitter, o.cg, lines, m, clk, m / clk);
    return 0;
  }

  if (o.mode == "bw") {
    size_t bytes = (size_t)o.array, n = bytes / 16;
    float4* a; CK(cudaMalloc(&a, bytes)); CK(cudaMemset(a, 0, bytes));
    int block, grid;   // W warps per SM: one block of 32W threads per SM, or two blocks above 32 warps
    if (o.warps <= 32) { block = 32 * o.warps; grid = SMS; } else { block = 16 * o.warps; grid = 2 * SMS; }
    auto run = [&]() {
      switch (o.ilp) { case 1: launch_read<1>(grid, block, a, n, o.passes, o.pipe, d_out, d_cyc); break; case 2: launch_read<2>(grid, block, a, n, o.passes, o.pipe, d_out, d_cyc); break;
        case 4: launch_read<4>(grid, block, a, n, o.passes, o.pipe, d_out, d_cyc); break; case 8: launch_read<8>(grid, block, a, n, o.passes, o.pipe, d_out, d_cyc); break;
        case 16: launch_read<16>(grid, block, a, n, o.passes, o.pipe, d_out, d_cyc); break;
        default: fprintf(stderr, "ilp must be 1, 2, 4 or 8\n"); exit(2); }
    };
    run(); CK(cudaDeviceSynchronize());
    std::vector<double> t, clk;
    for (int r = 0; r < o.reps; r++) { tm.start(); run(); double ms = tm.stop(); t.push_back(ms); clk.push_back(clock_of(ms)); }
    double ms = median(t);
    printf("{\"mode\":\"bw\",\"warps_per_sm\":%d,\"ilp\":%d,\"array\":%zu,\"passes\":%d,\"pipe\":%d,\"cg\":%d,\"block\":%d,\"grid\":%d,\"ms\":%.4f,\"GBps\":%.2f,\"clock_GHz\":%.4f}\n",
           o.warps, o.ilp, bytes, o.passes, o.pipe, o.cg, block, grid, ms, (double)bytes * o.passes / (ms * 1e-3) / 1e9, median(clk));
    return 0;
  }

  if (o.mode == "stride") {
    size_t m = o.span > 0 ? (size_t)o.span / (4 * (size_t)o.s) : (1u << 21), bytes = m * (size_t)o.s * 4 + 4 * (size_t)o.halfoff + 4;
    float* a; CK(cudaMalloc(&a, bytes)); CK(cudaMemset(a, 0, bytes));
    int block = 256, grid = SMS * 6;
    k_stride<<<grid, block>>>(a, m, o.s, o.halfoff, d_out); CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < o.reps; r++) { tm.start(); k_stride<<<grid, block>>>(a, m, o.s, o.halfoff, d_out); t.push_back(tm.stop()); }
    double ms = median(t);
    printf("{\"mode\":\"stride\",\"s\":%d,\"span\":%lld,\"halfoff\":%d,\"accesses\":%zu,\"ms\":%.4f,\"useful_GBps\":%.2f}\n", o.s, o.span, o.halfoff, m, ms, m * 4.0 / (ms * 1e-3) / 1e9);
    return 0;
  }

  if (o.mode == "ai") {
    bool f64 = o.prec == "f64";
    size_t bytes = 64ull << 20, n = bytes / 16;
    void* a; CK(cudaMalloc(&a, bytes)); CK(cudaMemset(a, 0, bytes));
    AiFn fn = f64 ? pick_ai<double, double2>(o.flops) : pick_ai<float, float4>(o.flops);
    int block = 256, grid = SMS * 6;
    fn(grid, block, a, n, d_out, d_cyc); CK(cudaDeviceSynchronize());
    std::vector<double> t, clk;
    for (int r = 0; r < o.reps; r++) { tm.start(); fn(grid, block, a, n, d_out, d_cyc); double ms = tm.stop(); t.push_back(ms); clk.push_back(clock_of(ms)); }
    double ms = median(t), flop = 2.0 * o.flops * (bytes / (f64 ? 8 : 4));
    printf("{\"mode\":\"ai\",\"prec\":\"%s\",\"flops_per_elem\":%d,\"intensity\":%.4f,\"ms\":%.4f,\"GFLOPs\":%.3f,\"GBps\":%.2f,\"clock_GHz\":%.4f}\n",
           o.prec.c_str(), o.flops, flop / bytes, ms, flop / (ms * 1e-3) / 1e9, bytes / (ms * 1e-3) / 1e9, median(clk));
    return 0;
  }

  if (o.mode == "div") {
    if (o.paths < 1 || o.paths > 32 || o.active < 0 || o.active > 32) { fprintf(stderr, "paths in 1..32, active in 0..32\n"); exit(2); }
    int block = 256, grid = SMS * 6, it = 4096;
    double* out; CK(cudaMalloc(&out, 64));
    k_div<<<grid, block>>>(it, o.active, o.paths, out, d_cyc); CK(cudaDeviceSynchronize());
    std::vector<double> t, clk;
    for (int r = 0; r < o.reps; r++) { tm.start(); k_div<<<grid, block>>>(it, o.active, o.paths, out, d_cyc); double ms = tm.stop(); t.push_back(ms); clk.push_back(clock_of(ms)); }
    double ms = median(t), warp_fma = (double)grid * block / 32 * it;     // FMA instructions per path, per warp
    printf("{\"mode\":\"div\",\"active\":%d,\"paths\":%d,\"iters\":%d,\"ms\":%.4f,\"clock_GHz\":%.4f,\"cycles_per_warp_fma_per_sm\":%.3f}\n",
           o.active, o.paths, it, ms, median(clk), ms * 1e-3 * median(clk) * 1e9 * SMS / warp_fma);
    return 0;
  }

  if (o.mode == "bank") {
    int block = 256, grid = SMS * 6, iters = 1 << 14;
    k_bank<<<grid, block>>>(o.s, iters, d_out, d_cyc); CK(cudaDeviceSynchronize());
    std::vector<double> t, clk;
    for (int r = 0; r < o.reps; r++) { tm.start(); k_bank<<<grid, block>>>(o.s, iters, d_out, d_cyc); double ms = tm.stop(); t.push_back(ms); clk.push_back(clock_of(ms)); }
    double ms = median(t), warp_loads = (double)grid * block / 32 * iters * 8;
    printf("{\"mode\":\"bank\",\"s\":%d,\"ms\":%.4f,\"clock_GHz\":%.4f,\"cycles_per_warp_load_per_sm\":%.3f}\n",
           o.s, ms, median(clk), ms * 1e-3 * median(clk) * 1e9 * SMS / warp_loads);
    return 0;
  }

  if (o.mode == "fp64") {
    if (o.warps < 1 || o.warps > 32 || (o.chains != 1 && o.chains != 8)) { fprintf(stderr, "warps in 1..32, chains 1 or 8\n"); exit(2); }
    int it = 1 << 16;
    double* out; CK(cudaMalloc(&out, 64));
    auto run = [&]() { if (o.chains == 1) k_fp64<1><<<SMS, 32 * o.warps>>>(it, out, d_cyc); else k_fp64<8><<<SMS, 32 * o.warps>>>(it, out, d_cyc); };
    run(); CK(cudaDeviceSynchronize());
    std::vector<double> cpi, clk;
    for (int r = 0; r < o.reps; r++) {
      run(); CK(cudaDeviceSynchronize());
      long long c[2]; CK(cudaMemcpy(c, d_cyc, 16, cudaMemcpyDeviceToHost));
      cpi.push_back((double)c[0] / ((double)o.warps * it)); clk.push_back((double)c[0] / c[1]);
    }
    double m = median(cpi);
    printf("{\"mode\":\"fp64\",\"warps_per_sm\":%d,\"chains\":%d,\"cycles_per_warp_fma_per_sm\":%.3f,\"cycles_per_fma_per_warp\":%.3f,\"clock_GHz\":%.4f}\n",
           o.warps, o.chains, m, m * o.warps, median(clk));
    return 0;
  }

  if (o.mode == "bankw") {
    int block = 256, grid = SMS * 6, iters = 1 << 14;
    auto run = [&]() {
      if (o.width == 4) k_bankw<float><<<grid, block>>>(o.s, iters, d_out, d_cyc);
      else if (o.width == 8) k_bankw<float2><<<grid, block>>>(o.s, iters, d_out, d_cyc);
      else if (o.width == 16) k_bankw<float4><<<grid, block>>>(o.s, iters, d_out, d_cyc);
      else { fprintf(stderr, "width must be 4, 8 or 16\n"); exit(2); }
    };
    run(); CK(cudaDeviceSynchronize());
    std::vector<double> t, clk;
    for (int r = 0; r < o.reps; r++) { tm.start(); run(); double ms = tm.stop(); t.push_back(ms); clk.push_back(clock_of(ms)); }
    double ms = median(t), warp_loads = (double)grid * block / 32 * iters * 8, cpl = ms * 1e-3 * median(clk) * 1e9 * SMS / warp_loads;
    printf("{\"mode\":\"bankw\",\"width\":%d,\"s\":%d,\"ms\":%.4f,\"clock_GHz\":%.4f,\"cycles_per_warp_load_per_sm\":%.3f,\"bytes_per_cycle_per_sm\":%.1f}\n",
           o.width, o.s, ms, median(clk), cpl, 32.0 * o.width / cpl);
    return 0;
  }

  fprintf(stderr, "modes: info chase bw stride ai div bank bankw fp64\n");
  return 2;
}
