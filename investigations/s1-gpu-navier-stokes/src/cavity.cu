// cavity.cu  Lid-driven cavity, 2D incompressible Navier-Stokes on the unit square, GPU, FP64.
//
// Same scheme as ns2d.cu (MAC grid, conservative central advection, SSP-RK3 with a projection
// after every stage, geometric multigrid), with walls:
//   u[i,j] at (i h, (j+1/2) h), i = 0..N, j = 0..N-1     u = 0 on x = 0 and x = 1
//   v[i,j] at ((i+1/2) h, j h), i = 0..N-1, j = 0..N     v = 0 on y = 0 and y = 1
//   no-slip tangential velocity through ghost values: u_ghost = 2 U_wall - u_inside,
//   lid at y = 1 moves with U = 1. Pressure correction satisfies homogeneous Neumann conditions,
//   so the multigrid uses a stencil that counts only interior neighbours.
//
// Usage: cavity --N 128 --Re 100 [--cfl 0.5] [--tss 1e-7] [--tmax 200] [--dump file]
// Runs to steady state (max |du/dt| < tss) and prints the centreline profiles as JSON.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static cudaStream_t S;
static dim3 B2(32, 8);
static dim3 G2(int nx, int ny) { return dim3((nx + 31) / 32, (ny + 7) / 8); }

struct Grid { int N; double h, ulid; };

__device__ __forceinline__ double Ug(const double* u, int i, int j, int N, double ulid) {
  if (j < 0) return -u[i];                                   // bottom wall, u = 0
  if (j >= N) return 2.0 * ulid - u[(N - 1) * (N + 1) + i];  // lid
  return u[j * (N + 1) + i];
}
__device__ __forceinline__ double Vg(const double* v, int i, int j, int N) {
  if (i < 0) return -v[j * N];                               // left wall, v = 0
  if (i >= N) return -v[j * N + N - 1];                      // right wall
  return v[j * N + i];
}

__global__ void k_rhs_u(const double* u, const double* v, double* ru, int N, double h, double nu, double ulid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > N || j >= N) return;
  int k = j * (N + 1) + i;
  if (i == 0 || i == N) { ru[k] = 0; return; }
  double uc = u[k], uE = u[k + 1], uW = u[k - 1], uN = Ug(u, i, j + 1, N, ulid), uS = Ug(u, i, j - 1, N, ulid);
  double a = 0.5 * (uc + uE), b = 0.5 * (uW + uc);
  double uT = 0.5 * (uc + uN), vT = 0.5 * (Vg(v, i - 1, j + 1, N) + Vg(v, i, j + 1, N));
  double uB = 0.5 * (uS + uc), vB = 0.5 * (Vg(v, i - 1, j, N) + Vg(v, i, j, N));
  ru[k] = -((a * a - b * b) + (uT * vT - uB * vB)) / h + nu * (uE + uW + uN + uS - 4.0 * uc) / (h * h);
}

__global__ void k_rhs_v(const double* u, const double* v, double* rv, int N, double h, double nu, double ulid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j > N) return;
  int k = j * N + i;
  if (j == 0 || j == N) { rv[k] = 0; return; }
  double vc = v[k], vN = v[k + N], vS = v[k - N], vE = Vg(v, i + 1, j, N), vW = Vg(v, i - 1, j, N);
  double c = 0.5 * (vc + vN), d = 0.5 * (vS + vc);
  double uR = 0.5 * (Ug(u, i + 1, j - 1, N, ulid) + Ug(u, i + 1, j, N, ulid)), vR = 0.5 * (vc + vE);
  double uL = 0.5 * (Ug(u, i, j - 1, N, ulid) + Ug(u, i, j, N, ulid)), vL = 0.5 * (vW + vc);
  rv[k] = -((uR * vR - uL * vL) + (c * c - d * d)) / h + nu * (vE + vW + vN + vS - 4.0 * vc) / (h * h);
}

__global__ void k_axpy(int m, double a, const double* x0, double b, const double* xk, double dt, const double* r, double* out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) out[k] = a * x0[k] + b * (xk[k] + dt * r[k]);
}

__global__ void k_div(const double* u, const double* v, double* f, int N, double h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N) return;
  f[j * N + i] = (u[j * (N + 1) + i + 1] - u[j * (N + 1) + i] + v[(j + 1) * N + i] - v[j * N + i]) / h;
}

__global__ void k_correct(double* u, double* v, const double* p, int N, double h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > N || j > N) return;
  if (j < N && i >= 1 && i <= N - 1) u[j * (N + 1) + i] -= (p[j * N + i] - p[j * N + i - 1]) / h;
  if (i < N && j >= 1 && j <= N - 1) v[j * N + i] -= (p[j * N + i] - p[(j - 1) * N + i]) / h;
}

// ---- Neumann multigrid: neighbours outside the domain are dropped from the stencil
__device__ __forceinline__ void nsum(const double* p, int i, int j, int n, double& s, int& c) {
  s = 0; c = 0;
  if (i > 0) { s += p[j * n + i - 1]; c++; }
  if (i < n - 1) { s += p[j * n + i + 1]; c++; }
  if (j > 0) { s += p[(j - 1) * n + i]; c++; }
  if (j < n - 1) { s += p[(j + 1) * n + i]; c++; }
}
__global__ void k_smooth(double* p, const double* f, int n, double h2, int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n || ((i + j) & 1) != color) return;
  double s; int c; nsum(p, i, j, n, s, c);
  p[j * n + i] = (s - h2 * f[j * n + i]) / c;
}
__global__ void k_resid(const double* p, const double* f, double* r, int n, double ih2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  double s; int c; nsum(p, i, j, n, s, c);
  r[j * n + i] = f[j * n + i] - (s - c * p[j * n + i]) * ih2;
}
__global__ void k_restrict(const double* r, double* fc, int nc) {
  int I = blockIdx.x * blockDim.x + threadIdx.x, J = blockIdx.y * blockDim.y + threadIdx.y;
  if (I >= nc || J >= nc) return;
  int nf = 2 * nc, i = 2 * I, j = 2 * J;
  fc[J * nc + I] = 0.25 * (r[j * nf + i] + r[j * nf + i + 1] + r[(j + 1) * nf + i] + r[(j + 1) * nf + i + 1]);
}
__global__ void k_prolong_add(const double* ec, double* p, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nf || j >= nf) return;
  int nc = nf / 2, I = i >> 1, J = j >> 1;
  int I2 = min(max(I + ((i & 1) ? 1 : -1), 0), nc - 1), J2 = min(max(J + ((j & 1) ? 1 : -1), 0), nc - 1);
  p[j * nf + i] += 0.5625 * ec[J * nc + I] + 0.1875 * (ec[J * nc + I2] + ec[J2 * nc + I]) + 0.0625 * ec[J2 * nc + I2];
}
__global__ void k_coarse(double* p, const double* f, int n, double h2, int sweeps) {
  __shared__ double sp[1024];
  int t = threadIdx.x, i = t % n, j = t / n; bool act = t < n * n;
  double ff = act ? f[t] : 0.0;
  if (act) sp[t] = 0.0;
  __syncthreads();
  for (int s = 0; s < sweeps; s++)
    for (int color = 0; color < 2; color++) {
      if (act && ((i + j) & 1) == color) { double ss; int c; nsum(sp, i, j, n, ss, c); sp[t] = (ss - h2 * ff) / c; }
      __syncthreads();
    }
  if (act) p[t] = sp[t];
}
__global__ void k_fill(double* x, int m, double val) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < m) x[k] = val; }
__global__ void k_shift(double* x, int m, double val) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < m) x[k] -= val; }
__global__ void k_absdiff(const double* a, const double* b, double* d, int m) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < m) d[k] = fabs(a[k] - b[k]); }

__global__ void k_reduce(const double* x, int m, double* psum, double* pmax) {
  __shared__ double ss[256], sm[256];
  double s = 0, mx = 0;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < m; k += gridDim.x * blockDim.x) { s += x[k]; mx = fmax(mx, fabs(x[k])); }
  ss[threadIdx.x] = s; sm[threadIdx.x] = mx; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) { ss[threadIdx.x] += ss[threadIdx.x + o]; sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + o]); }
    __syncthreads();
  }
  if (threadIdx.x == 0) { psum[blockIdx.x] = ss[0]; pmax[blockIdx.x] = sm[0]; }
}

static const int RB = 120;
static double *d_ps, *d_pm, *h_ps, *h_pm;
static void reduce(const double* x, int m, double& s, double& mx) {
  k_reduce<<<RB, 256, 0, S>>>(x, m, d_ps, d_pm);
  CK(cudaMemcpyAsync(h_ps, d_ps, RB * 8, cudaMemcpyDeviceToHost, S));
  CK(cudaMemcpyAsync(h_pm, d_pm, RB * 8, cudaMemcpyDeviceToHost, S));
  CK(cudaStreamSynchronize(S));
  s = 0; mx = 0; for (int b = 0; b < RB; b++) { s += h_ps[b]; mx = std::max(mx, h_pm[b]); }
}

struct MG {
  int L; std::vector<int> n; std::vector<double*> p, f, r; double h0; int last = 0; cudaGraphExec_t gx = nullptr;
  MG(int N, double h) : h0(h) {
    for (int m = N; m >= 4; m /= 2) n.push_back(m);
    L = (int)n.size() - 1;
    for (int l = 0; l <= L; l++) {
      size_t b = (size_t)n[l] * n[l] * 8; double *a, *c, *d;
      CK(cudaMalloc(&a, b)); CK(cudaMalloc(&c, b)); CK(cudaMalloc(&d, b)); CK(cudaMemset(a, 0, b));
      p.push_back(a); f.push_back(c); r.push_back(d);
    }
  }
  double h(int l) const { return h0 * (double)(1 << l); }
  void smooth(int l, int s) { for (int k = 0; k < s; k++) for (int c = 0; c < 2; c++) k_smooth<<<G2(n[l], n[l]), B2, 0, S>>>(p[l], f[l], n[l], h(l) * h(l), c); }
  void vcycle(int l) {
    if (l == L) { k_coarse<<<1, n[l] * n[l], 0, S>>>(p[l], f[l], n[l], h(l) * h(l), 80); return; }
    smooth(l, 2);
    k_resid<<<G2(n[l], n[l]), B2, 0, S>>>(p[l], f[l], r[l], n[l], 1.0 / (h(l) * h(l)));
    k_restrict<<<G2(n[l + 1], n[l + 1]), B2, 0, S>>>(r[l], f[l + 1], n[l + 1]);
    k_fill<<<(n[l + 1] * n[l + 1] + 255) / 256, 256, 0, S>>>(p[l + 1], n[l + 1] * n[l + 1], 0.0);
    vcycle(l + 1);
    k_prolong_add<<<G2(n[l], n[l]), B2, 0, S>>>(p[l + 1], p[l], n[l]);
    smooth(l, 2);
  }
  void cycle() {
    if (!gx) { cudaGraph_t g; CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal)); vcycle(0);
               CK(cudaStreamEndCapture(S, &g)); CK(cudaGraphInstantiate(&gx, g, 0)); }
    CK(cudaGraphLaunch(gx, S));
  }
  double resid_max() { k_resid<<<G2(n[0], n[0]), B2, 0, S>>>(p[0], f[0], r[0], n[0], 1.0 / (h0 * h0)); double s, m; reduce(r[0], n[0] * n[0], s, m); return m; }
  int solve(double rtol, int maxc) {
    int m = n[0] * n[0]; double s, fm; reduce(f[0], m, s, fm);
    k_shift<<<(m + 255) / 256, 256, 0, S>>>(f[0], m, s / m);
    if (fm == 0) return 0;
    int c0 = last > 0 ? std::max(1, last - 1) : 8, c = 0;
    for (; c < c0; c++) cycle();
    double rm = resid_max();
    while (c < maxc && rm > rtol * fm) { cycle(); c++; rm = resid_max(); }
    last = c; return c;
  }
};

int main(int argc, char** argv) {
  int N = 128; double Re = 100, cfl = 0.5, tss = 1e-7, tmax = 300, tol = 1e-10; std::string dump;
  for (int k = 1; k + 1 < argc; k += 2) {
    std::string a = argv[k], b = argv[k + 1];
    if (a == "--N") N = atoi(b.c_str()); else if (a == "--Re") Re = atof(b.c_str()); else if (a == "--cfl") cfl = atof(b.c_str());
    else if (a == "--tss") tss = atof(b.c_str()); else if (a == "--tmax") tmax = atof(b.c_str()); else if (a == "--dump") dump = b;
    else if (a == "--tol") tol = atof(b.c_str());
  }
  CK(cudaStreamCreate(&S));
  CK(cudaMalloc(&d_ps, RB * 8)); CK(cudaMalloc(&d_pm, RB * 8)); CK(cudaMallocHost(&h_ps, RB * 8)); CK(cudaMallocHost(&h_pm, RB * 8));
  double h = 1.0 / N, nu = 1.0 / Re, ulid = 1.0;
  int mu = (N + 1) * N, mv = N * (N + 1);
  double *u, *v, *u1, *v1, *ru, *rv, *uo, *dd;
  for (double** q : {&u, &u1, &ru, &uo}) { CK(cudaMalloc(q, mu * 8)); CK(cudaMemset(*q, 0, mu * 8)); }
  for (double** q : {&v, &v1, &rv}) { CK(cudaMalloc(q, mv * 8)); CK(cudaMemset(*q, 0, mv * 8)); }
  CK(cudaMalloc(&dd, mu * 8));
  MG mg(N, h);
  long long cycles = 0, solves = 0;
  auto project = [&](double* uu, double* vv) {
    k_div<<<G2(N, N), B2, 0, S>>>(uu, vv, mg.f[0], N, h);
    cycles += mg.solve(tol, 40); solves++;
    k_correct<<<G2(N + 1, N + 1), B2, 0, S>>>(uu, vv, mg.p[0], N, h);
  };
  auto rhs = [&](const double* uu, const double* vv) {
    k_rhs_u<<<G2(N + 1, N), B2, 0, S>>>(uu, vv, ru, N, h, nu, ulid);
    k_rhs_v<<<G2(N, N + 1), B2, 0, S>>>(uu, vv, rv, N, h, nu, ulid);
  };
  auto ax = [&](int m, double a, const double* x0, double b, const double* xk, double dt, const double* r, double* out) {
    k_axpy<<<(m + 255) / 256, 256, 0, S>>>(m, a, x0, b, xk, dt, r, out);
  };
  double dt = std::min(cfl * h / ulid, 0.2 * h * h / nu);
  int check = std::max(100, (int)(0.5 / dt));    // check for steady state every half time unit
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, S);
  double t = 0, rate = 1; long long steps = 0;
  std::vector<std::pair<double, double>> hist;
  while (t < tmax) {
    CK(cudaMemcpyAsync(uo, u, mu * 8, cudaMemcpyDeviceToDevice, S));
    for (int s = 0; s < check; s++) {
      rhs(u, v);   ax(mu, 0, u, 1, u, dt, ru, u1); ax(mv, 0, v, 1, v, dt, rv, v1); project(u1, v1);
      rhs(u1, v1); ax(mu, 0.75, u, 0.25, u1, dt, ru, u1); ax(mv, 0.75, v, 0.25, v1, dt, rv, v1); project(u1, v1);
      rhs(u1, v1); ax(mu, 1.0 / 3, u, 2.0 / 3, u1, dt, ru, u); ax(mv, 1.0 / 3, v, 2.0 / 3, v1, dt, rv, v); project(u, v);
      t += dt; steps++;
    }
    k_absdiff<<<(mu + 255) / 256, 256, 0, S>>>(u, uo, dd, mu);
    double sdum; reduce(dd, mu, sdum, rate); rate /= (check * dt);
    hist.push_back({t, rate});
    if (!(rate == rate) || rate > 1e6) { rate = NAN; break; }   // blow-up guard
    if (rate < tss) break;
  }
  cudaEventRecord(e1, S); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
  std::vector<double> hu(mu), hv(mv);
  CK(cudaMemcpy(hu.data(), u, mu * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hv.data(), v, mv * 8, cudaMemcpyDeviceToHost));
  k_div<<<G2(N, N), B2, 0, S>>>(u, v, mg.r[0], N, h); double sd, maxdiv; reduce(mg.r[0], N * N, sd, maxdiv);
  if (!dump.empty()) { FILE* fp = fopen(dump.c_str(), "wb"); fwrite(hu.data(), 8, mu, fp); fwrite(hv.data(), 8, mv, fp); fclose(fp); }
  printf("{\"N\":%d,\"Re\":%g,\"dt\":%.6e,\"t\":%.4f,\"steps\":%lld,\"steady_rate\":%.3e,\"converged\":%s,\"max_div\":%.3e,"
         "\"mg_cycles_per_solve\":%.3f,\"wall_s\":%.2f,\"u_x05\":[", N, Re, dt, t, steps, rate, (rate == rate && rate < tss) ? "true" : "false",
         maxdiv, (double)cycles / solves, ms / 1000);
  for (int j = 0; j < N; j++) printf("%s%.8f", j ? "," : "", hu[j * (N + 1) + N / 2]);
  printf("],\"v_y05\":[");
  for (int i = 0; i < N; i++) printf("%s%.8f", i ? "," : "", hv[(N / 2) * N + i]);
  printf("],\"rate_history\":[");
  for (size_t k = 0; k < hist.size(); k++) printf("%s[%.3f,%.3e]", k ? "," : "", hist[k].first, hist[k].second);
  printf("]}\n");
  return 0;
}
