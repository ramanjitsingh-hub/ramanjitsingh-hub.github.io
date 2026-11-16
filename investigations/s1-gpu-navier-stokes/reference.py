"""Independent NumPy reference for ns2d.cu.

Same MAC discretisation, same SSP-RK3, but the pressure Poisson equation is solved exactly
with an FFT using the eigenvalues of the discrete 5-point Laplacian. Agreement with the GPU
solver therefore checks the advection stencils, the time integrator and the multigrid solver
independently.
"""
from __future__ import annotations

import numpy as np

R = np.roll  # R(a, s, axis): a[i - s]


def taylor_green(n, nu, t):
    h = 2 * np.pi / n
    i = np.arange(n)
    X, Y = np.meshgrid(i * h, i * h)          # arrays indexed [j, i]
    dec = np.exp(-2 * nu * t)
    u = np.sin(X) * np.cos(Y + h / 2) * dec
    v = -np.cos(X + h / 2) * np.sin(Y) * dec
    return u, v


def rhs(u, v, h, nu):
    E = lambda a: R(a, -1, 1)   # a[i+1]
    W = lambda a: R(a, 1, 1)    # a[i-1]
    N = lambda a: R(a, -1, 0)   # a[j+1]
    S = lambda a: R(a, 1, 0)    # a[j-1]
    lap = lambda a: (E(a) + W(a) + N(a) + S(a) - 4 * a) / h**2
    duu = ((0.5 * (u + E(u)))**2 - (0.5 * (W(u) + u))**2) / h
    uvT = 0.5 * (u + N(u)) * 0.5 * (N(W(v)) + N(v))
    uvB = 0.5 * (S(u) + u) * 0.5 * (W(v) + v)
    ru = -(duu + (uvT - uvB) / h) + nu * lap(u)
    dvv = ((0.5 * (v + N(v)))**2 - (0.5 * (S(v) + v))**2) / h
    uvR = 0.5 * (S(E(u)) + E(u)) * 0.5 * (v + E(v))
    uvL = uvB
    rv = -((uvR - uvL) / h + dvv) + nu * lap(v)
    return ru, rv


def project(u, v, h):
    n = u.shape[0]
    div = (R(u, -1, 1) - u + R(v, -1, 0) - v) / h
    k = np.arange(n)
    lam = (2 * np.cos(2 * np.pi * k / n) - 2) / h**2
    L = lam[None, :] + lam[:, None]
    L[0, 0] = 1.0
    ph = np.fft.fft2(div) / L
    ph[0, 0] = 0.0
    phi = np.real(np.fft.ifft2(ph))
    u = u - (phi - R(phi, 1, 1)) / h
    v = v - (phi - R(phi, 1, 0)) / h
    return u, v


def step(u, v, h, nu, dt):
    ru, rv = rhs(u, v, h, nu)
    u1, v1 = project(u + dt * ru, v + dt * rv, h)
    ru, rv = rhs(u1, v1, h, nu)
    u2, v2 = project(0.75 * u + 0.25 * (u1 + dt * ru), 0.75 * v + 0.25 * (v1 + dt * rv), h)
    ru, rv = rhs(u2, v2, h, nu)
    return project(u / 3 + 2 / 3 * (u2 + dt * ru), v / 3 + 2 / 3 * (v2 + dt * rv), h)


def run(n, nu=0.01, T=1.0, cfl=0.5):
    h = 2 * np.pi / n
    dt = min(cfl * h, 0.2 * h * h / nu)
    steps = int(np.ceil(T / dt)); dt = T / steps
    u, v = project(*taylor_green(n, nu, 0.0), h)
    for _ in range(steps):
        u, v = step(u, v, h, nu, dt)
    return u, v, steps


if __name__ == "__main__":
    u, v, s = run(64)
    ue, ve = taylor_green(64, 0.01, 1.0)
    print("steps", s, "linf", max(abs(u - ue).max(), abs(v - ve).max()))
