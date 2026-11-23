"""Stacked kernel with x-dependent layer-0 scatter (build_stacked_kernel, xdep_scatter_layer0).

Kernel-level test with a synthetic two-layer vector observable; no survey files, no cosmology.

  1. constant sigma(lnM): the direct-sum branch reproduces the stationary (FFT) branch;
  2. sigma varying with lnM: the direct-sum branch reproduces an independent reference,
     E[g_k] = int dlnM p(lnM) int dy N(y - x1(lnM); sigma(lnM)) g_k(y), the inner integral by
     Gauss-Hermite quadrature (no grid in y), and the same for the second moment;
  3. the result depends on the slope of sigma(lnM) (the table is used);
  4. diag(full covariance) == variance in the direct-sum branch;
  5. jit(vmap) over clusters works and equals the per-cluster calls.

Run: python tests/test_stacked_xdep_scatter.py
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy import integrate

import cosmocnc_jax
from cosmocnc_jax.cnc import build_stacked_kernel

print("cosmocnc_jax from", cosmocnc_jax.__file__)

N_PTS = 4096
A0, B0 = 0.3, 1.1                      # layer 0: x1 = A0 + B0 * lnM
R = np.array([0.5, 1.0, 2.0, 4.0])     # "radii" of the vector observable


def layer0_fn(x0, a, b):
    return a + b * x0


def mean_fn(x1, amp, sigma_intrinsic, compute_var=False):
    # vector observable, nonlinear in x1; measurement variance per component
    g = amp * jnp.exp(0.6 * x1[:, None]) / (1. + R[None, :] * jnp.exp(0.3 * x1[:, None]))
    var = jnp.broadcast_to(0.01 * (1. + R)[None, :], g.shape)
    return g, var


def g_np(y, amp):
    return amp * np.exp(0.6 * y[..., None]) / (1. + R * np.exp(0.3 * y[..., None]))


def sigma_of_lnM(lnM, s0, slope):
    return s0 * np.exp(slope * lnM)


def make_cluster(mu, width, n=512):
    lnM = np.linspace(mu - 9. * width - 3., mu + 9. * width + 3., n)
    cpdf = 7.3 * np.exp(-0.5 * ((lnM - mu) / width)**2)     # arbitrary normalisation
    return lnM, cpdf


def reference(lnM, cpdf, amp, s0, slope, n_gh=80):
    """Independent: Gauss-Hermite in y at every mass point, Simpson in lnM."""
    t, w = np.polynomial.hermite_e.hermegauss(n_gh)          # weight exp(-t^2/2)
    w = w / np.sqrt(2. * np.pi)
    p = cpdf / integrate.simpson(cpdf, x=lnM)
    x1 = A0 + B0 * lnM
    sig = sigma_of_lnM(lnM, s0, slope)
    y = x1[:, None] + sig[:, None] * t[None, :]              # (n_m, n_gh)
    g = g_np(y, amp)                                          # (n_m, n_gh, K)
    Eg = np.einsum("mtk,t->mk", g, w)
    Eg2 = np.einsum("mtk,t->mk", g**2, w)
    mean = integrate.simpson(Eg * p[:, None], x=lnM, axis=0)
    second = integrate.simpson((Eg2 + 0.01 * (1. + R)[None, :]) * p[:, None], x=lnM, axis=0)
    return mean, second - mean**2


k_stat = build_stacked_kernel(layer0_fn, mean_fn)
k_xdep = build_stacked_kernel(layer0_fn, mean_fn, xdep_scatter_layer0=True, xdep_chunk=256)
k_xdep_odd = build_stacked_kernel(layer0_fn, mean_fn, xdep_scatter_layer0=True, xdep_chunk=300)
k_xdep_b8 = build_stacked_kernel(layer0_fn, mean_fn, xdep_scatter_layer0=True, xdep_src_bin=8)
k_xdep_b32 = build_stacked_kernel(layer0_fn, mean_fn, xdep_scatter_layer0=True, xdep_src_bin=32)

lnM_ref = np.linspace(-6., 8., 2048)
SIG_MIN = jnp.float64(1e-5)
amp = jnp.float64(1.7)
fails = []


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        fails.append(name)


def run(kernel, lnM, cpdf, s0, slope, full=False):
    table = jnp.asarray(sigma_of_lnM(lnM_ref, s0, slope))
    return kernel(jnp.asarray(cpdf), jnp.asarray(lnM), (jnp.float64(A0), jnp.float64(B0)),
                  (), (amp,), jnp.float64(s0), SIG_MIN, N_PTS, True, 2, full,
                  table, jnp.asarray(lnM_ref))


# 1. constant sigma: direct sum == stationary
for (mu, width, s0) in [(1.0, 0.25, 0.30), (0.2, 0.10, 0.45), (2.0, 0.40, 0.15)]:
    lnM, cpdf = make_cluster(mu, width)
    m_s, v_s = run(k_stat, lnM, cpdf, s0, 0.)
    m_x, v_x = run(k_xdep, lnM, cpdf, s0, 0.)
    dm = float(np.max(np.abs(np.asarray(m_x) / np.asarray(m_s) - 1.)))
    dv = float(np.max(np.abs(np.asarray(v_x) / np.asarray(v_s) - 1.)))
    check(f"1 constant sigma == stationary (mu {mu}, width {width}, sigma {s0})",
          dm < 1e-9 and dv < 1e-8, f"max rel diff mean {dm:.2e}, var {dv:.2e}")

# 2. varying sigma: direct sum == Gauss-Hermite reference; 3. slope matters
for (mu, width, s0, slope) in [(1.0, 0.25, 0.30, -0.30), (0.2, 0.10, 0.45, 0.25),
                               (2.0, 0.40, 0.15, -0.60), (1.0, 0.25, 0.30, 0.0)]:
    # input mass pdf on 4096 points: the kernel interpolates it linearly onto its uniform
    # layer-0 grid (a step shared with the stationary branch), an O(h^2) broadening that at
    # 512 input points dominates the comparison (printed below for information)
    lnM, cpdf = make_cluster(mu, width, n=4096)
    m_x, v_x = run(k_xdep, lnM, cpdf, s0, slope)
    m_r, v_r = reference(lnM, cpdf, float(amp), s0, slope)
    dm = float(np.max(np.abs(np.asarray(m_x) / m_r - 1.)))
    dv = float(np.max(np.abs(np.asarray(v_x) / v_r - 1.)))
    check(f"2 varying sigma == reference (mu {mu}, width {width}, sigma0 {s0}, slope {slope})",
          dm < 1e-6 and dv < 2e-5, f"max rel diff mean {dm:.2e}, var {dv:.2e}")
    lnM5, cpdf5 = make_cluster(mu, width, n=512)
    m_5, v_5 = run(k_xdep, lnM5, cpdf5, s0, slope)
    m_r5, v_r5 = reference(lnM5, cpdf5, float(amp), s0, slope)
    print(f"     (512-point input: max rel diff mean "
          f"{float(np.max(np.abs(np.asarray(m_5) / m_r5 - 1.))):.2e}, "
          f"var {float(np.max(np.abs(np.asarray(v_5) / v_r5 - 1.))):.2e})")
    if slope != 0.:
        m_0, _ = run(k_xdep, lnM, cpdf, s0, 0.)
        d0 = float(np.max(np.abs(np.asarray(m_x) / np.asarray(m_0) - 1.)))
        check(f"3 slope {slope} changes the mean", d0 > 1e-4, f"max rel change {d0:.2e}")
    m_o, v_o = run(k_xdep_odd, lnM, cpdf, s0, slope)
    do = float(np.max(np.abs(np.asarray(m_o) / np.asarray(m_x) - 1.)))
    check("2b chunk size that does not divide n_points gives the same result", do < 1e-12,
          f"max rel diff {do:.2e}")

# 2c. source points merged in groups (xdep_src_bin): against the sum over every point
for (mu, width, s0, slope) in [(1.0, 0.25, 0.30, -0.30), (0.2, 0.10, 0.45, 0.25), (2.0, 0.40, 0.15, -0.60)]:
    for n_in in (512, 4096):
        lnM, cpdf = make_cluster(mu, width, n=n_in)
        m_x, v_x = run(k_xdep, lnM, cpdf, s0, slope)
        for nb, kk, tol_m, tol_v in ((8, k_xdep_b8, 2e-7, 5e-5), (32, k_xdep_b32, 5e-6, 5e-4)):   # stated accuracies of the grouping
            m_g, v_g = run(kk, lnM, cpdf, s0, slope)
            dm = float(np.max(np.abs(np.asarray(m_g) / np.asarray(m_x) - 1.)))
            dv = float(np.max(np.abs(np.asarray(v_g) / np.asarray(v_x) - 1.)))
            check(f"2c groups of {nb} == every point (mu {mu}, sigma0 {s0}, slope {slope}, {n_in}-point input)",
                  dm < tol_m and dv < tol_v, f"max rel diff mean {dm:.2e}, var {dv:.2e}")

# 4. diag(full covariance) == variance
lnM, cpdf = make_cluster(1.0, 0.25)
m_f, v_f, c_f = run(k_xdep, lnM, cpdf, 0.30, -0.30, full=True)
dd = float(np.max(np.abs(np.diag(np.asarray(c_f)) / np.asarray(v_f) - 1.)))
check("4 diag(full covariance) == variance", dd < 1e-10, f"max rel diff {dd:.2e}")

# 5. jit(vmap) over clusters
cl = [make_cluster(1.0, 0.25), make_cluster(0.2, 0.10), make_cluster(2.0, 0.40)]
lnM_b = jnp.asarray(np.stack([c[0] for c in cl]))
cpdf_b = jnp.asarray(np.stack([c[1] for c in cl]))
s0_b = np.array([0.30, 0.45, 0.15])
tab_b = jnp.asarray(np.stack([sigma_of_lnM(lnM_ref, s, -0.3) for s in s0_b]))


def one(cp, lm, tab, s0c):
    return k_xdep(cp, lm, (jnp.float64(A0), jnp.float64(B0)), (), (amp,), s0c, SIG_MIN,
                  N_PTS, True, 2, False, tab, jnp.asarray(lnM_ref))


m_b, v_b = jax.jit(jax.vmap(one))(cpdf_b, lnM_b, tab_b, jnp.asarray(s0_b))
dmax = 0.
for i in range(3):
    m_i, v_i = run(k_xdep, cl[i][0], cl[i][1], s0_b[i], -0.3)
    dmax = max(dmax, float(np.max(np.abs(np.asarray(m_b[i]) / np.asarray(m_i) - 1.))),
               float(np.max(np.abs(np.asarray(v_b[i]) / np.asarray(v_i) - 1.))))
check("5 jit(vmap) == per-cluster calls", dmax < 1e-12, f"max rel diff {dmax:.2e}")

print()
if fails:
    print(f"TEST FAILED: {len(fails)} check(s): {fails}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
