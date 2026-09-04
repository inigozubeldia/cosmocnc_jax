"""Rigorous unit test suite for the Tinker10 and Castro23 HMF implementations
(2026-08-14). Self-contained (cosmocncenv, CPU, ~1 min): formula-level and
machinery-level identities that must hold EXACTLY or to quantified quadrature
accuracy. External-package cross-checks (hmf package, CCToolkit end-to-end)
and end-to-end NumPy<->JAX pipeline parity live in the calling analysis' own
test suite.

Tests:
  T1  Tinker10 triple parity (JAX-jit / JAX-class / NumPy) — dense z x Delta x
      sigma grid, both interp modes.                              gate 1e-13
  T2  Tinker10 normalisation identity int f(nu) dnu = 1 (the analytic alpha is
      the closed form of exactly this integral).                  gate 5e-4
  T3  Tinker10 analytic alpha vs the paper's fitted Table-4 alpha (z=0,
      tabulated Delta).                                           gate 0.5%
  T4  Castro23 normalisation identity int f(nu) dnu = 1 at fixed (dlns, Om)
      (validates A(p,q) numerically).                             gate 5e-4
  T5  Castro23 multiplicity NumPy (castro23_nuf_nu) vs JAX.       gate 1e-14
  T6  M_200c->M_vir Newton: converged residual of the defining equation
      M_DEL/M_vir = m(C)/m(c_vir), plus NumPy<->JAX parity.       gates 1e-10 / 1e-13
  T7  Jacobian dlnM_vir/dlnM_200c (grid gradient) vs direct finite difference
      of the Newton solve (independent path).                     gate 1e-4
  T8  Number conservation under the 200c->vir change of variables with an
      analytic power-law sigma(R): int dn/dlnM200c == int dn/dlnMvir over
      mapped limits.                                              gate 1e-4
  T9  Stress: finite + positive over extreme (z, Delta, sigma) and
      (Om, dlns, sigma) grids.
  T10 z>3 cap: g(z=3) == g(z=5) exactly, both codes.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = ""
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # this repo
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), "cosmocnc"))    # sibling NumPy cosmocnc checkout

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cosmocnc_jax.hmf import (g_sigma_tinker10_jit, nuf_nu_castro23_jit,
                              TINKER08_DELTA_LOG, TINKER08_DELTA_LIN,
                              TINKER10_ALPHA, TINKER10_BETA0, TINKER10_GAMMA0,
                              TINKER10_PHI0, TINKER10_ETA0, TINKER10_DELTA_C)
from cosmocnc_jax import hmf as jhmf
from cosmocnc_jax import mass_conversion as jmc
from cosmocnc import hmf as nhmf
from cosmocnc import mass_conversion as nmc

FAILURES = []


def check(name, value, gate):
    ok = value < gate
    print(f"  {name:58s} {value:.3e}  (gate {gate:.0e})  {'OK' if ok else '** FAIL **'}",
          flush=True)
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- T1
print("T1: Tinker10 triple parity (jit / class / NumPy), dense grid")
sig = np.exp(np.linspace(np.log(0.3), np.log(3.0), 20))
worst = 0.0
for z in [0.0, 0.5, 1.0, 2.0, 3.0, 3.5]:
    for Delta in [200., 300., 450., 800., 1300., 3200.]:
        for mode, tab in (("log", TINKER08_DELTA_LOG), ("linear", TINKER08_DELTA_LIN)):
            gj = np.array(g_sigma_tinker10_jit(jnp.asarray(sig), z, Delta, tab,
                                               TINKER10_BETA0, TINKER10_GAMMA0,
                                               TINKER10_PHI0, TINKER10_ETA0,
                                               interp_log=(mode == "log")))
            gn = nhmf.f_sigma(sig, redshift=z, hmf_type="Tinker10", Delta=Delta,
                              other_params={"interp_tinker": mode})
            gc = np.array(jhmf.f_sigma(jnp.asarray(sig), redshift=z, hmf_type="Tinker10",
                                       Delta=Delta, other_params={"interp_tinker": mode}))
            worst = max(worst, np.max(np.abs(gj/gn - 1)), np.max(np.abs(gc/gn - 1)))
check("max rel diff over 72 (z,Delta,mode) combos x 20 sigma", worst, 1e-13)

# ---------------------------------------------------------------- T2
print("T2: Tinker10 normalisation  int f(nu) dnu = 1")
lnnu = np.linspace(np.log(1e-8), np.log(20.0), 40000)
nu = np.exp(lnnu)
worst = 0.0
from scipy.special import gammaln as _gln0
for z in [0.0, 1.0, 3.0]:
    for Delta in [200., 450., 1300.]:
        sigma_of_nu = TINKER10_DELTA_C / nu
        g = np.array(g_sigma_tinker10_jit(jnp.asarray(sigma_of_nu), z, Delta,
                                          TINKER08_DELTA_LOG, TINKER10_BETA0,
                                          TINKER10_GAMMA0, TINKER10_PHI0,
                                          TINKER10_ETA0, interp_log=True))
        integral = np.trapezoid(g, lnnu)          # int nu f(nu) dln nu = int f dnu
        # analytic [0, nu_min] tail: f ~ alpha*[nu^(2eta) + beta^(-2phi)*
        # nu^(2eta-2phi)] there (exp ~ 1); the tail carries up to ~10% at
        # high z/Delta where 2eta -> -0.9 — NOT quadrature-negligible.
        zc = min(z, 3.0)
        ld = np.log10(Delta)
        be = np.interp(ld, np.array(TINKER08_DELTA_LOG), np.array(TINKER10_BETA0))*(1+zc)**0.20
        ga = np.interp(ld, np.array(TINKER08_DELTA_LOG), np.array(TINKER10_GAMMA0))*(1+zc)**(-0.01)
        ph = np.interp(ld, np.array(TINKER08_DELTA_LOG), np.array(TINKER10_PHI0))*(1+zc)**(-0.08)
        et = np.interp(ld, np.array(TINKER08_DELTA_LOG), np.array(TINKER10_ETA0))*(1+zc)**0.27
        alz = 1.0/(2.0**(et-ph-0.5)*be**(-2*ph)*ga**(-0.5-et)
                   *(2.0**ph*be**(2*ph)*np.exp(_gln0(et+0.5)) + ga**ph*np.exp(_gln0(0.5+et-ph))))
        nmin = nu[0]
        tail = alz*(nmin**(2*et+1)/(2*et+1) + be**(-2*ph)*nmin**(2*et-2*ph+1)/(2*et-2*ph+1))
        worst = max(worst, abs(integral + tail - 1.0))
check("max |int f dnu - 1| over 9 (z,Delta), tail-corrected", worst, 5e-4)

# ---------------------------------------------------------------- T3
print("T3: analytic alpha vs Table-4 fitted alpha (z=0, tabulated Delta)")
from scipy.special import gammaln as _gln
B0, G0 = np.array(TINKER10_BETA0), np.array(TINKER10_GAMMA0)
P0, E0 = np.array(TINKER10_PHI0), np.array(TINKER10_ETA0)
al = 1.0/(2.0**(E0-P0-0.5)*B0**(-2*P0)*G0**(-0.5-E0)
          *(2.0**P0*B0**(2*P0)*np.exp(_gln(E0+0.5)) + G0**P0*np.exp(_gln(0.5+E0-P0))))
check("max |alpha_analytic/alpha_Table4 - 1|", np.max(np.abs(al/np.array(TINKER10_ALPHA) - 1)), 5e-3)

# ---------------------------------------------------------------- T4
print("T4: Castro23 normalisation  int f(nu) dnu = 1 at fixed (dlns, Om)")
worst = 0.0
for dlns in [-2.2, -1.5, -0.8]:
    for Om in [0.15, 0.30, 0.90]:
        delta_c = (3.0/20.0)*(12.0*np.pi)**(2.0/3.0)*(1.0 + 0.012299*np.log10(Om))
        sigma_of_nu = delta_c / nu
        nufv = np.array(nuf_nu_castro23_jit(jnp.asarray(sigma_of_nu),
                                            jnp.float64(dlns), jnp.float64(Om)))
        integral = np.trapezoid(nufv, lnnu)
        # analytic [0, nu_min] tail (u = sqrt(a) nu, exp ~ 1):
        # int f dnu = A sqrt(2/pi) [u_min^q/q + u_min^(q-2p)/(q-2p)]
        aR = 0.7962 + 0.1449*(dlns + 0.6125)**2
        qR = 0.3688 - 0.2804*(dlns + 0.5)
        a_c = aR*Om**(-0.0658)
        p_c = -0.5612 - 0.4743*(dlns + 0.5)
        q_c = qR*Om**0.0251
        A_pq = 1.0/(2.0**(-0.5 - p_c + q_c/2.0)/np.sqrt(np.pi)
                    *(2.0**p_c*np.exp(_gln0(q_c/2.0)) + np.exp(_gln0(-p_c + q_c/2.0))))
        umin = np.sqrt(a_c)*nu[0]
        tail = A_pq*np.sqrt(2.0/np.pi)*(umin**q_c/q_c + umin**(q_c-2*p_c)/(q_c-2*p_c))
        worst = max(worst, abs(integral + tail - 1.0))
check("max |int f dnu - 1| over 9 (dlns,Om), tail-corrected", worst, 5e-4)

# ---------------------------------------------------------------- T5
print("T5: Castro23 multiplicity NumPy vs JAX, dense grid")
sigg = np.exp(np.linspace(np.log(0.25), np.log(4.0), 25))
worst = 0.0
for dlns in [-2.5, -1.8, -1.2, -0.6]:
    for Om in [0.10, 0.20, 0.35, 0.70, 1.0]:
        a = nhmf.castro23_nuf_nu(sigg, dlns, Om)
        b = np.array(nuf_nu_castro23_jit(jnp.asarray(sigg), jnp.float64(dlns), jnp.float64(Om)))
        worst = max(worst, np.max(np.abs(b/a - 1)))
check("max rel diff over 20 (dlns,Om) x 25 sigma", worst, 1e-14)

# ---------------------------------------------------------------- T6
print("T6: Newton solve — defining-equation residual + NumPy<->JAX parity")
logM_grid = np.linspace(np.log(1e11), np.log(1e18), 400)
sigma_grid = 2.6*np.exp(-0.24*(logM_grid - np.log(1e14)))
M200 = np.exp(np.linspace(np.log(1e13), np.log(1e16), 30))
worst_res, worst_par = 0.0, 0.0
for rho_c_z, Om_z, D_z in [(1.3e11, 0.20, 0.95), (1.9e11, 0.45, 0.75), (5.5e11, 0.90, 0.45)]:
    Mv_n = nmc.solve_M_vir_from_M_200c(M200, rho_c_z, Om_z, D_z, logM_grid, sigma_grid)
    Mv_j = np.array([float(jmc._m200c_to_mvir_one(
        jnp.float64(m), jnp.float64(rho_c_z), jnp.float64(Om_z), jnp.float64(D_z),
        jnp.asarray(logM_grid), jnp.asarray(sigma_grid))) for m in M200])
    worst_par = max(worst_par, np.max(np.abs(Mv_n/Mv_j - 1)))
    # residual of the defining equation at the converged solution
    dcv = nmc.delta_c_virial(Om_z)
    sv = np.interp(np.log(Mv_n), logM_grid, sigma_grid)
    cv = nmc.b13_cvir_sigma_based(sv, D_z)
    Rv = (3.0*Mv_n/(4.0*np.pi*dcv*rho_c_z))**(1.0/3.0)
    RD = (3.0*M200/(4.0*np.pi*200.0*rho_c_z))**(1.0/3.0)
    C = RD/(Rv/cv)
    res = np.abs(M200/Mv_n - nmc._nfw_m(C)/nmc._nfw_m(cv))
    worst_res = max(worst_res, res.max())
check("max |residual| of M_DEL/M_vir - m(C)/m(c_vir)", worst_res, 1e-10)
check("max NumPy-vs-JAX rel diff on M_vir", worst_par, 1e-13)

# ---------------------------------------------------------------- T7
print("T7: grid-gradient Jacobian vs direct finite difference of the Newton")
rho_c_z, Om_z, D_z = 1.9e11, 0.45, 0.75
n_g = 4096
lnM200 = np.linspace(np.log(1e13), np.log(1e16), n_g)
Mv = nmc.solve_M_vir_from_M_200c(np.exp(lnM200), rho_c_z, Om_z, D_z, logM_grid, sigma_grid)
dln = lnM200[1] - lnM200[0]
jac_grid = 1.0 + np.gradient(np.log(Mv/np.exp(lnM200)), dln)
eps = 1e-4
worst = 0.0
for i in np.linspace(50, n_g - 50, 12).astype(int):
    Mp = np.exp(lnM200[i] + eps); Mm = np.exp(lnM200[i] - eps)
    Mvp = nmc.solve_M_vir_from_M_200c(np.array([Mp]), rho_c_z, Om_z, D_z, logM_grid, sigma_grid)[0]
    Mvm = nmc.solve_M_vir_from_M_200c(np.array([Mm]), rho_c_z, Om_z, D_z, logM_grid, sigma_grid)[0]
    jac_fd = (np.log(Mvp) - np.log(Mvm))/(2*eps)
    worst = max(worst, abs(jac_grid[i]/jac_fd - 1))
check("max |jac_grid/jac_fd - 1| at 12 interior points", worst, 1e-4)

# ---------------------------------------------------------------- T8
print("T8: number conservation under the 200c->vir change of variables")
# analytic power-law sigma(R): sigma = s0 (R/R0)^(-s)  ->  dln sigma/dlnR = -s
s_slope, s0 = 0.9, 1.8
rho_m = 8.6e10
R_of = lambda M: (3.0*M/(4.0*np.pi*rho_m))**(1.0/3.0)
R0 = R_of(1e14)
sig_of_M = lambda M: s0*(R_of(M)/R0)**(-s_slope)
lg = np.linspace(np.log(1e11), np.log(1e18), 2000)
sg = sig_of_M(np.exp(lg))
Om_z = 0.45
dc = (3.0/20.0)*(12.0*np.pi)**(2.0/3.0)*(1.0 + 0.012299*np.log10(Om_z))

def dndlnMvir(Mvir):
    return (nhmf.castro23_nuf_nu(sig_of_M(Mvir), -s_slope, Om_z)
            * rho_m / Mvir * (s_slope/3.0))

lnA, lnB = np.log(3e13), np.log(3e15)
lnM_f = np.linspace(lnA, lnB, 6000)
M_f = np.exp(lnM_f)
Mv_f = nmc.solve_M_vir_from_M_200c(M_f, rho_c_z, Om_z, D_z, lg, sg)
jac_f = 1.0 + np.gradient(np.log(Mv_f/M_f), lnM_f[1]-lnM_f[0])
N_200c = np.trapezoid(dndlnMvir(Mv_f)*jac_f, lnM_f)
lnMv_f = np.linspace(np.log(Mv_f[0]), np.log(Mv_f[-1]), 6000)
N_vir = np.trapezoid(dndlnMvir(np.exp(lnMv_f)), lnMv_f)
check("|N_200c-grid / N_vir-grid - 1|", abs(N_200c/N_vir - 1), 1e-4)

# ---------------------------------------------------------------- T9
print("T9: stress — finite and positive over extreme grids")
bad = 0
for z in [0.0, 1.5, 3.5]:
    for Delta in [200., 1300., 3200.]:
        g = np.array(g_sigma_tinker10_jit(jnp.asarray(np.linspace(0.2, 5.0, 50)), z, Delta,
                                          TINKER08_DELTA_LOG, TINKER10_BETA0, TINKER10_GAMMA0,
                                          TINKER10_PHI0, TINKER10_ETA0, interp_log=True))
        bad += int(np.any(~np.isfinite(g)) or np.any(g < 0))
for Om in [0.05, 0.30, 1.0]:
    for dlns in [-2.6, -1.5, -0.3]:
        # positivity holds on the model's NORMALISABLE domain dlns > -2.7317
        # (q - 2p > 0); beyond it the signed A(p,q) legitimately goes negative
        # — that regime is covered by T11 instead.
        v = np.array(nuf_nu_castro23_jit(jnp.asarray(np.linspace(0.2, 5.0, 50)),
                                         jnp.float64(dlns), jnp.float64(Om)))
        bad += int(np.any(~np.isfinite(v)) or np.any(v < 0))
check("count of grids with non-finite/negative values", float(bad), 0.5)

# ---------------------------------------------------------------- T11
print("T11: signed-Gamma domain corner (dlns < -2.7317) vs scipy signed reference")
from scipy.special import gamma as _sgamma_ref
worst = 0.0
for dlns in [-3.0, -2.8]:
    for Om in [0.15, 0.45]:
        delta_c = (3.0/20.0)*(12.0*np.pi)**(2.0/3.0)*(1.0 + 0.012299*np.log10(Om))
        sv = np.array([0.5, 1.0, 2.0])
        nuv = delta_c/sv
        aR = 0.7962 + 0.1449*(dlns + 0.6125)**2
        qR = 0.3688 - 0.2804*(dlns + 0.5)
        a_c = aR*Om**(-0.0658)
        p_c = -0.5612 - 0.4743*(dlns + 0.5)
        q_c = qR*Om**0.0251
        A_pq = 1.0/(2.0**(-0.5 - p_c + q_c/2.0)/np.sqrt(np.pi)
                    *(2.0**p_c*_sgamma_ref(q_c/2.0) + _sgamma_ref(-p_c + q_c/2.0)))
        ref = (A_pq*np.sqrt(2.0*a_c*nuv**2/np.pi)*np.exp(-a_c*nuv**2/2.0)
               *(1.0 + 1.0/(a_c*nuv**2)**p_c)*(nuv*np.sqrt(a_c))**(q_c - 1.0))
        vj = np.array(nuf_nu_castro23_jit(jnp.asarray(sv), jnp.float64(dlns), jnp.float64(Om)))
        vn = nhmf.castro23_nuf_nu(sv, dlns, Om)
        worst = max(worst, np.max(np.abs(vj/ref - 1)), np.max(np.abs(vn/ref - 1)))
        assert np.all(ref < 0), "corner reference should be negative (diverging normalisation)"
check("max rel diff vs signed reference at 4 corner combos", worst, 1e-13)

# ---------------------------------------------------------------- T10
print("T10: z-evolution cap — g(z=3) == g(z=5)")
g3 = np.array(g_sigma_tinker10_jit(jnp.asarray(sig), 3.0, 300., TINKER08_DELTA_LOG,
                                   TINKER10_BETA0, TINKER10_GAMMA0, TINKER10_PHI0,
                                   TINKER10_ETA0, interp_log=True))
g5 = np.array(g_sigma_tinker10_jit(jnp.asarray(sig), 5.0, 300., TINKER08_DELTA_LOG,
                                   TINKER10_BETA0, TINKER10_GAMMA0, TINKER10_PHI0,
                                   TINKER10_ETA0, interp_log=True))
g3n = nhmf.f_sigma(sig, redshift=3.0, hmf_type="Tinker10", Delta=300.,
                   other_params={"interp_tinker": "log"})
g5n = nhmf.f_sigma(sig, redshift=5.0, hmf_type="Tinker10", Delta=300.,
                   other_params={"interp_tinker": "log"})
check("max |g(3)/g(5) - 1| (JAX + NumPy)",
      max(np.max(np.abs(g3/g5 - 1)), np.max(np.abs(g3n/g5n - 1))), 1e-15)

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("RESULT: ALL PASSED")
