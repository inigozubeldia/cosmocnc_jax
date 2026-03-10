"""
Benchmark: cosmocnc (NumPy) vs cosmocnc_jax (JAX, classy_sz_jax)

Compares unbinned likelihood with backward convolution:
  1. Numerical accuracy (log_lik agreement)
  2. Per-evaluation timing (MCMC-like repeated evals)
  3. Timing breakdown (update_params, get_hmf, get_abundance, get_log_lik)

Usage:
  taskset -c 0-9 python tests/benchmark_numpy_vs_jax.py
"""

# ── Thread control (must be set before any library imports) ──────────
import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc
import cosmocnc_jax


# ── Configuration ────────────────────────────────────────────────────

SHARED_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim"], ["p_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False,
    "compute_abundance_matrix": True,

    "number_cores_hmf": 1,
    "number_cores_abundance": 1,
    "number_cores_data": 1,
    "number_cores_stacked": 1,
    "parallelise_type": "redshift",

    "obs_select_min": 5.,
    "obs_select_max": 200.,

    "z_min": 0.01,
    "z_max": 3.,
    "n_z": 100,

    "M_min": 1e13,
    "M_max": 1e16,
    "n_points": 16384,
    "n_points_data_lik": 2048,
    "sigma_mass_prior": 10,
    "downsample_hmf_bc": 2,
    "delta_m_with_ref": True,
    "scalrel_type_deriv": "numerical",

    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "hmf_calc": "cnc",
    "interp_tinker": "linear",

    "stacked_likelihood": False,
    "likelihood_type": "unbinned",
}

SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}


def setup(pkg, cosmology_tool):
    nc = pkg.cluster_number_counts()
    nc.cnc_params = dict(nc.cnc_params)
    nc.scal_rel_params = dict(nc.scal_rel_params)
    nc.cosmo_params = dict(nc.cosmo_params)
    nc.cnc_params.update(SHARED_PARAMS)
    nc.cnc_params["cosmology_tool"] = cosmology_tool
    nc.scal_rel_params.update(SCAL_REL)
    nc.initialise()
    return nc


def timed_eval(nc, label=""):
    """Run update_params → get_log_lik with timing breakdown."""
    t0 = time.time()
    nc.get_hmf()
    t1 = time.time()
    nc.get_cluster_abundance()
    nc.get_number_counts()
    t2 = time.time()
    ll = nc.get_log_lik()
    t3 = time.time()
    return {
        "hmf": t1 - t0,
        "abundance_nc": t2 - t1,
        "log_lik": t3 - t2,
        "total": t3 - t0,
        "ll": float(np.asarray(ll).ravel()[0]),
    }


def main():
    print("=" * 70)
    print("BENCHMARK: cosmocnc (NumPy) vs cosmocnc_jax (JAX)")
    print("Unbinned likelihood with backward convolution")
    print("=" * 70)

    # ── Initialise ──
    print("\n1. Initialising cosmocnc (NumPy, classy_sz)...")
    t0 = time.time()
    nc_np = setup(cosmocnc, "classy_sz")
    t_init_np = time.time() - t0
    print(f"   Time: {t_init_np:.2f}s")

    print("\n2. Initialising cosmocnc_jax (JAX, classy_sz_jax)...")
    t0 = time.time()
    nc_jax = setup(cosmocnc_jax, "classy_sz_jax")
    t_init_jax = time.time() - t0
    print(f"   Time: {t_init_jax:.2f}s")

    # ── First evaluation (includes JIT compilation for JAX) ──
    print("\n3. First evaluation (warmup / JIT compilation)...")

    t0 = time.time()
    nc_np.get_number_counts()
    ll_np_first = nc_np.get_log_lik()
    t_warmup_np = time.time() - t0
    print(f"   cosmocnc:     {t_warmup_np:.3f}s  ll={float(ll_np_first):.4f}")

    t0 = time.time()
    nc_jax.get_number_counts()
    ll_jax_first = nc_jax.get_log_lik()
    t_warmup_jax = time.time() - t0
    print(f"   cosmocnc_jax: {t_warmup_jax:.3f}s  ll={float(np.asarray(ll_jax_first)):.4f}")

    rel_first = abs(float(ll_np_first) - float(np.asarray(ll_jax_first))) / max(abs(float(ll_np_first)), 1e-30)
    print(f"   log_lik agreement: rel={rel_first:.3e}")

    # ── MCMC-like repeated evaluations ──
    print("\n4. MCMC-like repeated evaluations (vary sigma_8)...")
    n_evals = 10
    sigma_8_vec = np.linspace(0.805, 0.818, n_evals)

    scal_np = dict(nc_np.scal_rel_params)
    scal_jax = dict(nc_jax.scal_rel_params)

    # -- NumPy --
    print(f"\n   Running {n_evals} evals with cosmocnc (NumPy)...")
    ll_np_arr = np.zeros(n_evals)
    times_np = []
    for i in range(n_evals):
        cp = dict(nc_np.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        t0 = time.time()
        nc_np.update_params(cp, scal_np)
        t1 = time.time()
        ll_np_arr[i] = nc_np.get_log_lik()
        t2 = time.time()
        times_np.append({"update": t1 - t0, "get_log_lik": t2 - t1, "total": t2 - t0})
        if i < 3 or i == n_evals - 1:
            print(f"     [{i}] update={t1-t0:.3f}s  get_log_lik={t2-t1:.3f}s  total={t2-t0:.3f}s  ll={ll_np_arr[i]:.2f}")

    # -- JAX --
    print(f"\n   Running {n_evals} evals with cosmocnc_jax (JAX)...")
    ll_jax_arr = np.zeros(n_evals)
    times_jax = []
    for i in range(n_evals):
        cp = dict(nc_jax.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        t0 = time.time()
        nc_jax.update_params(cp, scal_jax)
        t1 = time.time()
        ll_jax_arr[i] = nc_jax.get_log_lik()
        t2 = time.time()
        times_jax.append({"update": t1 - t0, "get_log_lik": t2 - t1, "total": t2 - t0})
        if i < 3 or i == n_evals - 1:
            print(f"     [{i}] update={t1-t0:.4f}s  get_log_lik={t2-t1:.4f}s  total={t2-t0:.4f}s  ll={ll_jax_arr[i]:.2f}")

    # ── Accuracy ──
    print("\n5. Numerical accuracy...")
    max_abs = np.max(np.abs(ll_np_arr - ll_jax_arr))
    max_rel = np.max(np.abs(ll_np_arr - ll_jax_arr) / np.maximum(np.abs(ll_np_arr), 1e-30))
    print(f"   log_lik: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

    # Normalised likelihood curve
    lik_np = np.exp(ll_np_arr - np.max(ll_np_arr))
    lik_jax = np.exp(ll_jax_arr - np.max(ll_jax_arr))
    lik_rel = np.max(np.abs(lik_np - lik_jax) / np.maximum(np.abs(lik_np), 1e-30))
    print(f"   normalised lik curve: max_rel={lik_rel:.3e}")

    # ── Timing summary ──
    # Skip first 2 evals as warmup
    skip = 2
    avg_np = {k: np.mean([t[k] for t in times_np[skip:]]) for k in times_np[0]}
    avg_jax = {k: np.mean([t[k] for t in times_jax[skip:]]) for k in times_jax[0]}

    print("\n" + "=" * 70)
    print("TIMING SUMMARY (average of last {} evals)".format(n_evals - skip))
    print("=" * 70)
    print(f"{'':30s} {'cosmocnc':>12s} {'cosmocnc_jax':>12s} {'speedup':>10s}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    for key in ["update", "get_log_lik", "total"]:
        t_np = avg_np[key]
        t_jax = avg_jax[key]
        speedup = t_np / max(t_jax, 1e-10)
        print(f"  {key:28s} {t_np:10.4f}s  {t_jax:10.4f}s  {speedup:8.1f}x")

    print(f"\n  Init (one-time):             {t_init_np:.2f}s vs {t_init_jax:.2f}s")
    print(f"  First eval (incl. JIT):      {t_warmup_np:.2f}s vs {t_warmup_jax:.2f}s")

    speedup_total = avg_np["total"] / max(avg_jax["total"], 1e-10)
    print(f"\n  OVERALL SPEEDUP: {speedup_total:.1f}x per MCMC evaluation")
    print(f"  ({avg_np['total']*1000:.0f}ms → {avg_jax['total']*1000:.0f}ms per eval)")


if __name__ == "__main__":
    main()
