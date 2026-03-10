"""
Benchmark: classy_sz vs classy_sz_jax cosmology tool.

Compares numerical accuracy and timing of:
  1. HMF matrix
  2. Background quantities (D_A, E_z, rho_c)
  3. Full unbinned likelihood (data_lik_from_abundance=True)
  4. Repeated update_params + get_log_lik (MCMC-like)

Usage:
  python tests/test_jax_cosmo.py
"""

# ── Thread control (must be set before any library imports) ──────────────
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

import cosmocnc_jax


# ── Configuration ────────────────────────────────────────────────────────

def get_params(cosmology_tool="classy_sz"):
    cnc_params_update = {
        "cluster_catalogue": "SO_sim_0",
        "observables": [["q_so_sim"]],
        "obs_select": "q_so_sim",
        "data_lik_from_abundance": True,
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

        "cosmology_tool": cosmology_tool,
        "cosmo_param_density": "critical",
        "cosmo_model": "lcdm",
        "hmf_calc": "cnc",
        "interp_tinker": "linear",

        "stacked_likelihood": False,
        "likelihood_type": "unbinned",
    }

    scal_rel_update = {
        "corr_lnq_lnp": 0.,
        "bias_sz": 0.8,
        "dof": 0.,
    }

    cosmo_update = {}

    return cnc_params_update, scal_rel_update, cosmo_update


def setup_and_init(cnc_params_update, scal_rel_update, cosmo_update):
    nc = cosmocnc_jax.cluster_number_counts()
    # Deep copy to avoid shared-dict mutation between instances
    nc.cnc_params = dict(nc.cnc_params)
    nc.scal_rel_params = dict(nc.scal_rel_params)
    nc.cosmo_params = dict(nc.cosmo_params)
    nc.cnc_params.update(cnc_params_update)
    nc.scal_rel_params.update(scal_rel_update)
    nc.cosmo_params.update(cosmo_update)
    nc.initialise()
    return nc


def compare_arrays(name, arr1, arr2, rtol=1e-4, atol=1e-10):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if arr1.shape != arr2.shape:
        print(f"  FAIL {name}: shape mismatch {arr1.shape} vs {arr2.shape}")
        return False
    max_abs = np.max(np.abs(arr1 - arr2))
    max_rel = np.max(np.abs(arr1 - arr2) / np.maximum(np.abs(arr1), 1e-30))
    passed = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    if not passed:
        idx = np.unravel_index(np.argmax(np.abs(arr1 - arr2)), arr1.shape)
        print(f"        worst at {idx}: ref={arr1[idx]:.10e}, jax={arr2[idx]:.10e}")
    return passed


def compare_scalars(name, v1, v2, rtol=1e-4):
    v1 = float(np.asarray(v1).ravel()[0])
    v2 = float(np.asarray(v2).ravel()[0])
    rel = abs(v1 - v2) / max(abs(v1), 1e-30)
    passed = rel < rtol
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: ref={v1:.10e}, jax={v2:.10e}, rel={rel:.3e}")
    return passed


def main():
    print("=" * 70)
    print("BENCHMARK: classy_sz vs classy_sz_jax")
    print("=" * 70)

    all_passed = True
    timings = {}

    # ── Init reference (classy_sz) ──
    print("\n1. Initialising with classy_sz (reference)...")
    cnc_ref, sr_ref, cosmo_ref = get_params("classy_sz")
    t0 = time.time()
    nc_ref = setup_and_init(cnc_ref, sr_ref, cosmo_ref)
    timings["init_ref"] = time.time() - t0
    print(f"   Time: {timings['init_ref']:.2f}s")

    # ── Init JAX (classy_sz_jax) ──
    print("\n2. Initialising with classy_sz_jax...")
    cnc_jax, sr_jax, cosmo_jax = get_params("classy_sz_jax")
    t0 = time.time()
    nc_jax = setup_and_init(cnc_jax, sr_jax, cosmo_jax)
    timings["init_jax"] = time.time() - t0
    print(f"   Time: {timings['init_jax']:.2f}s")

    # ── Compute number counts ──
    print("\n3. Computing number counts...")

    t0 = time.time()
    nc_ref.get_number_counts()
    timings["nc_ref"] = time.time() - t0
    print(f"   classy_sz:     {timings['nc_ref']:.3f}s (hmf={nc_ref.t_hmf:.3f}s)")

    t0 = time.time()
    nc_jax.get_number_counts()
    timings["nc_jax"] = time.time() - t0
    print(f"   classy_sz_jax: {timings['nc_jax']:.3f}s (hmf={nc_jax.t_hmf:.3f}s)")

    # ── Compare HMF ──
    print("\n4. Comparing HMF matrix...")
    p = compare_arrays("hmf_matrix", nc_ref.hmf_matrix, nc_jax.hmf_matrix, rtol=1e-3)
    all_passed &= p

    # ── Compare background quantities ──
    print("\n5. Comparing background quantities...")
    p = compare_arrays("D_A", nc_ref.D_A, nc_jax.D_A, rtol=1e-3)
    all_passed &= p
    p = compare_arrays("E_z", nc_ref.E_z, nc_jax.E_z, rtol=1e-3)
    all_passed &= p
    p = compare_arrays("D_l_CMB", nc_ref.D_l_CMB, nc_jax.D_l_CMB, rtol=1e-3)
    all_passed &= p
    p = compare_arrays("rho_c", nc_ref.rho_c, nc_jax.rho_c, rtol=1e-3)
    all_passed &= p

    # ── Compare n_tot ──
    print("\n6. Comparing abundance...")
    p = compare_scalars("n_tot", nc_ref.n_tot, nc_jax.n_tot, rtol=1e-3)
    all_passed &= p

    # ── Compare likelihood ──
    print("\n7. Computing likelihood...")
    t0 = time.time()
    ll_ref = nc_ref.get_log_lik()
    timings["lik_ref"] = time.time() - t0

    t0 = time.time()
    ll_jax = nc_jax.get_log_lik()
    timings["lik_jax"] = time.time() - t0

    p = compare_scalars("log_lik", ll_ref, ll_jax, rtol=1e-3)
    all_passed &= p
    print(f"   classy_sz:     {timings['lik_ref']:.4f}s")
    print(f"   classy_sz_jax: {timings['lik_jax']:.4f}s")

    # ── MCMC-like repeated evaluations ──
    print("\n8. Repeated update_params + get_log_lik (MCMC simulation)...")
    n_evals = 5
    sigma_8_vec = np.linspace(0.808, 0.815, n_evals)

    scal_ref = dict(nc_ref.scal_rel_params)
    scal_jax = dict(nc_jax.scal_rel_params)

    ll_ref_arr = np.zeros(n_evals)
    ll_jax_arr = np.zeros(n_evals)

    print(f"   Running {n_evals} evals with classy_sz...")
    t0 = time.time()
    for i in range(n_evals):
        cp = dict(nc_ref.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc_ref.update_params(cp, scal_ref)
        ll_ref_arr[i] = nc_ref.get_log_lik()
    timings["scan_ref"] = time.time() - t0

    print(f"   Running {n_evals} evals with classy_sz_jax...")
    t0 = time.time()
    for i in range(n_evals):
        cp = dict(nc_jax.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc_jax.update_params(cp, scal_jax)
        ll_jax_arr[i] = nc_jax.get_log_lik()
    timings["scan_jax"] = time.time() - t0

    p = compare_arrays("log_lik_scan", ll_ref_arr, ll_jax_arr, rtol=1e-3)
    all_passed &= p

    t_ref = timings["scan_ref"] / n_evals
    t_jax = timings["scan_jax"] / n_evals

    # ── Summary ──
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"  Init:            ref={timings['init_ref']:.2f}s  jax={timings['init_jax']:.2f}s")
    print(f"  Number counts:   ref={timings['nc_ref']:.3f}s  jax={timings['nc_jax']:.3f}s")
    print(f"  Likelihood:      ref={timings['lik_ref']:.4f}s  jax={timings['lik_jax']:.4f}s")
    print(f"  MCMC scan ({n_evals}x): ref={timings['scan_ref']:.3f}s ({t_ref:.3f}s/eval)  jax={timings['scan_jax']:.3f}s ({t_jax:.3f}s/eval)")
    print(f"  Speedup (per eval): {t_ref/max(t_jax, 1e-10):.1f}x")

    if all_passed:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
