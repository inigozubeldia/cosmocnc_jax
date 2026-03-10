"""
Comprehensive comparison test between cosmocnc (numpy) and cosmocnc_jax (JAX).

Compares:
  1. HMF matrix
  2. Cluster abundance (n_tot, dn/dz, dn/dSNR)
  3. Abundance matrix
  4. Binned likelihood (z_and_obs_select)
  5. Unbinned likelihood (data_lik_from_abundance = True)
  6. Unbinned likelihood (backward convolutional approach)
  7. Stacked likelihood
  8. Goodness of fit (C statistic)
  9. Extreme value statistics
 10. Likelihood evaluation timing comparison

Uses classy_sz as the cosmology tool and the SO-like simulated catalogue (SO_sim_0).

Usage:
  python tests/test_comparison.py                  # Run all tests
  python tests/test_comparison.py --skip-bc        # Skip backward convolutional (slow)
  python tests/test_comparison.py --skip-stacked   # Skip stacked likelihood
"""

# ── Thread control (must be set before any library imports) ──────────────
import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS          # CLASS/classy_sz OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS      # NumPy OpenBLAS
os.environ["MKL_NUM_THREADS"] = _N_THREADS           # NumPy MKL
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS       # numexpr
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""                          # Clear stale GPU cache flag
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable TF GPU before import (Blackwell compute 12.0 incompatible)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys

# Fix sys.path to avoid cosmocnc_jax/cosmocnc shadowing installed packages
sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

# ── Configuration (shared between both packages) ────────────────────────────

def get_params(observables=None, data_lik_from_abundance=False,
               stacked_likelihood=False, likelihood_type="unbinned"):
    """Return cnc_params, scal_rel_params, cosmo_params dicts for both packages."""

    if observables is None:
        observables = [["q_so_sim"], ["p_so_sim"]]

    cnc_params_update = {
        "cluster_catalogue": "SO_sim_0",
        "observables": observables,
        "obs_select": "q_so_sim",
        "data_lik_from_abundance": data_lik_from_abundance,
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

        "cosmology_tool": "classy_sz",
        "cosmo_param_density": "critical",
        "cosmo_model": "lcdm",
        "hmf_calc": "cnc",
        "interp_tinker": "linear",

        "binned_lik_type": "z_and_obs_select",

        "stacked_likelihood": stacked_likelihood,
        "stacked_data": ["p_so_sim_stacked"],
        "compute_stacked_cov": True,

        "likelihood_type": likelihood_type,

        "bins_edges_z": np.linspace(0.01, 3., 9),
        "bins_edges_obs_select": np.exp(np.linspace(np.log(5.), np.log(200.), 7)),
    }

    scal_rel_update = {
        "corr_lnq_lnp": 0.,
        "bias_sz": 0.8,
        "dof": 0.,
    }

    cosmo_update = {}

    return cnc_params_update, scal_rel_update, cosmo_update


def setup_and_init(pkg, cnc_params_update, scal_rel_update, cosmo_update):
    """Set up and initialise a cluster_number_counts object."""

    nc = pkg.cluster_number_counts()
    nc.cnc_params.update(cnc_params_update)
    nc.scal_rel_params.update(scal_rel_update)
    nc.cosmo_params.update(cosmo_update)
    nc.initialise()

    return nc


# ── Comparison helpers ───────────────────────────────────────────────────────

def compare_arrays(name, arr_orig, arr_jax, rtol=1e-6, atol=1e-10):
    """Compare two arrays, print results, return True if close."""
    arr_orig = np.asarray(arr_orig)
    arr_jax = np.asarray(arr_jax)

    if arr_orig.shape != arr_jax.shape:
        print(f"  FAIL {name}: shape mismatch {arr_orig.shape} vs {arr_jax.shape}")
        return False

    if arr_orig.size == 0:
        print(f"  PASS {name}: both empty")
        return True

    max_abs_diff = np.max(np.abs(arr_orig - arr_jax))
    denom = np.maximum(np.abs(arr_orig), 1e-30)
    max_rel_diff = np.max(np.abs(arr_orig - arr_jax) / denom)

    passed = np.allclose(arr_orig, arr_jax, rtol=rtol, atol=atol)
    status = "PASS" if passed else "FAIL"

    print(f"  {status} {name}: max_abs_diff={max_abs_diff:.3e}, max_rel_diff={max_rel_diff:.3e}")

    if not passed:
        idx = np.unravel_index(np.argmax(np.abs(arr_orig - arr_jax)), arr_orig.shape)
        print(f"        worst at index {idx}: orig={arr_orig[idx]:.10e}, jax={arr_jax[idx]:.10e}")

    return passed


def compare_scalars(name, val_orig, val_jax, rtol=1e-6):
    """Compare two scalars."""
    val_orig = float(np.asarray(val_orig).ravel()[0])
    val_jax = float(np.asarray(val_jax).ravel()[0])

    abs_diff = abs(val_orig - val_jax)
    rel_diff = abs_diff / max(abs(val_orig), 1e-30)

    passed = rel_diff < rtol
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: orig={val_orig:.10e}, jax={val_jax:.10e}, rel_diff={rel_diff:.3e}")
    return passed


# ── Main test ────────────────────────────────────────────────────────────────

def main():

    skip_bc = "--skip-bc" in sys.argv
    skip_stacked = "--skip-stacked" in sys.argv

    print("=" * 80)
    print("COMPARISON TEST: cosmocnc vs cosmocnc_jax")
    print("=" * 80)

    # Import both packages
    print("\nImporting packages...")
    import cosmocnc
    import cosmocnc_jax

    all_passed = True
    results = {}
    timings = {}

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: HMF + Cluster Abundance + Binned/Unbinned-from-abundance
    # Single initialisation with observables=[["q_so_sim"]]
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("1. INITIALISATION + HMF + CLUSTER ABUNDANCE")
    print("─" * 60)

    cnc_params, scal_rel, cosmo = get_params(
        observables=[["q_so_sim"]],
        data_lik_from_abundance=True,
        likelihood_type="unbinned",
    )

    print("  Initialising cosmocnc...")
    t0 = time.time()
    nc_orig = setup_and_init(cosmocnc, dict(cnc_params), dict(scal_rel), dict(cosmo))
    timings["init_orig"] = time.time() - t0
    print(f"    cosmocnc init: {timings['init_orig']:.2f}s")

    print("  Initialising cosmocnc_jax...")
    t0 = time.time()
    nc_jax = setup_and_init(cosmocnc_jax, dict(cnc_params), dict(scal_rel), dict(cosmo))
    timings["init_jax"] = time.time() - t0
    print(f"    cosmocnc_jax init: {timings['init_jax']:.2f}s")

    # ── Number counts ──

    print("\n  Computing number counts (cosmocnc)...")
    t0 = time.time()
    nc_orig.get_number_counts()
    timings["nc_orig"] = time.time() - t0
    print(f"    Time: {timings['nc_orig']:.2f}s")

    print("  Computing number counts (cosmocnc_jax)...")
    t0 = time.time()
    nc_jax.get_number_counts()
    timings["nc_jax"] = time.time() - t0
    print(f"    Time: {timings['nc_jax']:.2f}s")

    # Compare HMF matrix
    p = compare_arrays("hmf_matrix", nc_orig.hmf_matrix, nc_jax.hmf_matrix, rtol=1e-4)
    all_passed &= p; results["hmf_matrix"] = p

    # Compare n_tot
    p = compare_scalars("n_tot", nc_orig.n_tot, nc_jax.n_tot, rtol=1e-4)
    all_passed &= p; results["n_tot"] = p

    # Compare dn/dz
    p = compare_arrays("dn_dz (n_z)", nc_orig.n_z, nc_jax.n_z, rtol=1e-4)
    all_passed &= p; results["dn_dz"] = p

    # Compare dn/dSNR
    p = compare_arrays("dn_dSNR (n_obs)", nc_orig.n_obs, nc_jax.n_obs, rtol=1e-4)
    all_passed &= p; results["dn_dSNR"] = p

    # Compare abundance matrix (use generous atol since small bins are near zero)
    if nc_orig.abundance_matrix is not None and nc_jax.abundance_matrix is not None:
        p = compare_arrays("abundance_matrix", nc_orig.abundance_matrix, nc_jax.abundance_matrix, rtol=1e-3, atol=1e-6)
        all_passed &= p; results["abundance_matrix"] = p

    # Compare cosmological quantities
    p = compare_arrays("D_A", nc_orig.D_A, nc_jax.D_A, rtol=1e-6)
    all_passed &= p; results["D_A"] = p

    p = compare_arrays("E_z", nc_orig.E_z, nc_jax.E_z, rtol=1e-6)
    all_passed &= p; results["E_z"] = p

    # ── Binned likelihood (reuse same number counts) ──

    print("\n" + "─" * 60)
    print("2. BINNED LIKELIHOOD")
    print("─" * 60)

    nc_orig.cnc_params["likelihood_type"] = "binned"
    nc_jax.cnc_params["likelihood_type"] = "binned"

    t0 = time.time()
    log_lik_binned_orig = nc_orig.get_log_lik()
    timings["binned_orig"] = time.time() - t0

    t0 = time.time()
    log_lik_binned_jax = nc_jax.get_log_lik()
    timings["binned_jax"] = time.time() - t0

    p = compare_scalars("log_lik_binned", log_lik_binned_orig, log_lik_binned_jax, rtol=1e-4)
    all_passed &= p; results["log_lik_binned"] = p

    p = compare_arrays("n_binned_theory", nc_orig.n_binned, nc_jax.n_binned, rtol=1e-3)
    all_passed &= p; results["n_binned"] = p

    print(f"  Timing: cosmocnc={timings['binned_orig']:.4f}s, cosmocnc_jax={timings['binned_jax']:.4f}s")

    # ── Unbinned likelihood (data_lik_from_abundance) ──

    print("\n" + "─" * 60)
    print("3. UNBINNED LIKELIHOOD (data_lik_from_abundance=True)")
    print("─" * 60)

    nc_orig.cnc_params["likelihood_type"] = "unbinned"
    nc_jax.cnc_params["likelihood_type"] = "unbinned"

    t0 = time.time()
    log_lik_unbinned_fa_orig = nc_orig.get_log_lik()
    timings["unbinned_fa_orig"] = time.time() - t0

    t0 = time.time()
    log_lik_unbinned_fa_jax = nc_jax.get_log_lik()
    timings["unbinned_fa_jax"] = time.time() - t0

    p = compare_scalars("log_lik_unbinned_from_abundance", log_lik_unbinned_fa_orig, log_lik_unbinned_fa_jax, rtol=1e-4)
    all_passed &= p; results["log_lik_unbinned_from_abundance"] = p

    print(f"  Timing: cosmocnc={timings['unbinned_fa_orig']:.4f}s, cosmocnc_jax={timings['unbinned_fa_jax']:.4f}s")

    # ── Extreme value statistics (reuse same number counts) ──

    print("\n" + "─" * 60)
    print("4. EXTREME VALUE STATISTICS")
    print("─" * 60)

    nc_orig.get_log_lik_extreme_value()
    nc_orig.eval_extreme_value_quantities()

    nc_jax.get_log_lik_extreme_value()
    nc_jax.eval_extreme_value_quantities()

    p = compare_scalars("obs_select_max_mean", nc_orig.obs_select_max_mean, nc_jax.obs_select_max_mean, rtol=1e-4)
    all_passed &= p; results["obs_select_max_mean"] = p

    p = compare_scalars("obs_select_max_std", nc_orig.obs_select_max_std, nc_jax.obs_select_max_std, rtol=1e-4)
    all_passed &= p; results["obs_select_max_std"] = p

    # ── Goodness of fit (C statistic, reuse same counts) ──

    print("\n" + "─" * 60)
    print("5. GOODNESS OF FIT (C statistic)")
    print("─" * 60)

    C_orig, C_mean_orig, C_std_orig = nc_orig.get_c_statistic()
    C_jax, C_mean_jax, C_std_jax = nc_jax.get_c_statistic()

    p = compare_scalars("C_observed", C_orig, C_jax, rtol=2e-3)
    all_passed &= p; results["C_observed"] = p

    p = compare_scalars("C_mean", C_mean_orig, C_mean_jax, rtol=1e-4)
    all_passed &= p; results["C_mean"] = p

    p = compare_scalars("C_std", C_std_orig, C_std_jax, rtol=1e-4)
    all_passed &= p; results["C_std"] = p

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Backward convolutional (requires observables=[q, p])
    # ══════════════════════════════════════════════════════════════════════

    if not skip_bc:
        print("\n" + "─" * 60)
        print("6. UNBINNED LIKELIHOOD (backward convolutional)")
        print("─" * 60)

        cnc_params_bc, scal_rel_bc, cosmo_bc = get_params(
            observables=[["q_so_sim"], ["p_so_sim"]],
            data_lik_from_abundance=False,
            likelihood_type="unbinned",
        )

        print("  Initialising for backward convolutional...")
        nc_orig_bc = setup_and_init(cosmocnc, dict(cnc_params_bc), dict(scal_rel_bc), dict(cosmo_bc))
        nc_jax_bc = setup_and_init(cosmocnc_jax, dict(cnc_params_bc), dict(scal_rel_bc), dict(cosmo_bc))

        print("  Computing number counts...")
        nc_orig_bc.get_number_counts()
        nc_jax_bc.get_number_counts()

        t0 = time.time()
        log_lik_bc_orig = nc_orig_bc.get_log_lik()
        timings["bc_orig"] = time.time() - t0

        t0 = time.time()
        log_lik_bc_jax = nc_jax_bc.get_log_lik()
        timings["bc_jax"] = time.time() - t0

        p = compare_scalars("log_lik_backward_conv", log_lik_bc_orig, log_lik_bc_jax, rtol=1e-4)
        all_passed &= p; results["log_lik_backward_conv"] = p

        print(f"  Timing: cosmocnc={timings['bc_orig']:.4f}s, cosmocnc_jax={timings['bc_jax']:.4f}s")
    else:
        print("\n  [SKIPPED] Backward convolutional (use --skip-bc to skip)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Stacked likelihood (requires separate init)
    # ══════════════════════════════════════════════════════════════════════

    if not skip_stacked:
        print("\n" + "─" * 60)
        print("7. STACKED LIKELIHOOD")
        print("─" * 60)

        cnc_params_st, scal_rel_st, cosmo_st = get_params(
            observables=[["q_so_sim"]],
            data_lik_from_abundance=False,
            stacked_likelihood=True,
            likelihood_type="unbinned",
        )

        print("  Initialising for stacked likelihood...")
        nc_orig_st = setup_and_init(cosmocnc, dict(cnc_params_st), dict(scal_rel_st), dict(cosmo_st))
        nc_jax_st = setup_and_init(cosmocnc_jax, dict(cnc_params_st), dict(scal_rel_st), dict(cosmo_st))

        print("  Computing number counts...")
        nc_orig_st.get_number_counts()
        nc_jax_st.get_number_counts()

        t0 = time.time()
        log_lik_stacked_orig = nc_orig_st.get_log_lik()
        timings["stacked_orig"] = time.time() - t0

        t0 = time.time()
        log_lik_stacked_jax = nc_jax_st.get_log_lik()
        timings["stacked_jax"] = time.time() - t0

        p = compare_scalars("log_lik_stacked", log_lik_stacked_orig, log_lik_stacked_jax, rtol=1e-4)
        all_passed &= p; results["log_lik_stacked"] = p

        if hasattr(nc_orig_st, 'stacked_model') and hasattr(nc_jax_st, 'stacked_model'):
            for key in nc_orig_st.stacked_model:
                if key in nc_jax_st.stacked_model:
                    p = compare_scalars(
                        f"stacked_model[{key}]",
                        nc_orig_st.stacked_model[key],
                        nc_jax_st.stacked_model[key],
                        rtol=1e-4
                    )
                    all_passed &= p; results[f"stacked_model_{key}"] = p

        print(f"  Timing: cosmocnc={timings['stacked_orig']:.4f}s, cosmocnc_jax={timings['stacked_jax']:.4f}s")
    else:
        print("\n  [SKIPPED] Stacked likelihood (use --skip-stacked to skip)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Repeated likelihood evaluations (timing comparison)
    # Uses update_params to vary sigma_8 without re-initialising cosmology
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 60)
    print("8. REPEATED LIKELIHOOD EVALUATIONS (timing comparison)")
    print("─" * 60)

    n_evals = 5
    scal_rel_params_orig = dict(nc_orig.scal_rel_params)
    scal_rel_params_jax = dict(nc_jax.scal_rel_params)

    sigma_8_vec = np.linspace(0.808, 0.815, n_evals)
    log_lik_orig_arr = np.zeros(n_evals)
    log_lik_jax_arr = np.zeros(n_evals)

    # Restore unbinned likelihood type from phase 1
    nc_orig.cnc_params["likelihood_type"] = "unbinned"
    nc_jax.cnc_params["likelihood_type"] = "unbinned"

    print(f"  Running {n_evals} evaluations with cosmocnc...")
    t0 = time.time()
    for i in range(n_evals):
        cosmo_params_i = dict(nc_orig.cosmo_params)
        cosmo_params_i["sigma_8"] = sigma_8_vec[i]
        nc_orig.update_params(cosmo_params_i, scal_rel_params_orig)
        log_lik_orig_arr[i] = nc_orig.get_log_lik()
    timings["scan_orig"] = time.time() - t0

    print(f"  Running {n_evals} evaluations with cosmocnc_jax...")
    t0 = time.time()
    for i in range(n_evals):
        cosmo_params_i = dict(nc_jax.cosmo_params)
        cosmo_params_i["sigma_8"] = sigma_8_vec[i]
        nc_jax.update_params(cosmo_params_i, scal_rel_params_jax)
        log_lik_jax_arr[i] = nc_jax.get_log_lik()
    timings["scan_jax"] = time.time() - t0

    p = compare_arrays("log_lik_scan", log_lik_orig_arr, log_lik_jax_arr, rtol=1e-4)
    all_passed &= p; results["log_lik_scan"] = p

    lik_orig = np.exp(log_lik_orig_arr - np.max(log_lik_orig_arr))
    lik_jax = np.exp(log_lik_jax_arr - np.max(log_lik_jax_arr))
    p = compare_arrays("lik_curve_normalised", lik_orig, lik_jax, rtol=1e-4)
    all_passed &= p; results["lik_curve"] = p

    t_per_orig = timings["scan_orig"] / n_evals
    t_per_jax = timings["scan_jax"] / n_evals

    print(f"\n  Timing summary ({n_evals} evaluations):")
    print(f"    cosmocnc:     {timings['scan_orig']:.3f}s total, {t_per_orig:.3f}s/eval")
    print(f"    cosmocnc_jax: {timings['scan_jax']:.3f}s total, {t_per_jax:.3f}s/eval")
    print(f"    Speedup: {timings['scan_orig']/max(timings['scan_jax'],1e-10):.2f}x")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n_pass = sum(results.values())
    n_total = len(results)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {n_pass}/{n_total} tests passed")

    print("\n  Timing comparison:")
    print(f"    Init:             cosmocnc={timings['init_orig']:.2f}s, cosmocnc_jax={timings['init_jax']:.2f}s")
    print(f"    Number counts:    cosmocnc={timings['nc_orig']:.2f}s, cosmocnc_jax={timings['nc_jax']:.2f}s")
    print(f"    Binned lik:       cosmocnc={timings['binned_orig']:.4f}s, cosmocnc_jax={timings['binned_jax']:.4f}s")
    print(f"    Unbinned (abund): cosmocnc={timings['unbinned_fa_orig']:.4f}s, cosmocnc_jax={timings['unbinned_fa_jax']:.4f}s")
    if "bc_orig" in timings:
        print(f"    Unbinned (BC):    cosmocnc={timings['bc_orig']:.4f}s, cosmocnc_jax={timings['bc_jax']:.4f}s")
    if "stacked_orig" in timings:
        print(f"    Stacked:          cosmocnc={timings['stacked_orig']:.4f}s, cosmocnc_jax={timings['stacked_jax']:.4f}s")
    print(f"    Lik scan ({n_evals}x):  cosmocnc={timings['scan_orig']:.3f}s, cosmocnc_jax={timings['scan_jax']:.3f}s")

    if all_passed:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
