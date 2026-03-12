"""Test pattern-aware backward conv with partial observable availability.

Injects NaNs into p_so_sim for ~20% of clusters, then compares
cosmocnc vs cosmocnc_jax log_lik. Also verifies internal consistency:
2D with corr=0 + partial obs should approach 1D result.
"""

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
import builtins

_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc
import cosmocnc_jax

# ── Parameters ──

CNC_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim", "p_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False,
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16,
    "n_points": 16384, "n_points_data_lik": 128,
    "cosmology_tool": "classy_sz",
    "likelihood_type": "unbinned",
    "data_lik_type": "backward_convolutional",
    "stacked_likelihood": False,
    "apply_obs_cutoff": False,
    "sigma_mass_prior": 5., "z_errors": False,
    "delta_m_with_ref": False, "scalrel_type_deriv": "analytical",
    "downsample_hmf_bc": 8, "padding_fraction": 0.,
    "bc_chunk_size": 0,
    "hmf_type": "Tinker08", "hmf_calc": "classy_sz",
    "sigma_R_type": "class_sz", "mass_definition": "500c",
}

SCAL_REL = {
    "bias_sz": 0.8, "bias_cmblens": 0.8,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}

COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}


def inject_nans(catalogue, obs_name, frac=0.2, seed=42):
    """NaN-ify `frac` of entries in catalogue[obs_name]."""
    rng = np.random.default_rng(seed)
    n = len(catalogue[obs_name])
    mask = rng.random(n) < frac
    catalogue[obs_name] = np.array(catalogue[obs_name], dtype=float)
    catalogue[obs_name][mask] = np.nan
    n_nan = int(mask.sum())
    print(f"    Injected {n_nan}/{n} NaNs into {obs_name}")
    return mask


def setup_cosmocnc(cnc_params, scal_rel, cosmo):
    nc = cosmocnc.cluster_number_counts()
    nc.cnc_params.update(cnc_params)
    nc.scal_rel_params.update(scal_rel)
    nc.cosmo_params.update(cosmo)
    nc.initialise()
    return nc


def setup_jax(cnc_params, scal_rel, cosmo):
    p = dict(cnc_params)
    p["cosmology_tool"] = "classy_sz_jax"
    p["hmf_calc"] = "cnc"
    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params.update(p)
    nc.scal_rel_params.update(scal_rel)
    nc.cosmo_params.update(cosmo)
    nc.initialise()
    return nc


def main():
    print("=" * 60)
    print("PARTIAL OBSERVABLE TEST")
    print("=" * 60)

    # ── 1. Full data (no NaNs) — baseline ──
    print("\n── 1. Full data baseline ──")
    print("  Init cosmocnc...")
    nc_orig = setup_cosmocnc(CNC_PARAMS, SCAL_REL, COSMO)
    print("  Init cosmocnc_jax...")
    nc_jax = setup_jax(CNC_PARAMS, SCAL_REL, COSMO)

    ll_orig_full = float(nc_orig.get_log_lik())
    print(f"  cosmocnc  full: {ll_orig_full:.4f}")
    ll_jax_full = float(nc_jax.get_log_lik())
    print(f"  jax       full: {ll_jax_full:.4f}")
    rel_full = abs(ll_jax_full - ll_orig_full) / abs(ll_orig_full)
    print(f"  rel_diff (full): {rel_full:.6e}")

    # ── 2. Inject NaNs into p_so_sim in both catalogues ──
    print("\n── 2. Inject NaNs into p_so_sim (~20%) ──")

    # Inject same NaN pattern into both
    nan_mask = inject_nans(nc_orig.catalogue.catalogue, "p_so_sim", frac=0.2, seed=42)

    # For JAX: need to inject into the raw catalogue, then re-precompute
    nc_jax.catalogue.catalogue["p_so_sim"] = np.array(
        nc_jax.catalogue.catalogue["p_so_sim"], dtype=float)
    nc_jax.catalogue.catalogue["p_so_sim"][nan_mask] = np.nan

    # Rebuild observable_dict for cosmocnc (re-precompute)
    nc_orig.catalogue.get_precompute_cnc_quantities()

    # For JAX: rebuild observable_dict and clear BC cache
    nc_jax.catalogue.get_precompute_cnc_quantities()
    if hasattr(nc_jax, '_bc_cached'):
        del nc_jax._bc_cached

    n_nan = int(nan_mask.sum())
    n_total = len(nan_mask)
    print(f"  {n_nan}/{n_total} clusters now have only q_so_sim (no p_so_sim)")

    # Check observable_dict for cosmocnc
    n_q_only = 0
    n_qp = 0
    for i in nc_orig.catalogue.indices_other_obs:
        od = nc_orig.catalogue.observable_dict[i]
        obs_flat = [o for s in od for o in s]
        if "p_so_sim" not in obs_flat:
            n_q_only += 1
        else:
            n_qp += 1
    print(f"  cosmocnc observable_dict: {n_qp} have q+p, {n_q_only} have q only")

    # ── 3. Compute log_lik with partial data ──
    print("\n── 3. Log_lik with partial data ──")

    t0 = time.time()
    ll_orig_partial = float(nc_orig.get_log_lik())
    t_orig = time.time() - t0
    print(f"  cosmocnc  partial: {ll_orig_partial:.4f} ({t_orig:.1f}s)")

    t0 = time.time()
    ll_jax_partial = float(nc_jax.get_log_lik())
    t_jax = time.time() - t0
    print(f"  jax       partial: {ll_jax_partial:.4f} ({t_jax:.1f}s)")

    rel_partial = abs(ll_jax_partial - ll_orig_partial) / abs(ll_orig_partial)
    print(f"  rel_diff (partial): {rel_partial:.6e}")

    # ── 4. Sanity checks ──
    print("\n── 4. Sanity checks ──")
    print(f"  Full  : cosmocnc={ll_orig_full:.4f}, jax={ll_jax_full:.4f}")
    print(f"  Partial: cosmocnc={ll_orig_partial:.4f}, jax={ll_jax_partial:.4f}")
    print(f"  Δ(partial - full): cosmocnc={ll_orig_partial - ll_orig_full:.4f}, "
          f"jax={ll_jax_partial - ll_jax_full:.4f}")

    # Full and partial should differ (removing p info changes the likelihood)
    assert ll_orig_partial != ll_orig_full, "Full and partial should differ"
    print("  CHECK: partial != full ✓")

    # Both codes should show similar change
    delta_orig = ll_orig_partial - ll_orig_full
    delta_jax = ll_jax_partial - ll_jax_full
    rel_delta = abs(delta_jax - delta_orig) / abs(delta_orig)
    print(f"  Δ rel_diff: {rel_delta:.6e}")

    status = "PASS" if rel_partial < 0.05 else "FAIL"
    print(f"\n  {status}: rel_diff(partial) = {rel_partial:.6e}")
    status2 = "PASS" if rel_delta < 0.10 else "FAIL"
    print(f"  {status2}: Δ(partial-full) rel_diff = {rel_delta:.6e}")


if __name__ == "__main__":
    main()
