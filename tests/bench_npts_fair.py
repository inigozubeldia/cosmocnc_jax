"""Fair comparison: both packages use classy_sz (same cosmology).

Tests n_points_data_lik = 256, 512, 1024, 2048 with both numpy and jax
using the SAME cosmology tool so differences are purely from the backward conv.
Also includes timing for both.
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

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
import cosmocnc
import cosmocnc_jax

SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}
BASE_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16, "n_points": 16384,
    "sigma_mass_prior": 10, "downsample_hmf_bc": 2,
    "delta_m_with_ref": True, "scalrel_type_deriv": "numerical",
    "cosmo_param_density": "critical", "cosmo_model": "lcdm",
    "hmf_calc": "cnc", "interp_tinker": "linear",
    "likelihood_type": "unbinned",
    "observables": [["q_so_sim"], ["p_so_sim"]],
    "data_lik_from_abundance": False,
    "stacked_likelihood": False,
    "cosmology_tool": "classy_sz",  # SAME for both
}

n_eval = 5
sigma_8_vec = np.linspace(0.805, 0.815, n_eval)

print("=" * 90)
print("FAIR COMPARISON: both use classy_sz cosmology")
print("=" * 90)

for n_pts_dl in [256, 512, 1024, 2048]:
    print(f"\n{'─' * 90}")
    print(f"n_points_data_lik = {n_pts_dl}")
    print(f"{'─' * 90}")

    params = dict(BASE_PARAMS)
    params["n_points_data_lik"] = n_pts_dl

    # ── NumPy ──
    print(f"  Init cosmocnc (numpy)...")
    nc_np = cosmocnc.cluster_number_counts()
    nc_np.cnc_params.update(dict(params))
    nc_np.scal_rel_params.update(dict(SCAL_REL))
    nc_np.initialise()
    nc_np.get_number_counts()

    ll_np = nc_np.get_log_lik()
    print(f"    ll = {float(ll_np):.6f}")

    ll_np_arr = np.zeros(n_eval)
    t0 = time.time()
    for i in range(n_eval):
        cp = dict(nc_np.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc_np.update_params(cp, dict(nc_np.scal_rel_params))
        ll_np_arr[i] = nc_np.get_log_lik()
    t_np = (time.time() - t0) / n_eval

    # ── JAX ──
    print(f"  Init cosmocnc_jax...")
    nc_jax = cosmocnc_jax.cluster_number_counts()
    nc_jax.cnc_params.update(dict(params))
    nc_jax.scal_rel_params.update(dict(SCAL_REL))
    nc_jax.initialise()
    nc_jax.get_number_counts()

    # Warmup
    ll_jax = nc_jax.get_log_lik()
    jax.block_until_ready(ll_jax)
    print(f"    ll = {float(np.asarray(ll_jax)):.6f}")

    ll_jax_arr = np.zeros(n_eval)
    t0 = time.time()
    for i in range(n_eval):
        cp = dict(nc_jax.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc_jax.update_params(cp, dict(nc_jax.scal_rel_params))
        ll_tmp = nc_jax.get_log_lik()
        jax.block_until_ready(ll_tmp)
        ll_jax_arr[i] = float(np.asarray(ll_tmp))
    t_jax = (time.time() - t0) / n_eval

    # ── Comparison ──
    # log_lik agreement
    ll_max_rel = np.max(np.abs(ll_np_arr - ll_jax_arr) / np.maximum(np.abs(ll_np_arr), 1e-30))
    ll_max_abs = np.max(np.abs(ll_np_arr - ll_jax_arr))

    # Likelihood curve shape
    lik_np = np.exp(ll_np_arr - np.max(ll_np_arr))
    lik_jax = np.exp(ll_jax_arr - np.max(ll_jax_arr))
    lik_max_rel = np.max(np.abs(lik_np - lik_jax) / np.maximum(lik_np, 1e-30))

    print(f"\n  Results:")
    print(f"    log_lik max_rel = {ll_max_rel:.2e}  max_abs = {ll_max_abs:.4f}")
    print(f"    lik_curve max_rel = {lik_max_rel:.2e}")
    print(f"    numpy: {t_np*1000:.0f}ms/eval   jax: {t_jax*1000:.0f}ms/eval   speedup: {t_np/max(t_jax,1e-10):.1f}x")

    # Per-evaluation detail
    print(f"\n    {'sigma_8':>8s}  {'numpy':>14s}  {'jax':>14s}  {'abs_diff':>10s}  {'rel_diff':>10s}")
    for i in range(n_eval):
        ad = abs(ll_np_arr[i] - ll_jax_arr[i])
        rd = ad / max(abs(ll_np_arr[i]), 1e-30)
        print(f"    {sigma_8_vec[i]:8.3f}  {ll_np_arr[i]:14.4f}  {ll_jax_arr[i]:14.4f}  {ad:10.4f}  {rd:10.2e}")

# ── Also run JAX with classy_sz_jax for speed reference ──
print(f"\n{'═' * 90}")
print("SPEED REFERENCE: JAX with classy_sz_jax (emulators, fastest)")
print(f"{'═' * 90}")

for n_pts_dl in [256, 512, 1024, 2048]:
    params = dict(BASE_PARAMS)
    params["n_points_data_lik"] = n_pts_dl
    params["cosmology_tool"] = "classy_sz_jax"

    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params.update(dict(params))
    nc.scal_rel_params.update(dict(SCAL_REL))
    nc.initialise()
    nc.get_number_counts()

    # Warmup
    ll = nc.get_log_lik()
    jax.block_until_ready(ll)

    t0 = time.time()
    for i in range(n_eval):
        cp = dict(nc.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc.update_params(cp, dict(nc.scal_rel_params))
        ll = nc.get_log_lik()
        jax.block_until_ready(ll)
    t_per = (time.time() - t0) / n_eval

    print(f"  n_pts_dl={n_pts_dl:5d}:  {t_per*1000:.1f}ms/eval  ll={float(np.asarray(ll)):.4f}")
