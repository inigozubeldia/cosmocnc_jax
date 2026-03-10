"""Sweep n_points_data_lik: find the accuracy/speed sweet spot.

Compares likelihood curve shape accuracy for different n_points_data_lik values
against the 2048-point reference.
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
}

n_eval = 10
sigma_8_vec = np.linspace(0.800, 0.825, n_eval)

# ── Reference: n_points_data_lik=2048 with NumPy ──
print("Computing reference (numpy, n_pts_dl=2048)...")
params_ref = dict(BASE_PARAMS)
params_ref["n_points_data_lik"] = 2048
params_ref["cosmology_tool"] = "classy_sz"
nc_ref = cosmocnc.cluster_number_counts()
nc_ref.cnc_params.update(params_ref)
nc_ref.scal_rel_params.update(dict(SCAL_REL))
nc_ref.initialise()
nc_ref.get_number_counts()

ll_ref = np.zeros(n_eval)
for i in range(n_eval):
    cp = dict(nc_ref.cosmo_params)
    cp["sigma_8"] = sigma_8_vec[i]
    nc_ref.update_params(cp, dict(nc_ref.scal_rel_params))
    ll_ref[i] = nc_ref.get_log_lik()
lik_ref = np.exp(ll_ref - np.max(ll_ref))
print(f"  Done. ll range: [{ll_ref.min():.2f}, {ll_ref.max():.2f}]")

# ── Sweep n_points_data_lik with JAX ──
print("\n" + "=" * 80)
print(f"{'n_pts':>6s}  {'time_ms':>8s}  {'ll_max_rel':>12s}  {'lik_max_rel':>12s}  {'lik_mean_rel':>12s}  {'ll_abs_diff':>12s}")
print("=" * 80)

for n_pts_dl in [64, 128, 256, 384, 512, 768, 1024, 2048]:
    params = dict(BASE_PARAMS)
    params["n_points_data_lik"] = n_pts_dl
    params["cosmology_tool"] = "classy_sz_jax"

    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params.update(params)
    nc.scal_rel_params.update(dict(SCAL_REL))
    nc.initialise()
    nc.get_number_counts()

    # Warmup
    ll_tmp = nc.get_log_lik()
    jax.block_until_ready(ll_tmp)

    ll_arr = np.zeros(n_eval)
    t0 = time.time()
    for i in range(n_eval):
        cp = dict(nc.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc.update_params(cp, dict(nc.scal_rel_params))
        ll_tmp = nc.get_log_lik()
        jax.block_until_ready(ll_tmp)
        ll_arr[i] = float(np.asarray(ll_tmp))
    t_total = time.time() - t0
    t_per = t_total / n_eval * 1000  # ms

    # Accuracy metrics
    ll_max_rel = np.max(np.abs(ll_arr - ll_ref) / np.maximum(np.abs(ll_ref), 1e-30))
    ll_abs_diff = np.max(np.abs(ll_arr - ll_ref))

    lik_arr = np.exp(ll_arr - np.max(ll_arr))
    lik_max_rel = np.max(np.abs(lik_arr - lik_ref) / np.maximum(lik_ref, 1e-30))
    lik_mean_rel = np.mean(np.abs(lik_arr - lik_ref) / np.maximum(lik_ref, 1e-30))

    print(f"{n_pts_dl:6d}  {t_per:8.1f}  {ll_max_rel:12.2e}  {lik_max_rel:12.2e}  {lik_mean_rel:12.2e}  {ll_abs_diff:12.4f}")

# ── Also sweep with NumPy for comparison ──
print("\n" + "=" * 80)
print("NumPy reference sweep (for timing comparison)")
print(f"{'n_pts':>6s}  {'time_ms':>8s}  {'ll_max_rel':>12s}  {'lik_max_rel':>12s}")
print("=" * 80)

for n_pts_dl in [256, 512, 1024, 2048]:
    params = dict(BASE_PARAMS)
    params["n_points_data_lik"] = n_pts_dl
    params["cosmology_tool"] = "classy_sz"

    nc = cosmocnc.cluster_number_counts()
    nc.cnc_params.update(params)
    nc.scal_rel_params.update(dict(SCAL_REL))
    nc.initialise()
    nc.get_number_counts()

    ll_arr_np = np.zeros(n_eval)
    t0 = time.time()
    for i in range(n_eval):
        cp = dict(nc.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        nc.update_params(cp, dict(nc.scal_rel_params))
        ll_arr_np[i] = nc.get_log_lik()
    t_total = time.time() - t0
    t_per = t_total / n_eval * 1000

    ll_max_rel = np.max(np.abs(ll_arr_np - ll_ref) / np.maximum(np.abs(ll_ref), 1e-30))
    lik_arr_np = np.exp(ll_arr_np - np.max(ll_arr_np))
    lik_max_rel = np.max(np.abs(lik_arr_np - lik_ref) / np.maximum(lik_ref, 1e-30))

    print(f"{n_pts_dl:6d}  {t_per:8.0f}  {ll_max_rel:12.2e}  {lik_max_rel:12.2e}")
