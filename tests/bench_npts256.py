"""Benchmark: n_points_data_lik=256 vs 2048.

Compares accuracy and speed between cosmocnc (NumPy) and cosmocnc_jax (JAX)
with n_points_data_lik=256. Also shows accuracy vs n_points_data_lik=2048
reference values.
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

# ── Helpers ──
def compare(name, val_orig, val_jax, ref_orig=None, ref_jax=None):
    """Compare scalars. If ref provided, also show diff vs reference (2048)."""
    v_o = float(np.asarray(val_orig).ravel()[0])
    v_j = float(np.asarray(val_jax).ravel()[0])
    rel = abs(v_o - v_j) / max(abs(v_o), 1e-30)
    line = f"  {name:40s}: numpy={v_o:.6f}  jax={v_j:.6f}  rel_diff={rel:.2e}"
    if ref_orig is not None:
        r_o = float(np.asarray(ref_orig).ravel()[0])
        diff_256_vs_2048 = abs(v_o - r_o) / max(abs(r_o), 1e-30)
        line += f"  |256vs2048|={diff_256_vs_2048:.2e}"
    print(line)
    return rel

def compare_arrays_summary(name, arr_o, arr_j, ref_o=None):
    arr_o = np.asarray(arr_o)
    arr_j = np.asarray(arr_j)
    denom = np.maximum(np.abs(arr_o), 1e-30)
    max_rel = np.max(np.abs(arr_o - arr_j) / denom)
    mean_rel = np.mean(np.abs(arr_o - arr_j) / denom)
    line = f"  {name:40s}: max_rel={max_rel:.2e}  mean_rel={mean_rel:.2e}"
    if ref_o is not None:
        ref_o = np.asarray(ref_o)
        denom_ref = np.maximum(np.abs(ref_o), 1e-30)
        # Compare 256 numpy vs 2048 numpy
        if arr_o.shape == ref_o.shape:
            max_rel_ref = np.max(np.abs(arr_o - ref_o) / denom_ref)
            line += f"  |256vs2048|={max_rel_ref:.2e}"
    print(line)
    return max_rel

def setup(pkg, cnc_params_update, scal_rel_update):
    nc = pkg.cluster_number_counts()
    nc.cnc_params.update(cnc_params_update)
    nc.scal_rel_params.update(scal_rel_update)
    nc.initialise()
    return nc

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
}

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Reference values at n_points_data_lik=2048
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PHASE 1: Reference values at n_points_data_lik=2048")
print("=" * 80)

params_2048 = dict(BASE_PARAMS)
params_2048["n_points_data_lik"] = 2048
params_2048["observables"] = [["q_so_sim"], ["p_so_sim"]]
params_2048["data_lik_from_abundance"] = False
params_2048["stacked_likelihood"] = False

print("\n  Initialising cosmocnc (2048)...")
params_2048_np = dict(params_2048)
params_2048_np["cosmology_tool"] = "classy_sz"
nc_np_2048 = setup(cosmocnc, params_2048_np, dict(SCAL_REL))
nc_np_2048.get_number_counts()

t0 = time.time()
ll_np_2048 = nc_np_2048.get_log_lik()
t_np_2048 = time.time() - t0
print(f"  cosmocnc 2048: ll={float(ll_np_2048):.6f}  time={t_np_2048:.4f}s")

# Multiple evals for timing
t0 = time.time()
n_eval = 5
for i in range(n_eval):
    cp = dict(nc_np_2048.cosmo_params)
    cp["sigma_8"] = 0.808 + 0.002 * i
    nc_np_2048.update_params(cp, dict(nc_np_2048.scal_rel_params))
    ll_tmp = nc_np_2048.get_log_lik()
t_np_2048_scan = time.time() - t0

print(f"\n  Initialising cosmocnc_jax (2048)...")
params_2048_jax = dict(params_2048)
params_2048_jax["cosmology_tool"] = "classy_sz_jax"
nc_jax_2048 = setup(cosmocnc_jax, params_2048_jax, dict(SCAL_REL))
nc_jax_2048.get_number_counts()

# Warmup
ll_jax_2048 = nc_jax_2048.get_log_lik()
jax.block_until_ready(ll_jax_2048)

t0 = time.time()
ll_jax_2048 = nc_jax_2048.get_log_lik()
jax.block_until_ready(ll_jax_2048)
t_jax_2048 = time.time() - t0
print(f"  cosmocnc_jax 2048: ll={float(np.asarray(ll_jax_2048)):.6f}  time={t_jax_2048:.4f}s")

# Multiple evals for timing
t0 = time.time()
for i in range(n_eval):
    cp = dict(nc_jax_2048.cosmo_params)
    cp["sigma_8"] = 0.808 + 0.002 * i
    nc_jax_2048.update_params(cp, dict(nc_jax_2048.scal_rel_params))
    ll_tmp = nc_jax_2048.get_log_lik()
    jax.block_until_ready(ll_tmp)
t_jax_2048_scan = time.time() - t0

# Store reference values
ref_ll_np_2048 = float(ll_np_2048)
ref_n_tot_np_2048 = float(nc_np_2048.n_tot)
ref_n_z_np_2048 = np.asarray(nc_np_2048.n_z).copy()

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: n_points_data_lik=256
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 2: n_points_data_lik=256")
print("=" * 80)

params_256 = dict(BASE_PARAMS)
params_256["n_points_data_lik"] = 256
params_256["observables"] = [["q_so_sim"], ["p_so_sim"]]
params_256["data_lik_from_abundance"] = False
params_256["stacked_likelihood"] = False

print("\n  Initialising cosmocnc (256)...")
params_256_np = dict(params_256)
params_256_np["cosmology_tool"] = "classy_sz"
nc_np_256 = setup(cosmocnc, params_256_np, dict(SCAL_REL))
nc_np_256.get_number_counts()

t0 = time.time()
ll_np_256 = nc_np_256.get_log_lik()
t_np_256 = time.time() - t0
print(f"  cosmocnc 256: ll={float(ll_np_256):.6f}  time={t_np_256:.4f}s")

# Multiple evals for timing
ll_np_256_arr = np.zeros(n_eval)
t0 = time.time()
for i in range(n_eval):
    cp = dict(nc_np_256.cosmo_params)
    cp["sigma_8"] = 0.808 + 0.002 * i
    nc_np_256.update_params(cp, dict(nc_np_256.scal_rel_params))
    ll_np_256_arr[i] = nc_np_256.get_log_lik()
t_np_256_scan = time.time() - t0

print(f"\n  Initialising cosmocnc_jax (256)...")
params_256_jax = dict(params_256)
params_256_jax["cosmology_tool"] = "classy_sz_jax"
nc_jax_256 = setup(cosmocnc_jax, params_256_jax, dict(SCAL_REL))
nc_jax_256.get_number_counts()

# Warmup
ll_jax_256 = nc_jax_256.get_log_lik()
jax.block_until_ready(ll_jax_256)

t0 = time.time()
ll_jax_256 = nc_jax_256.get_log_lik()
jax.block_until_ready(ll_jax_256)
t_jax_256 = time.time() - t0
print(f"  cosmocnc_jax 256: ll={float(np.asarray(ll_jax_256)):.6f}  time={t_jax_256:.4f}s")

# Multiple evals for timing
ll_jax_256_arr = np.zeros(n_eval)
t0 = time.time()
for i in range(n_eval):
    cp = dict(nc_jax_256.cosmo_params)
    cp["sigma_8"] = 0.808 + 0.002 * i
    nc_jax_256.update_params(cp, dict(nc_jax_256.scal_rel_params))
    ll_tmp = nc_jax_256.get_log_lik()
    jax.block_until_ready(ll_tmp)
    ll_jax_256_arr[i] = float(np.asarray(ll_tmp))
t_jax_256_scan = time.time() - t0

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Accuracy comparison
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ACCURACY COMPARISON")
print("=" * 80)

print("\n── Backward conv log-likelihood (single eval) ──")
compare("log_lik_bc (numpy vs jax, 2048)", ll_np_2048, ll_jax_2048)
compare("log_lik_bc (numpy vs jax, 256)", ll_np_256, ll_jax_256, ref_ll_np_2048)
compare("log_lik_bc (numpy 256 vs 2048)", ll_np_256, ll_np_2048)

print("\n── N_tot (abundance, same for both n_pts_dl) ──")
compare("n_tot (numpy vs jax, 256)", nc_np_256.n_tot, nc_jax_256.n_tot, ref_n_tot_np_2048)

print("\n── HMF matrix (same for both n_pts_dl) ──")
compare_arrays_summary("hmf_matrix (numpy vs jax)", nc_np_256.hmf_matrix, nc_jax_256.hmf_matrix)

print("\n── dn/dz (same for both n_pts_dl) ──")
compare_arrays_summary("n_z (numpy vs jax)", nc_np_256.n_z, nc_jax_256.n_z, ref_n_z_np_2048)

print("\n── Likelihood scan (5 evaluations) ──")
max_rel_scan = 0
for i in range(n_eval):
    rel = abs(ll_np_256_arr[i] - ll_jax_256_arr[i]) / max(abs(ll_np_256_arr[i]), 1e-30)
    max_rel_scan = max(max_rel_scan, rel)
    sigma_8 = 0.808 + 0.002 * i
    print(f"  sigma_8={sigma_8:.3f}: numpy={ll_np_256_arr[i]:.4f}  jax={ll_jax_256_arr[i]:.4f}  rel={rel:.2e}")
print(f"  max_rel over scan: {max_rel_scan:.2e}")

# Also compare the likelihood curves (relative shape)
lik_np = np.exp(ll_np_256_arr - np.max(ll_np_256_arr))
lik_jax = np.exp(ll_jax_256_arr - np.max(ll_jax_256_arr))
lik_rel = np.max(np.abs(lik_np - lik_jax) / np.maximum(lik_np, 1e-30))
print(f"  lik_curve max_rel: {lik_rel:.2e}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Timing summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TIMING SUMMARY")
print("=" * 80)

t_per_np_2048 = t_np_2048_scan / n_eval
t_per_jax_2048 = t_jax_2048_scan / n_eval
t_per_np_256 = t_np_256_scan / n_eval
t_per_jax_256 = t_jax_256_scan / n_eval

print(f"\n  n_points_data_lik=2048:")
print(f"    cosmocnc (numpy):   {t_per_np_2048:.4f}s/eval  ({t_np_2048_scan:.3f}s total for {n_eval} evals)")
print(f"    cosmocnc_jax (GPU): {t_per_jax_2048:.4f}s/eval  ({t_jax_2048_scan:.3f}s total for {n_eval} evals)")
print(f"    Speedup: {t_per_np_2048/max(t_per_jax_2048,1e-10):.1f}x")

print(f"\n  n_points_data_lik=256:")
print(f"    cosmocnc (numpy):   {t_per_np_256:.4f}s/eval  ({t_np_256_scan:.3f}s total for {n_eval} evals)")
print(f"    cosmocnc_jax (GPU): {t_per_jax_256:.4f}s/eval  ({t_jax_256_scan:.3f}s total for {n_eval} evals)")
print(f"    Speedup: {t_per_np_256/max(t_per_jax_256,1e-10):.1f}x")

print(f"\n  Effect of reducing n_pts_dl (2048 → 256):")
print(f"    cosmocnc (numpy):   {t_per_np_2048:.4f}s → {t_per_np_256:.4f}s  ({t_per_np_2048/max(t_per_np_256,1e-10):.1f}x faster)")
print(f"    cosmocnc_jax (GPU): {t_per_jax_2048:.4f}s → {t_per_jax_256:.4f}s  ({t_per_jax_2048/max(t_per_jax_256,1e-10):.1f}x faster)")

print(f"\n  Accuracy cost of 256 vs 2048 (numpy reference):")
ll_diff = abs(float(ll_np_256) - ref_ll_np_2048) / max(abs(ref_ll_np_2048), 1e-30)
print(f"    log_lik rel_diff: {ll_diff:.2e}")
print(f"    log_lik abs_diff: {abs(float(ll_np_256) - ref_ll_np_2048):.4f}")
print(f"    log_lik values: 2048={ref_ll_np_2048:.4f}  256={float(ll_np_256):.4f}")
