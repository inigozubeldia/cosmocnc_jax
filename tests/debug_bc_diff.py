"""Debug: find the source of numpy vs jax backward conv discrepancy.

Compares per-cluster log_lik values between numpy and jax with SAME cosmology.
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
import jax.numpy as jnp

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
    "cosmology_tool": "classy_sz",
    "n_points_data_lik": 256,
}

print("Init numpy...")
nc_np = cosmocnc.cluster_number_counts()
nc_np.cnc_params.update(dict(BASE_PARAMS))
nc_np.scal_rel_params.update(dict(SCAL_REL))
nc_np.initialise()
nc_np.get_number_counts()
ll_np = nc_np.get_log_lik()
print(f"  numpy ll = {float(ll_np):.6f}")

print("\nInit jax...")
nc_jax = cosmocnc_jax.cluster_number_counts()
nc_jax.cnc_params.update(dict(BASE_PARAMS))
nc_jax.scal_rel_params.update(dict(SCAL_REL))
nc_jax.initialise()
nc_jax.get_number_counts()
ll_jax = nc_jax.get_log_lik()
jax.block_until_ready(ll_jax)
print(f"  jax ll = {float(np.asarray(ll_jax)):.6f}")

print(f"\n  diff = {float(ll_np) - float(np.asarray(ll_jax)):.6f}")

# ── Compare per-cluster log-liks ──
# NumPy stores per-cluster cpdf_product_with_hmf in return_dict
# JAX stores bc_cpdf_array (cpdf_with_hmf) and bc_lnM_array

# Get JAX per-cluster log_liks
from cosmocnc_jax.utils import simpson
jax_cwh = np.asarray(nc_jax.bc_cpdf_array)  # (n_bc, n_pts)
jax_lnM = np.asarray(nc_jax.bc_lnM_array)   # (n_bc, n_pts)
jax_idx = nc_jax.bc_cluster_indices          # cluster indices

n_bc = len(jax_idx)
jax_log_liks = np.zeros(n_bc)
for i in range(n_bc):
    integral = np.trapezoid(jax_cwh[i], jax_lnM[i])
    jax_log_liks[i] = np.log(max(integral, 1e-300))

# NumPy stores per-cluster data in self
# Let's access the NumPy per-cluster cpdfs
np_cwh = {}
np_lnM_arr = {}
# The numpy code stores in return_dict["cpdf_" + str(cluster_index) + "_0"]
# and integrates with simpson. Let me access the stored data.
# Actually numpy stores self.bc_cpdf_array etc too if we check...

# Let me check what numpy stores
has_bc_cpdf = hasattr(nc_np, 'bc_cpdf_array')
print(f"\nnumpy has bc_cpdf_array: {has_bc_cpdf}")

# Actually the numpy code stores cpdf_product_with_hmf_list and lnM_list
has_lnM_list = hasattr(nc_np, 'lnM_list')
print(f"numpy has lnM_list: {has_lnM_list}")

# Let's look at what attributes are stored
bc_attrs = [a for a in dir(nc_np) if 'cpdf' in a.lower() or 'bc_' in a.lower() or 'lnM' in a.lower()]
print(f"numpy bc-related attrs: {bc_attrs}")

# Check indices
np_indices = nc_np.indices_bc if hasattr(nc_np, 'indices_bc') else None
print(f"numpy indices_bc: {np_indices is not None}, n={len(np_indices) if np_indices is not None else 0}")

# Compare N_tot (should be same since abundance doesn't depend on n_pts_dl)
print(f"\nn_tot: numpy={nc_np.n_tot:.6f}  jax={float(np.asarray(nc_jax.n_tot)):.6f}")
print(f"  diff={nc_np.n_tot - float(np.asarray(nc_jax.n_tot)):.6f}")

# Compare cosmological quantities at cluster redshifts
z_test = np.array([0.1, 0.5, 1.0, 2.0])
print("\nCosmological quantities at test redshifts:")
for z in z_test:
    da_np = np.interp(z, nc_np.redshift_vec, nc_np.D_A)
    da_jax = float(np.interp(z, np.asarray(nc_jax.redshift_vec), np.asarray(nc_jax.D_A)))
    ez_np = np.interp(z, nc_np.redshift_vec, nc_np.E_z)
    ez_jax = float(np.interp(z, np.asarray(nc_jax.redshift_vec), np.asarray(nc_jax.E_z)))
    print(f"  z={z:.1f}: D_A diff={abs(da_np-da_jax)/da_np:.2e}  E_z diff={abs(ez_np-ez_jax)/ez_np:.2e}")

# Compare HMF at test redshifts
print("\nHMF at z-index 50:")
hmf_np = nc_np.hmf_matrix[50]
hmf_jax = np.asarray(nc_jax.hmf_matrix)[50]
mask = hmf_np > 1e-30
if np.sum(mask) > 0:
    rel = np.abs(hmf_np[mask] - hmf_jax[mask]) / hmf_np[mask]
    print(f"  max_rel={rel.max():.2e}  mean_rel={rel.mean():.2e}")

# Compare lnM grids for a few clusters
print(f"\nComparing lnM grids (JAX n_pts_dl={nc_jax.cnc_params['n_points_data_lik']}):")
# JAX lnM_min, lnM_max per cluster
jax_lnM_min = jax_lnM[:, 0]
jax_lnM_max = jax_lnM[:, -1]
print(f"  JAX lnM range: [{jax_lnM_min.min():.4f}, {jax_lnM_max.max():.4f}]")
print(f"  JAX lnM range mean: [{jax_lnM_min.mean():.4f}, {jax_lnM_max.mean():.4f}]")

# Check the mass range differences
# The numpy code uses n_points_data_lik for coarse mass range
# The jax code uses _N_COARSE_MASS=128
from cosmocnc_jax.cnc import _N_COARSE_MASS
print(f"\n  JAX _N_COARSE_MASS = {_N_COARSE_MASS}")
print(f"  NumPy uses n_points_data_lik={nc_np.cnc_params['n_points_data_lik']} for mass range coarse grid")
print(f"  This means: numpy uses {nc_np.cnc_params['n_points_data_lik']} pts, jax uses {_N_COARSE_MASS} pts for argmin")

# Quick check: does the difference scale with n_clusters?
# The diff is ~6 for ~16k clusters = ~0.0004 per cluster
print(f"\n  Total log_lik diff = {float(ll_np) - float(np.asarray(ll_jax)):.4f}")
print(f"  Per-cluster avg diff = {(float(ll_np) - float(np.asarray(ll_jax))) / n_bc:.6f}")
