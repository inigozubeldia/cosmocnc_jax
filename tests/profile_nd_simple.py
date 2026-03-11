"""Simple profile: just call get_log_lik multiple times."""
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
import time, sys
sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
import cosmocnc_jax

print(f"JAX backend: {jax.default_backend()}")

cnc_params = {
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
    "bc_chunk_size": 1000,
    "hmf_type": "Tinker08", "hmf_calc": "classy_sz",
    "sigma_R_type": "class_sz", "mass_definition": "500c",
}
scal_rel = {
    "bias_sz": 3., "bias_cmblens": 3.,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}
cosmo = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params.update(cnc_params)
nc.scal_rel_params.update(scal_rel)
nc.cosmo_params.update(cosmo)
nc.initialise()

print("\n--- 2D correlated [[q_so_sim, p_so_sim]] ---")
for i in range(5):
    t0 = time.time()
    ll = nc.get_log_lik()
    jax.block_until_ready(ll)
    dt = time.time() - t0
    label = "JIT" if i == 0 else "run"
    print(f"  Call {i+1} ({label}): {dt:.3f}s  log_lik={float(ll):.2f}")

# Now 1D for comparison
print("\n--- 1D independent [[q_so_sim], [p_so_sim]] ---")
cnc_params_1d = dict(cnc_params)
cnc_params_1d["observables"] = [["q_so_sim"], ["p_so_sim"]]
nc_1d = cosmocnc_jax.cluster_number_counts()
nc_1d.cnc_params.update(cnc_params_1d)
nc_1d.scal_rel_params.update(scal_rel)
nc_1d.cosmo_params.update(cosmo)
nc_1d.initialise()

for i in range(5):
    t0 = time.time()
    ll = nc_1d.get_log_lik()
    jax.block_until_ready(ll)
    dt = time.time() - t0
    label = "JIT" if i == 0 else "run"
    print(f"  Call {i+1} ({label}): {dt:.3f}s  log_lik={float(ll):.2f}")
