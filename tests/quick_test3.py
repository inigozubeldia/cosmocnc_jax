"""Profiling: per-stage timing of backward conv."""
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_MAX_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np, time, sys
sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
import cosmocnc_jax

SHARED_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim"], ["p_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False, "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16, "n_points": 16384,
    "n_points_data_lik": 2048, "sigma_mass_prior": 10,
    "downsample_hmf_bc": 2, "delta_m_with_ref": True,
    "scalrel_type_deriv": "numerical",
    "cosmo_param_density": "critical", "cosmo_model": "lcdm",
    "hmf_calc": "cnc", "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params = dict(nc.cnc_params)
nc.scal_rel_params = dict(nc.scal_rel_params)
nc.cosmo_params = dict(nc.cosmo_params)
nc.cnc_params.update(SHARED_PARAMS)
nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
nc.scal_rel_params.update({"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.})
nc.initialise()

# Full warmup
print("Warmup (JIT compilation)...")
nc.get_hmf()
nc.get_cluster_abundance()
nc.get_number_counts()
ll = nc.get_log_lik()
jax.block_until_ready(ll)
print(f"  ll={float(np.asarray(ll)):.4f}")

# Subsequent evals
for i in range(5):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = 0.805 + 0.003 * i
    nc.update_params(cp, dict(nc.scal_rel_params))

    t0 = time.time()
    nc.get_hmf()
    jax.block_until_ready(nc.hmf_matrix)
    t1 = time.time()
    nc.get_cluster_abundance()
    jax.block_until_ready(nc.abundance_tensor)
    t2 = time.time()
    nc.get_number_counts()
    ll = nc.get_log_lik()
    jax.block_until_ready(ll)
    t3 = time.time()

    print(f"\n[{i}] hmf={t1-t0:.4f}  abund={t2-t1:.4f}  loglik={t3-t2:.4f}  total={t3-t0:.4f}  ll={float(np.asarray(ll)):.4f}")
