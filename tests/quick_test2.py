"""Profiling test: measure time per stage in get_log_lik_data."""
import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc_jax

SHARED_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim"], ["p_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False,
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16,
    "n_points": 16384, "n_points_data_lik": 2048,
    "sigma_mass_prior": 10, "downsample_hmf_bc": 2,
    "delta_m_with_ref": True, "scalrel_type_deriv": "numerical",
    "cosmo_param_density": "critical", "cosmo_model": "lcdm",
    "hmf_calc": "cnc", "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}
SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params = dict(nc.cnc_params)
nc.scal_rel_params = dict(nc.scal_rel_params)
nc.cosmo_params = dict(nc.cosmo_params)
nc.cnc_params.update(SHARED_PARAMS)
nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
nc.scal_rel_params.update(SCAL_REL)
nc.initialise()

# Warmup
print("Warmup...")
nc.get_hmf()
nc.get_cluster_abundance()
nc.get_number_counts()
ll = nc.get_log_lik()
print(f"  ll = {float(np.asarray(ll)):.4f}")

# Change params and run again
for i_eval in range(3):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = 0.805 + 0.005 * i_eval
    scal = dict(nc.scal_rel_params)

    nc.update_params(cp, scal)
    nc.get_hmf()
    nc.get_cluster_abundance()
    nc.get_number_counts()

    # Profile get_log_lik_data step by step
    # We need to call get_log_lik which calls get_log_lik_data
    t0 = time.time()
    ll = nc.get_log_lik()
    jax.block_until_ready(ll)
    t1 = time.time()

    print(f"[{i_eval}] get_log_lik = {t1-t0:.4f}s  ll={float(np.asarray(ll)):.4f}")
    print(f"     Breakdown: abundance={nc.t_abundance:.4f}s  data_lik={nc.t_data:.4f}s")

print("\nDetailed profiling of 4th eval with manual timing...")
cp["sigma_8"] = 0.818
nc.update_params(cp, scal)

t0 = time.time()
nc.get_hmf()
jax.block_until_ready(nc.hmf_matrix)
t1 = time.time()

nc.get_cluster_abundance()
jax.block_until_ready(nc.abundance_tensor)
t2 = time.time()

nc.get_number_counts()
t3 = time.time()

ll = nc.get_log_lik()
jax.block_until_ready(ll)
t4 = time.time()

print(f"  hmf:       {t1-t0:.4f}s")
print(f"  abundance: {t2-t1:.4f}s")
print(f"  n_counts:  {t3-t2:.4f}s")
print(f"  log_lik:   {t4-t3:.4f}s (total={t4-t0:.4f}s)")
print(f"  ll = {float(np.asarray(ll)):.4f}")
