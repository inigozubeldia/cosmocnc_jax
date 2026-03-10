"""Profile per-MCMC-step timing breakdown for classy_sz_jax."""
import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
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

cnc_params_update = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": True,
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16, "n_points": 16384,
    "n_points_data_lik": 2048,
    "sigma_mass_prior": 10,
    "downsample_hmf_bc": 2,
    "delta_m_with_ref": True,
    "scalrel_type_deriv": "numerical",
    "cosmology_tool": "classy_sz_jax",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm", "hmf_calc": "cnc",
    "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}

scal_rel_update = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params = dict(nc.cnc_params)
nc.scal_rel_params = dict(nc.scal_rel_params)
nc.cosmo_params = dict(nc.cosmo_params)
nc.cnc_params.update(cnc_params_update)
nc.scal_rel_params.update(scal_rel_update)
nc.initialise()

# First call (includes JIT compilation)
print("=== First call (warmup) ===")
t0 = time.time()
nc.get_number_counts()
t1 = time.time()
ll = nc.get_log_lik()
t2 = time.time()
print(f"  get_number_counts: {t1-t0:.3f}s")
print(f"  get_log_lik: {t2-t1:.3f}s")
print(f"  total: {t2-t0:.3f}s")

# Now profile MCMC-like steps
print("\n=== Per-step profiling (10 evals) ===")
sigma_8_vec = np.linspace(0.808, 0.815, 10)
scal = dict(nc.scal_rel_params)

times_update = []
times_hmf = []
times_abundance = []
times_data_lik = []
times_total = []

for i in range(10):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = sigma_8_vec[i]

    t0 = time.time()
    nc.update_params(cp, scal)
    t1 = time.time()

    # HMF
    nc.get_hmf()
    t2 = time.time()

    # Background quantities are recomputed in get_hmf
    # Abundance
    nc.get_cluster_abundance()
    nc.get_number_counts()
    t3 = time.time()

    # Log lik
    ll = nc.get_log_lik()
    t4 = time.time()

    times_update.append(t1 - t0)
    times_hmf.append(t2 - t1)
    times_abundance.append(t3 - t2)
    times_data_lik.append(t4 - t3)
    times_total.append(t4 - t0)

    if i < 3 or i == 9:
        print(f"  [{i}] update={t1-t0:.4f}s hmf={t2-t1:.4f}s abundance={t3-t2:.4f}s lik={t4-t3:.4f}s total={t4-t0:.4f}s ll={float(ll):.2f}")

# Summary (skip first 2 warmup)
print("\n=== Average (last 8 evals) ===")
for name, arr in [("update_params", times_update), ("get_hmf", times_hmf),
                   ("get_abundance", times_abundance), ("get_log_lik", times_data_lik),
                   ("TOTAL", times_total)]:
    avg = np.mean(arr[2:])
    print(f"  {name:20s}: {avg:.4f}s")
