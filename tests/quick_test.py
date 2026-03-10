"""Quick test: verify cosmocnc_jax runs correctly after generalization."""
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

    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "hmf_calc": "cnc",
    "interp_tinker": "linear",

    "stacked_likelihood": False,
    "likelihood_type": "unbinned",
}

SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}


def main():
    print("Setting up cosmocnc_jax...")
    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params = dict(nc.cnc_params)
    nc.scal_rel_params = dict(nc.scal_rel_params)
    nc.cosmo_params = dict(nc.cosmo_params)
    nc.cnc_params.update(SHARED_PARAMS)
    nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
    nc.scal_rel_params.update(SCAL_REL)
    nc.initialise()

    # First evaluation (JIT warmup)
    print("\nFirst evaluation (JIT warmup)...")
    t0 = time.time()
    nc.get_hmf()
    t1 = time.time()
    print(f"  get_hmf: {t1-t0:.3f}s")

    nc.get_cluster_abundance()
    nc.get_number_counts()
    t2 = time.time()
    print(f"  get_cluster_abundance + get_number_counts: {t2-t1:.3f}s")
    print(f"  n_tot = {float(nc.n_tot):.2f}")

    ll = nc.get_log_lik()
    t3 = time.time()
    print(f"  get_log_lik: {t3-t2:.3f}s")
    print(f"  log_lik = {float(np.asarray(ll)):.4f}")
    print(f"  Total first eval: {t3-t0:.3f}s")

    # Second evaluation (should be fast)
    print("\nSecond evaluation (varying sigma_8)...")
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = 0.811
    scal = dict(nc.scal_rel_params)

    t0 = time.time()
    nc.update_params(cp, scal)
    t1 = time.time()
    nc.get_hmf()
    t2 = time.time()
    nc.get_cluster_abundance()
    nc.get_number_counts()
    t3 = time.time()
    ll2 = nc.get_log_lik()
    t4 = time.time()

    print(f"  update_params: {t1-t0:.4f}s")
    print(f"  get_hmf: {t2-t1:.4f}s")
    print(f"  get_cluster_abundance: {t3-t2:.4f}s")
    print(f"  get_log_lik: {t4-t3:.4f}s")
    print(f"  Total: {t4-t0:.4f}s")
    print(f"  log_lik = {float(np.asarray(ll2)):.4f}")

    # Third evaluation
    print("\nThird evaluation (varying sigma_8 again)...")
    cp["sigma_8"] = 0.815
    t0 = time.time()
    nc.update_params(cp, scal)
    t1 = time.time()
    nc.get_hmf()
    t2 = time.time()
    nc.get_cluster_abundance()
    nc.get_number_counts()
    t3 = time.time()
    ll3 = nc.get_log_lik()
    t4 = time.time()

    print(f"  update_params: {t1-t0:.4f}s")
    print(f"  get_hmf: {t2-t1:.4f}s")
    print(f"  get_cluster_abundance: {t3-t2:.4f}s")
    print(f"  get_log_lik: {t4-t3:.4f}s")
    print(f"  Total: {t4-t0:.4f}s")
    print(f"  log_lik = {float(np.asarray(ll3)):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
