"""Profile N-D backward conv: separate JIT compilation from execution time."""

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

import cosmocnc_jax

def get_params():
    cnc_params_update = {
        "cluster_catalogue": "SO_sim_0",
        "observables": [["q_so_sim", "p_so_sim"]],
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
        "n_points_data_lik": 128,
        "cosmology_tool": "classy_sz",
        "likelihood_type": "unbinned",
        "data_lik_type": "backward_convolutional",
        "stacked_likelihood": False,
        "apply_obs_cutoff": False,
        "sigma_mass_prior": 5.,
        "z_errors": False,
        "delta_m_with_ref": False,
        "scalrel_type_deriv": "analytical",
        "downsample_hmf_bc": 8,
        "padding_fraction": 0.,
        "bc_chunk_size": 1000,
        "hmf_type": "Tinker08",
        "hmf_calc": "classy_sz",
        "sigma_R_type": "class_sz",
        "mass_definition": "500c",
    }
    scal_rel = {
        "bias_sz": 3., "bias_cmblens": 3.,
        "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2,
        "corr_lnq_lnp": 0.5,
        "A_szifi": -4.439, "alpha_szifi": 1.617,
        "a_lens": 1., "dof": 0.,
    }
    cosmo = {
        "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
        "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
        "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    }
    return cnc_params_update, scal_rel, cosmo

def main():
    cnc_params, scal_rel, cosmo = get_params()

    print("Initialising cosmocnc_jax (2D correlated)...")
    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params.update(cnc_params)
    nc.scal_rel_params.update(scal_rel)
    nc.cosmo_params.update(cosmo)
    nc.initialise()

    print("\n--- Call 1 (includes JIT compilation) ---")
    t0 = time.time()
    ll1 = nc.get_log_lik()
    t1 = time.time() - t0
    print(f"  log_lik = {float(ll1):.6f}, time = {t1:.2f}s")

    print("\n--- Call 2 (JIT cached) ---")
    t0 = time.time()
    ll2 = nc.get_log_lik()
    t2 = time.time() - t0
    print(f"  log_lik = {float(ll2):.6f}, time = {t2:.2f}s")

    print("\n--- Call 3 (JIT cached, confirmed) ---")
    t0 = time.time()
    ll3 = nc.get_log_lik()
    t3 = time.time() - t0
    print(f"  log_lik = {float(ll3):.6f}, time = {t3:.2f}s")

    print(f"\n  JIT compilation overhead: {t1 - t2:.2f}s")
    print(f"  Steady-state per-eval: {t2:.2f}s")

if __name__ == "__main__":
    main()
