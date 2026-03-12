"""Test N-D backward convolution: compare cosmocnc vs cosmocnc_jax
when observables are in the same correlation set [["q_so_sim", "p_so_sim"]].
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

def get_params():
    """Return params with correlated observables: [["q_so_sim", "p_so_sim"]]"""
    cnc_params_update = {
        "cluster_catalogue": "SO_sim_0",
        "observables": [["q_so_sim", "p_so_sim"]],  # Single correlated set!
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
        "bias_sz": 3.,
        "bias_cmblens": 3.,
        "sigma_lnq_szifi": 0.2,
        "sigma_lnp": 0.2,
        "corr_lnq_lnp": 0.5,      # Non-zero correlation!
        "A_szifi": -4.439,
        "alpha_szifi": 1.617,
        "a_lens": 1.,
        "dof": 0.,
    }

    cosmo = {
        "Om0": 0.315,
        "Ob0": 0.04897,
        "h": 0.674,
        "sigma_8": 0.811,
        "n_s": 0.96,
        "m_nu": 0.06,
        "tau_reio": 0.0544,
        "w0": -1.,
        "N_eff": 3.046,
    }

    return cnc_params_update, scal_rel, cosmo


def setup_and_init(pkg, cnc_p, sr_p, cosmo_p):
    nc = pkg.cluster_number_counts()
    nc.cnc_params.update(cnc_p)
    nc.scal_rel_params.update(sr_p)
    nc.cosmo_params.update(cosmo_p)
    nc.initialise()
    return nc


def main():
    print("=" * 60)
    print("N-D BACKWARD CONV TEST: [['q_so_sim', 'p_so_sim']]")
    print("=" * 60)

    cnc_params, scal_rel, cosmo = get_params()

    # Also test the 1D case for comparison
    cnc_params_1d = dict(cnc_params)
    cnc_params_1d["observables"] = [["q_so_sim"], ["p_so_sim"]]

    # JAX params: use classy_sz_jax + hmf_calc=cnc
    cnc_params_jax = dict(cnc_params)
    cnc_params_jax["cosmology_tool"] = "classy_sz_jax"
    cnc_params_jax["hmf_calc"] = "cnc"
    cnc_params_1d_jax = dict(cnc_params_1d)
    cnc_params_1d_jax["cosmology_tool"] = "classy_sz_jax"
    cnc_params_1d_jax["hmf_calc"] = "cnc"

    print("\n  Initialising cosmocnc (2D correlated)...")
    nc_orig = setup_and_init(cosmocnc, dict(cnc_params), dict(scal_rel), dict(cosmo))

    print("  Initialising cosmocnc_jax (2D correlated)...")
    nc_jax = setup_and_init(cosmocnc_jax, dict(cnc_params_jax), dict(scal_rel), dict(cosmo))

    print("  Initialising cosmocnc (1D independent)...")
    nc_orig_1d = setup_and_init(cosmocnc, dict(cnc_params_1d), dict(scal_rel), dict(cosmo))

    print("  Initialising cosmocnc_jax (1D independent)...")
    nc_jax_1d = setup_and_init(cosmocnc_jax, dict(cnc_params_1d_jax), dict(scal_rel), dict(cosmo))

    # ── 2D correlated ──
    print("\n  Computing log_lik (cosmocnc, 2D correlated)...")
    t0 = time.time()
    log_lik_orig_2d = nc_orig.get_log_lik()
    t_orig = time.time() - t0
    print(f"    log_lik = {log_lik_orig_2d:.6f}, time = {t_orig:.2f}s")

    print("  Computing log_lik (cosmocnc_jax, 2D correlated)...")
    t0 = time.time()
    log_lik_jax_2d = nc_jax.get_log_lik()
    t_jax = time.time() - t0
    print(f"    log_lik = {float(log_lik_jax_2d):.6f}, time = {t_jax:.2f}s")

    rel_diff_2d = abs(float(log_lik_jax_2d) - log_lik_orig_2d) / abs(log_lik_orig_2d)
    status_2d = "PASS" if rel_diff_2d < 1e-3 else "FAIL"
    print(f"\n  {status_2d} 2D correlated: rel_diff = {rel_diff_2d:.6e}")

    # ── 1D independent ──
    print("\n  Computing log_lik (cosmocnc, 1D independent)...")
    t0 = time.time()
    log_lik_orig_1d = nc_orig_1d.get_log_lik()
    t_orig = time.time() - t0
    print(f"    log_lik = {log_lik_orig_1d:.6f}, time = {t_orig:.2f}s")

    print("  Computing log_lik (cosmocnc_jax, 1D independent)...")
    t0 = time.time()
    log_lik_jax_1d = nc_jax_1d.get_log_lik()
    t_jax = time.time() - t0
    print(f"    log_lik = {float(log_lik_jax_1d):.6f}, time = {t_jax:.2f}s")

    rel_diff_1d = abs(float(log_lik_jax_1d) - log_lik_orig_1d) / abs(log_lik_orig_1d)
    status_1d = "PASS" if rel_diff_1d < 1e-3 else "FAIL"
    print(f"\n  {status_1d} 1D independent: rel_diff = {rel_diff_1d:.6e}")

    # Show that 2D != 1D (correlation matters)
    print(f"\n  2D vs 1D difference (cosmocnc): {abs(log_lik_orig_2d - log_lik_orig_1d):.4f}")
    print(f"  2D vs 1D difference (jax):      {abs(float(log_lik_jax_2d) - float(log_lik_jax_1d)):.4f}")


if __name__ == "__main__":
    main()
