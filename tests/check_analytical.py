"""Check: does analytical Delta/volume match classy_szfast direct calls?"""
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sys
import time

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
    "cosmology_tool": "classy_sz_jax",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm", "hmf_calc": "cnc",
    "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params = dict(nc.cnc_params)
nc.scal_rel_params = dict(nc.scal_rel_params)
nc.cosmo_params = dict(nc.cosmo_params)
nc.cnc_params.update(cnc_params_update)
nc.scal_rel_params.update({"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.})
nc.initialise()

pvd = nc.cosmology._pvd
classy = nc.cosmology.classy
h = nc.cosmology.cosmo_params["h"]
Om0 = nc.cosmology.cosmo_params["Om0"]
z_vec = jnp.linspace(0.01, 3.0, 100)

print("=" * 60)
print("Comparing analytical vs classy_szfast direct calls")
print("=" * 60)

# === H(z) and E(z) ===
z_with_0 = jnp.concatenate([jnp.array([0.]), z_vec])
H_all = jnp.asarray(classy.get_hubble_at_z(z_with_0, params_values_dict=pvd))
H_0 = H_all[0]
H_z = H_all[1:]
E_z_new = H_z / H_0

# Old way: through wrapper
H_wrapper = jnp.asarray(classy.get_hubble_at_z(z_vec, params_values_dict=pvd))
E_z_old = H_wrapper * 299792.458 / (h * 100.)

err_Ez = float(jnp.max(jnp.abs(E_z_new - E_z_old) / jnp.maximum(jnp.abs(E_z_old), 1e-30)))
print(f"\nE_z: new (H/H0) vs old (H*c/(h*100)): max_rel={err_Ez:.3e}")
print(f"  H_0 from emulator: {float(H_0):.10e}")
print(f"  h*100/c_km_s:      {h*100./299792.458:.10e}")
print(f"  ratio:              {float(H_0) / (h*100./299792.458):.10e}")

# === D_A ===
D_A_new = jnp.asarray(classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd))
D_A_old = jnp.asarray(classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd))
err_DA = float(jnp.max(jnp.abs(D_A_new - D_A_old) / jnp.maximum(jnp.abs(D_A_old), 1e-30)))
print(f"\nD_A: consistency check: max_rel={err_DA:.3e}")

# === chi ===
chi_z = D_A_new * (1. + z_vec)

# === Delta ===
# Old: through classy_szfast
delta_num = 500.
delta_mean_old = jnp.asarray(classy.get_delta_mean_from_delta_crit_at_z(
    delta_num, z_vec, params_values_dict=pvd))

# New: analytical
Omega_m_z = Om0 * (1. + z_vec)**3 / E_z_new**2
delta_mean_new = delta_num / Omega_m_z

err_delta = float(jnp.max(jnp.abs(delta_mean_new - delta_mean_old) / jnp.maximum(jnp.abs(delta_mean_old), 1e-30)))
print(f"\nDelta: analytical vs get_delta_mean_from_delta_crit_at_z: max_rel={err_delta:.3e}")
print(f"  Delta[0]: old={float(delta_mean_old[0]):.6f}, new={float(delta_mean_new[0]):.6f}")
print(f"  Delta[50]: old={float(delta_mean_old[50]):.6f}, new={float(delta_mean_new[50]):.6f}")
print(f"  Delta[99]: old={float(delta_mean_old[99]):.6f}, new={float(delta_mean_new[99]):.6f}")

# === Volume ===
vol_old = jnp.asarray(classy.get_volume_dVdzdOmega_at_z(z_vec, params_values_dict=pvd)) * h**(-3)
vol_new = 2997.92458 * chi_z**2 / (h * E_z_new)

err_vol = float(jnp.max(jnp.abs(vol_new - vol_old) / jnp.maximum(jnp.abs(vol_old), 1e-30)))
print(f"\nVolume: analytical vs get_volume_dVdzdOmega_at_z: max_rel={err_vol:.3e}")
print(f"  vol[0]: old={float(vol_old[0]):.6f}, new={float(vol_new[0]):.6f}")
print(f"  vol[50]: old={float(vol_old[50]):.6f}, new={float(vol_new[50]):.6f}")

# === rho_c ===
rho_c_prefactor = 3. / (8. * jnp.pi * 6.67428e-11 * 1.98855e30) * 3.085677581282e22 * (2.99792458e8)**2
rho_c_new = rho_c_prefactor * H_z**2

rho_c_old = jnp.asarray(classy.get_rho_crit_at_z(z_vec, params_values_dict=pvd)) * h**2
err_rho = float(jnp.max(jnp.abs(rho_c_new - rho_c_old) / jnp.maximum(jnp.abs(rho_c_old), 1e-30)))
print(f"\nrho_c: analytical vs get_rho_crit_at_z * h^2: max_rel={err_rho:.3e}")

# === Timing comparison ===
print("\n" + "=" * 60)
print("TIMING: old (redundant NN calls) vs new (2 NN calls + analytical)")
print("=" * 60)

N = 20

# Old way: 5-6 separate wrapper calls
t0 = time.time()
for _ in range(N):
    _ = classy.get_hubble_at_z(z_vec, params_values_dict=pvd)
    _ = classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd)
    _ = classy.get_rho_crit_at_z(z_vec, params_values_dict=pvd)
    _ = classy.get_delta_mean_from_delta_crit_at_z(delta_num, z_vec, params_values_dict=pvd)
    _ = classy.get_volume_dVdzdOmega_at_z(z_vec, params_values_dict=pvd)
t_old = (time.time() - t0) / N

# New way: 2 calls + analytical
t0 = time.time()
for _ in range(N):
    z_with_0 = jnp.concatenate([jnp.array([0.]), z_vec])
    H_all = jnp.asarray(classy.get_hubble_at_z(z_with_0, params_values_dict=pvd))
    D_A = jnp.asarray(classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd))
    H_0 = H_all[0]; H_z = H_all[1:]
    E_z = H_z / H_0
    chi_z = D_A * (1. + z_vec)
    rho_c = rho_c_prefactor * H_z**2
    Omega_m_z = Om0 * (1. + z_vec)**3 / E_z**2
    Delta = delta_num / Omega_m_z
    vol = 2997.92458 * chi_z**2 / (h * E_z)
t_new = (time.time() - t0) / N

print(f"  Old (5 calls): {t_old*1000:.1f}ms")
print(f"  New (2 calls): {t_new*1000:.1f}ms")
print(f"  Speedup: {t_old/t_new:.1f}x")
