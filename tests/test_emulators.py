"""Verify emulators.py: direct _predict vs Cython class_sz calls."""
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
import jax.numpy as jnp
import numpy as np
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc_jax

# Setup
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

# Access cosmology model
cosmo = nc.cosmology
classy = cosmo.classy
pvd = cosmo._pvd

print("=" * 70)
print("EMULATOR VERIFICATION")
print("=" * 70)

# ==================== 1. Init emulators ====================
from cosmocnc_jax.emulators import (
    init_emulators, _call_emulator, build_cosmo_vec,
    extract_pk_power_fac, make_predict_fns, make_sigma8_solver
)

emulators, param_orders, z_interp = init_emulators('lcdm')
print("\n1. Emulator parameter orderings:")
for name, order in param_orders.items():
    print(f"   {name}: {order}")

# ==================== 2. Verify _predict matches predict ====================
print("\n2. Verifying _call_emulator vs predict() for each emulator...")

# H emulator
print("\n   H emulator (ten_to_predictions=True):")
h_em = emulators['h']
cosmo_vec_h = build_cosmo_vec(pvd, param_orders['h'])
direct_H = np.asarray(_call_emulator(h_em, cosmo_vec_h)).ravel()

# predict() via dict
h_dict = {k: [pvd[k]] for k in param_orders['h']}
predict_H = np.asarray(h_em.predict(h_dict)).ravel()

err_H = np.max(np.abs(direct_H - predict_H) / np.maximum(np.abs(predict_H), 1e-30))
print(f"   max_rel error: {err_H:.3e}")
print(f"   H[0]={direct_H[0]:.6e}, H[2500]={direct_H[2500]:.6e}, H[4999]={direct_H[4999]:.6e}")
print(f"   ten_to_predictions={h_em.ten_to_predictions}, log={h_em.log}")

# DA emulator
print("\n   DA emulator (ten_to_predictions=False):")
da_em = emulators['da']
cosmo_vec_da = build_cosmo_vec(pvd, param_orders['da'])
direct_DA = np.asarray(_call_emulator(da_em, cosmo_vec_da)).ravel()

da_dict = {k: [pvd[k]] for k in param_orders['da']}
predict_DA = np.asarray(da_em.predict(da_dict)).ravel()

err_DA = np.max(np.abs(direct_DA - predict_DA) / np.maximum(np.abs(predict_DA), 1e-30))
print(f"   max_rel error: {err_DA:.3e}")
print(f"   DA[0]={direct_DA[0]:.6e}, DA[2500]={direct_DA[2500]:.6e}")
print(f"   ten_to_predictions={da_em.ten_to_predictions}, log={da_em.log}")

# DER emulator
print("\n   DER emulator:")
der_em = emulators['der']
cosmo_vec_der = build_cosmo_vec(pvd, param_orders['der'])
direct_DER = np.asarray(_call_emulator(der_em, cosmo_vec_der)).ravel()

der_dict = {k: [pvd[k]] for k in param_orders['der']}
predict_DER = np.asarray(der_em.predict(der_dict)).ravel()

err_DER = np.max(np.abs(direct_DER - predict_DER) / np.maximum(np.abs(predict_DER), 1e-30))
print(f"   max_rel error: {err_DER:.3e}")
print(f"   sigma8={direct_DER[1]:.6f}, z_rec={direct_DER[6]:.4f}")
print(f"   ten_to_predictions={der_em.ten_to_predictions}, log={der_em.log}")

# PKL emulator
print("\n   PKL emulator (ten_to_predictions=False):")
pkl_em = emulators['pkl']
pkl_param_order = param_orders['pkl']
z_test = 0.5
cosmo_keys = [k for k in pkl_param_order if k != 'z_pk_save_nonclass']
cosmo_vec_pkl = jnp.array([pvd[k] for k in cosmo_keys])
input_vec_pkl = jnp.concatenate([cosmo_vec_pkl, jnp.array([z_test])])
direct_PKL = np.asarray(_call_emulator(pkl_em, input_vec_pkl)).ravel()

pkl_dict = {k: [pvd[k]] for k in cosmo_keys}
pkl_dict['z_pk_save_nonclass'] = [z_test]
predict_PKL = np.asarray(pkl_em.predict(pkl_dict)).ravel()

err_PKL = np.max(np.abs(direct_PKL - predict_PKL) / np.maximum(np.abs(predict_PKL), 1e-30))
print(f"   max_rel error: {err_PKL:.3e}")
print(f"   raw[0]={direct_PKL[0]:.6e}, raw[100]={direct_PKL[100]:.6e}")
print(f"   ten_to_predictions={pkl_em.ten_to_predictions}, log={pkl_em.log}")

# ==================== 3. Extract pk_power_fac ====================
print("\n3. Extracting pk_power_fac...")
pk_power_fac, k_arr = extract_pk_power_fac(emulators, param_orders, classy, pvd)
print(f"   k_arr: shape={k_arr.shape}, range=[{float(k_arr[0]):.4e}, {float(k_arr[-1]):.4e}]")
print(f"   pk_power_fac: shape={pk_power_fac.shape}")
print(f"   pk_power_fac[0]={float(pk_power_fac[0]):.6e}, pk_power_fac[100]={float(pk_power_fac[100]):.6e}")

# Verify: direct P(k) matches Cython P(k) at multiple z
print("\n4. Verifying P(k) at multiple z...")
for z_check in [0.01, 0.5, 1.0, 2.0]:
    pk_cython, _ = classy.get_pkl_at_z(z_check, params_values_dict=pvd)
    pk_cython = np.asarray(pk_cython).ravel()

    input_check = jnp.concatenate([cosmo_vec_pkl, jnp.array([z_check])])
    raw_check = _call_emulator(pkl_em, input_check)
    pk_direct = np.asarray(jnp.power(10., raw_check) * pk_power_fac).ravel()

    err_pk = np.max(np.abs(pk_direct - pk_cython) / np.maximum(np.abs(pk_cython), 1e-30))
    print(f"   z={z_check:.2f}: max_rel={err_pk:.3e}")

# ==================== 5. Verify H/DA ====================
print("\n5. Verifying H(z) and D_A(z)...")
z_vec = jnp.linspace(0.01, 3.0, 100)

# H: Cython path
H_cython = np.asarray(classy.get_hubble_at_z(z_vec, params_values_dict=pvd))

# H: direct
H_grid_direct = np.asarray(_call_emulator(h_em, cosmo_vec_h)).ravel()
H_direct = np.asarray(jnp.interp(z_vec, z_interp, H_grid_direct))

err_H_z = np.max(np.abs(H_direct - H_cython) / np.maximum(np.abs(H_cython), 1e-30))
print(f"   H(z) max_rel: {err_H_z:.3e}")

# DA: Cython path
DA_cython = np.asarray(classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd))

# DA: direct (via chi interpolation)
DA_grid_direct = np.asarray(_call_emulator(da_em, cosmo_vec_da)).ravel()
chi_grid_direct = DA_grid_direct * (1. + np.asarray(z_interp))
chi_at_z = np.asarray(jnp.interp(z_vec, z_interp, jnp.asarray(chi_grid_direct)))
DA_direct = chi_at_z / (1. + np.asarray(z_vec))

err_DA_z = np.max(np.abs(DA_direct - DA_cython) / np.maximum(np.abs(DA_cython), 1e-30))
print(f"   D_A(z) max_rel: {err_DA_z:.3e}")

# ==================== 6. Build predict functions ====================
print("\n6. Building JIT'd predict functions...")
predict_H, predict_DA, predict_pk_batch, predict_der = make_predict_fns(
    emulators, param_orders, z_interp, pk_power_fac)

# Warmup JIT
_ = predict_H(cosmo_vec_h, z_vec)
_ = predict_DA(cosmo_vec_da, z_vec)
_ = predict_pk_batch(cosmo_vec_pkl, z_vec)
_ = predict_der(cosmo_vec_der)

print("   JIT compilation done.")

# ==================== 7. Timing ====================
print("\n7. Timing (20 iterations each)...")
N = 20

# H
t0 = time.time()
for _ in range(N):
    h_result = predict_H(cosmo_vec_h, z_vec)
    h_result.block_until_ready()
t_H = (time.time() - t0) / N

# DA
t0 = time.time()
for _ in range(N):
    da_result = predict_DA(cosmo_vec_da, z_vec)
    da_result.block_until_ready()
t_DA = (time.time() - t0) / N

# PKL batch
t0 = time.time()
for _ in range(N):
    pk_result = predict_pk_batch(cosmo_vec_pkl, z_vec)
    pk_result.block_until_ready()
t_PK = (time.time() - t0) / N

# DER
t0 = time.time()
for _ in range(N):
    der_result = predict_der(cosmo_vec_der)
    der_result.block_until_ready()
t_DER = (time.time() - t0) / N

# Compare with Cython
t0 = time.time()
for _ in range(N):
    _ = classy.get_hubble_at_z(z_vec, params_values_dict=pvd)
t_H_cython = (time.time() - t0) / N

t0 = time.time()
for _ in range(N):
    _ = classy.get_angular_distance_at_z(z_vec, params_values_dict=pvd)
t_DA_cython = (time.time() - t0) / N

def _get_pk_z(z):
    pk, _ = classy.get_pkl_at_z(z, params_values_dict=pvd)
    return pk
pk_cython_batch = jax.vmap(_get_pk_z)(z_vec)
pk_cython_batch.block_until_ready()
t0 = time.time()
for _ in range(N):
    pk_cython_batch = jax.vmap(_get_pk_z)(z_vec)
    pk_cython_batch.block_until_ready()
t_PK_cython = (time.time() - t0) / N

t0 = time.time()
for _ in range(N):
    _ = classy.get_sigma8_and_der(params_values_dict=pvd)
t_DER_cython = (time.time() - t0) / N

print(f"   H(z):   direct={t_H*1000:.2f}ms  Cython={t_H_cython*1000:.2f}ms  speedup={t_H_cython/max(t_H,1e-10):.1f}x")
print(f"   D_A(z): direct={t_DA*1000:.2f}ms  Cython={t_DA_cython*1000:.2f}ms  speedup={t_DA_cython/max(t_DA,1e-10):.1f}x")
print(f"   P(k,z): direct={t_PK*1000:.2f}ms  Cython={t_PK_cython*1000:.2f}ms  speedup={t_PK_cython/max(t_PK,1e-10):.1f}x")
print(f"   DER:    direct={t_DER*1000:.2f}ms  Cython={t_DER_cython*1000:.2f}ms  speedup={t_DER_cython/max(t_DER,1e-10):.1f}x")

# ==================== 8. Sigma8 solver ====================
print("\n8. Testing sigma8->A_s Newton solver...")
lnAs_index = param_orders['der'].index('ln10^{10}A_s')
find_lnAs = make_sigma8_solver(emulators['der'], lnAs_index)

sigma8_target = cosmo.cosmo_params["sigma_8"]
lnAs_init = pvd['ln10^{10}A_s']
print(f"   sigma8_target = {sigma8_target:.6f}")
print(f"   lnAs_init = {lnAs_init:.6f}")

# Warmup
lnAs_result = find_lnAs(cosmo_vec_der, jnp.float64(sigma8_target), jnp.float64(lnAs_init))
print(f"   lnAs_result = {float(lnAs_result):.8f}")

# Verify sigma8 at result
der_check = predict_der(cosmo_vec_der.at[lnAs_index].set(lnAs_result))
sigma8_check = float(der_check[1])
print(f"   sigma8 at result = {sigma8_check:.8f}")
print(f"   residual = {abs(sigma8_check - sigma8_target):.3e}")

# Timing
t0 = time.time()
for _ in range(N):
    lnAs_result = find_lnAs(cosmo_vec_der, jnp.float64(sigma8_target), jnp.float64(lnAs_init))
    lnAs_result.block_until_ready()
t_solver = (time.time() - t0) / N
print(f"   solver time: {t_solver*1000:.2f}ms")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
