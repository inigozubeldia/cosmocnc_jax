"""Profile get_hmf: where is the 68ms spent?"""
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
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc_jax
from mcfit import TophatVar
from cosmocnc_jax.hmf import batch_sigma_R_from_tophat, build_batch_sigma_fns, compute_hmf_matrix_jit, TINKER08_DELTA_LIN, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c

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

# First call to compile everything
nc.get_number_counts()
nc.get_log_lik()

# Now profile a single update_params + get_hmf cycle
pvd = nc.cosmology._pvd
classy = nc.cosmology.classy
redshift_vec = jnp.linspace(0.01, 3.0, 100)
M_vec = jnp.exp(jnp.linspace(jnp.log(1e13), jnp.log(1e16), 16384))
rho_m = nc.halo_mass_function.rho_c_0 * nc.cosmology.cosmo_params["Om0"]

print("=" * 60)
print("PROFILING get_hmf components")
print("=" * 60)

N_ITER = 10

# 1. Power spectrum retrieval
def _get_pk_single_z(z):
    pk, _ = classy.get_pkl_at_z(z, params_values_dict=pvd)
    return pk

# warmup
pk_batch = jax.vmap(_get_pk_single_z)(redshift_vec)
pk_batch.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    pk_batch = jax.vmap(_get_pk_single_z)(redshift_vec)
    pk_batch.block_until_ready()
t_pk = (time.time() - t0) / N_ITER
print(f"\n1. Power spectrum (vmap get_pkl_at_z): {t_pk*1000:.1f}ms")

# Also try serial loop
t0 = time.time()
for _ in range(N_ITER):
    pk_list = []
    for i in range(100):
        pk, k = classy.get_pkl_at_z(float(redshift_vec[i]), params_values_dict=pvd)
        pk_list.append(np.asarray(pk))
    pk_serial = jnp.stack(pk_list)
t_pk_serial = (time.time() - t0) / N_ITER
print(f"   Power spectrum (serial loop):       {t_pk_serial*1000:.1f}ms")

_, k_arr_np = classy.get_pkl_at_z(float(redshift_vec[0]), params_values_dict=pvd)
k_arr = jnp.asarray(k_arr_np)

# 2. Background quantities
t0 = time.time()
for _ in range(N_ITER):
    delta_mean_vec = classy.get_delta_mean_from_delta_crit_at_z(500., redshift_vec, params_values_dict=pvd)
    Delta_vec = jnp.asarray(delta_mean_vec)
    vol_raw = classy.get_volume_dVdzdOmega_at_z(redshift_vec, params_values_dict=pvd)
    vol_vec = jnp.asarray(vol_raw) * 0.674**(-3)
t_bg = (time.time() - t0) / N_ITER
print(f"\n2. Background (Delta + volume):        {t_bg*1000:.1f}ms")

# 3. TophatVar FFTLog transforms (batch)
tv0 = TophatVar(np.asarray(k_arr), lowring=True, deriv=0, backend='jax')
tv1 = TophatVar(np.asarray(k_arr), lowring=True, deriv=1, backend='jax')
fns = build_batch_sigma_fns(tv0, tv1, k_arr, type_deriv="numerical")
vmap_sigma_fn = fns[0]

# warmup
sigma_raw, dsigma_raw = vmap_sigma_fn(pk_batch)
sigma_raw.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    sigma_raw, dsigma_raw = vmap_sigma_fn(pk_batch)
    dsigma_raw.block_until_ready()
t_fftlog = (time.time() - t0) / N_ITER
print(f"\n3. FFTLog transforms (vmap):            {t_fftlog*1000:.1f}ms")

# Try JIT-wrapping the vmap
vmap_sigma_jit = jax.jit(vmap_sigma_fn)
sigma_raw2, dsigma_raw2 = vmap_sigma_jit(pk_batch)
sigma_raw2.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    sigma_raw2, dsigma_raw2 = vmap_sigma_jit(pk_batch)
    dsigma_raw2.block_until_ready()
t_fftlog_jit = (time.time() - t0) / N_ITER
print(f"   FFTLog transforms (vmap + jit):      {t_fftlog_jit*1000:.1f}ms")

# 4. Interpolation to M grid
vmap_interp_fn = fns[1]
R_M = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)

# warmup
sigma_M, dsigma_M = vmap_interp_fn(sigma_raw, dsigma_raw, R_M)
sigma_M.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    sigma_M, dsigma_M = vmap_interp_fn(sigma_raw, dsigma_raw, R_M)
    dsigma_M.block_until_ready()
t_interp = (time.time() - t0) / N_ITER
print(f"\n4. Interpolation to M grid (vmap):      {t_interp*1000:.1f}ms")

# Try JIT
vmap_interp_jit = jax.jit(vmap_interp_fn)
sigma_M2, dsigma_M2 = vmap_interp_jit(sigma_raw, dsigma_raw, R_M)
sigma_M2.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    sigma_M2, dsigma_M2 = vmap_interp_jit(sigma_raw, dsigma_raw, R_M)
    dsigma_M2.block_until_ready()
t_interp_jit = (time.time() - t0) / N_ITER
print(f"   Interpolation to M grid (vmap+jit):  {t_interp_jit*1000:.1f}ms")

# 5. Tinker08 HMF computation
R_matrix = jnp.broadcast_to(R_M[None, :], sigma_M.shape)
tinker_Delta = TINKER08_DELTA_LIN

# warmup
hmf = compute_hmf_matrix_jit(sigma_M, dsigma_M, R_matrix, M_vec, rho_m,
    redshift_vec, Delta_vec, vol_vec,
    tinker_Delta, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c, -1.0, False)
hmf.block_until_ready()

t0 = time.time()
for _ in range(N_ITER):
    hmf = compute_hmf_matrix_jit(sigma_M, dsigma_M, R_matrix, M_vec, rho_m,
        redshift_vec, Delta_vec, vol_vec,
        tinker_Delta, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c, -1.0, False)
    hmf.block_until_ready()
t_tinker = (time.time() - t0) / N_ITER
print(f"\n5. Tinker08 HMF (jit):                  {t_tinker*1000:.1f}ms")

# 6. D_A, E_z, rho_c, D_l_CMB
t0 = time.time()
for _ in range(N_ITER):
    D_A = jnp.asarray(nc.cosmology.background_cosmology.angular_diameter_distance(np.asarray(redshift_vec)).value)
    E_z = jnp.asarray(nc.cosmology.background_cosmology.H(np.asarray(redshift_vec)).value / (nc.cosmology.cosmo_params["h"]*100.))
    rho_c = jnp.asarray(nc.cosmology.background_cosmology.critical_density(np.asarray(redshift_vec)).value * 1000. * 3.08567758149137e22**3 / 1.98855e30)
t_cosmo_bg = (time.time() - t0) / N_ITER
print(f"\n6. D_A, E_z, rho_c:                     {t_cosmo_bg*1000:.1f}ms")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total = t_pk + t_bg + t_fftlog + t_interp + t_tinker + t_cosmo_bg
total_jit = t_pk + t_bg + t_fftlog_jit + t_interp_jit + t_tinker + t_cosmo_bg
print(f"  Power spectrum:    {t_pk*1000:6.1f}ms  ({t_pk/total*100:4.1f}%)")
print(f"  Background:        {t_bg*1000:6.1f}ms  ({t_bg/total*100:4.1f}%)")
print(f"  FFTLog:            {t_fftlog*1000:6.1f}ms  ({t_fftlog/total*100:4.1f}%)  (JIT: {t_fftlog_jit*1000:.1f}ms)")
print(f"  Interpolation:     {t_interp*1000:6.1f}ms  ({t_interp/total*100:4.1f}%)  (JIT: {t_interp_jit*1000:.1f}ms)")
print(f"  Tinker08:          {t_tinker*1000:6.1f}ms  ({t_tinker/total*100:4.1f}%)")
print(f"  D_A/E_z/rho_c:     {t_cosmo_bg*1000:6.1f}ms  ({t_cosmo_bg/total*100:4.1f}%)")
print(f"  TOTAL:             {total*1000:6.1f}ms")
print(f"  TOTAL (w/ JIT):    {total_jit*1000:6.1f}ms")
