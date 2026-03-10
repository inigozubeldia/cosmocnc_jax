"""Profile the 136ms: where is time spent in cosmocnc_jax?"""
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

cnc_params_update = {
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
    "M_min": 1e13, "M_max": 1e16, "n_points": 16384,
    "n_points_data_lik": 2048,
    "sigma_mass_prior": 10, "downsample_hmf_bc": 2,
    "delta_m_with_ref": True, "scalrel_type_deriv": "numerical",
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

# Warmup
nc.get_number_counts()
nc.get_log_lik()

print("=" * 70)
print("PROFILING cosmocnc_jax per-MCMC-step breakdown")
print("=" * 70)

N = 10
sigma_8_vec = np.linspace(0.808, 0.815, N)
scal = dict(nc.scal_rel_params)

t_update = []
t_hmf = []
t_abundance = []
t_nc = []
t_lik_data = []
t_lik_total = []

for i in range(N):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = sigma_8_vec[i]

    t0 = time.time()
    nc.update_params(cp, scal)
    t1 = time.time()

    nc.get_hmf()
    t2 = time.time()

    nc.get_cluster_abundance()
    t3 = time.time()

    nc.get_number_counts()
    t4 = time.time()

    ll = nc.get_log_lik()
    t5 = time.time()

    t_update.append(t1 - t0)
    t_hmf.append(t2 - t1)
    t_abundance.append(t3 - t2)
    t_nc.append(t4 - t3)
    t_lik_data.append(t5 - t4)
    t_lik_total.append(t5 - t0)

    if i < 3 or i == N - 1:
        print(f"  [{i}] update={t1-t0:.4f}s hmf={t2-t1:.4f}s abund={t3-t2:.4f}s nc={t4-t3:.4f}s lik={t5-t4:.4f}s total={t5-t0:.4f}s")

# Summary (skip first 2)
skip = 2
print(f"\n{'='*70}")
print(f"AVERAGE (last {N-skip} evals)")
print(f"{'='*70}")

components = [
    ("update_params", t_update),
    ("get_hmf", t_hmf),
    ("get_cluster_abundance", t_abundance),
    ("get_number_counts", t_nc),
    ("get_log_lik", t_lik_data),
    ("TOTAL", t_lik_total),
]

total_avg = np.mean(t_lik_total[skip:])
for name, arr in components:
    avg = np.mean(arr[skip:])
    pct = avg / total_avg * 100
    print(f"  {name:25s}: {avg*1000:8.1f}ms  ({pct:5.1f}%)")

# Now profile inside get_hmf
print(f"\n{'='*70}")
print("PROFILING inside get_hmf")
print(f"{'='*70}")

pvd = nc.cosmology._pvd
classy = nc.cosmology.classy
h = nc.cosmology.cosmo_params["h"]
z_vec = nc.redshift_vec

# NN calls
from classy_szfast.classy_szfast import Class_szfast
_orig_calc_hubble = Class_szfast.calculate_hubble
_orig_calc_chi = Class_szfast.calculate_chi

N2 = 20

# 1. Background with caching
def run_background_cached():
    _hubble_done, _chi_done = [False], [False]
    def _once_hubble(csz_self, **kw):
        if not _hubble_done[0]:
            _orig_calc_hubble(csz_self, **kw)
            _hubble_done[0] = True
    def _once_chi(csz_self, **kw):
        if not _chi_done[0]:
            _orig_calc_chi(csz_self, **kw)
            _chi_done[0] = True
    Class_szfast.calculate_hubble = _once_hubble
    Class_szfast.calculate_chi = _once_chi
    try:
        D_A = jnp.asarray(nc.cosmology.background_cosmology.angular_diameter_distance(np.asarray(z_vec)).value)
        E_z = jnp.asarray(nc.cosmology.background_cosmology.H(np.asarray(z_vec)).value / (h * 100.))
        rho_c = jnp.asarray(nc.cosmology.background_cosmology.critical_density(np.asarray(z_vec)).value * 1000. * 3.085677581282e22**3 / 1.98855e30)
        Delta = jnp.asarray(classy.get_delta_mean_from_delta_crit_at_z(500., z_vec, params_values_dict=pvd))
        vol = jnp.asarray(classy.get_volume_dVdzdOmega_at_z(z_vec, params_values_dict=pvd)) * h**(-3)
    finally:
        Class_szfast.calculate_hubble = _orig_calc_hubble
        Class_szfast.calculate_chi = _orig_calc_chi
    return D_A, E_z, rho_c, Delta, vol

# warmup
run_background_cached()

t0 = time.time()
for _ in range(N2):
    run_background_cached()
t_bg = (time.time() - t0) / N2
print(f"\n  Background (cached NN):     {t_bg*1000:.1f}ms")

# 2. Power spectrum
def _get_pk_single_z(z):
    pk, _ = classy.get_pkl_at_z(z, params_values_dict=pvd)
    return pk

# warmup
pk_batch = jax.vmap(_get_pk_single_z)(z_vec)
pk_batch.block_until_ready()

t0 = time.time()
for _ in range(N2):
    pk_batch = jax.vmap(_get_pk_single_z)(z_vec)
    pk_batch.block_until_ready()
t_pk = (time.time() - t0) / N2
print(f"  Power spectrum (vmap PK):   {t_pk*1000:.1f}ms")

# 3. FFTLog + interp + Tinker
from cosmocnc_jax.hmf import batch_sigma_R_from_tophat, build_batch_sigma_fns, compute_hmf_matrix_jit
from cosmocnc_jax.hmf import TINKER08_DELTA_LIN, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c
from mcfit import TophatVar

_, k_arr_np = classy.get_pkl_at_z(float(z_vec[0]), params_values_dict=pvd)
k_arr = jnp.asarray(k_arr_np)
M_vec = jnp.exp(jnp.linspace(jnp.log(1e13), jnp.log(1e16), 16384))
rho_m = nc.halo_mass_function.rho_c_0 * nc.cosmology.cosmo_params["Om0"]

tv0 = TophatVar(np.asarray(k_arr), lowring=True, deriv=0, backend='jax')
tv1 = TophatVar(np.asarray(k_arr), lowring=True, deriv=1, backend='jax')
fns = build_batch_sigma_fns(tv0, tv1, k_arr, type_deriv="numerical")

# warmup
sigma_M, dsigma_M, R_M = batch_sigma_R_from_tophat(tv0, tv1, pk_batch, k_arr, M_vec, rho_m, type_deriv="numerical", _cached_fns=fns)

t0 = time.time()
for _ in range(N2):
    sigma_M, dsigma_M, R_M = batch_sigma_R_from_tophat(tv0, tv1, pk_batch, k_arr, M_vec, rho_m, type_deriv="numerical", _cached_fns=fns)
    sigma_M.block_until_ready()
t_fftlog = (time.time() - t0) / N2
print(f"  FFTLog + interp:            {t_fftlog*1000:.1f}ms")

# 4. Tinker08
_, _, _, _, vol = run_background_cached()
Delta_vec = jnp.asarray(classy.get_delta_mean_from_delta_crit_at_z(500., z_vec, params_values_dict=pvd))

hmf = compute_hmf_matrix_jit(sigma_M, dsigma_M, R_M, M_vec, rho_m,
    z_vec, Delta_vec, vol,
    TINKER08_DELTA_LIN, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c, -1.0, False)
hmf.block_until_ready()

t0 = time.time()
for _ in range(N2):
    hmf = compute_hmf_matrix_jit(sigma_M, dsigma_M, R_M, M_vec, rho_m,
        z_vec, Delta_vec, vol,
        TINKER08_DELTA_LIN, TINKER08_A, TINKER08_a, TINKER08_b, TINKER08_c, -1.0, False)
    hmf.block_until_ready()
t_tinker = (time.time() - t0) / N2
print(f"  Tinker08 HMF:               {t_tinker*1000:.1f}ms")

print(f"\n  get_hmf total (sum):        {(t_bg+t_pk+t_fftlog+t_tinker)*1000:.1f}ms")
