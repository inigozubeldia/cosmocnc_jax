"""Microbenchmark: backward conv with and without FFT convolution."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time

from cosmocnc_jax.utils import interp_uniform, gaussian_1d, convolve_nd

n_clusters = 16000
n_pts = 2048
n_lnM0 = 8192

# Test data
lnM_min = jnp.ones(n_clusters) * (-2.0)
lnM_max = jnp.ones(n_clusters) * 4.0
q_obs = jnp.ones(n_clusters) * 10.0
hmf_z = jnp.ones((n_clusters, n_lnM0)) * 1e-5
pref_logy0 = jnp.ones(n_clusters) * 5.0
pref_M500_theta = jnp.ones(n_clusters) * 0.1
sigma_sz_poly = jnp.array([0.1, -0.2, 0.3, 0.4])
alpha_szifi = jnp.float64(1.2)
dof = jnp.float64(0.0)
lnM0_min = jnp.float64(-2.3)
lnM0_max = jnp.float64(4.6)

# === Version WITH FFT convolution ===
def bc_with_conv(lnM, q_obs, pref_logy0, pref_M500_theta,
                  sigma_sz_poly, alpha_szifi, dof,
                  sigma_lnq, n_points):
    log_y0 = lnM * alpha_szifi + pref_logy0
    log_theta_500 = jnp.log(pref_M500_theta) + lnM / 3.
    log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
    x_l0 = log_y0 - log_sigma_sz

    x_l0_min = x_l0[0]; x_l0_max = x_l0[-1]
    x_l0_linear = jnp.linspace(x_l0_min, x_l0_max, n_points)
    x_l1 = jnp.sqrt(jnp.exp(x_l0_linear)**2 + dof)
    residual = x_l1 - q_obs
    cpdf = gaussian_1d(residual, 1.0)

    dx = x_l0_linear[1] - x_l0_linear[0]
    x_kernel = x_l0_linear - jnp.mean(x_l0_linear) + 0.5 * dx
    kernel = gaussian_1d(x_kernel, jnp.maximum(sigma_lnq, 1e-30))
    cpdf_conv = convolve_nd(cpdf, kernel)
    cpdf = jnp.where(sigma_lnq > 1e-10, cpdf_conv, cpdf)

    cpdf = jnp.maximum(cpdf, 0.)
    cpdf = interp_uniform(x_l0, x_l0_min, x_l0_max, n_points, cpdf)
    return cpdf

# === Version WITHOUT FFT convolution ===
def bc_no_conv(lnM, q_obs, pref_logy0, pref_M500_theta,
                sigma_sz_poly, alpha_szifi, dof, n_points):
    log_y0 = lnM * alpha_szifi + pref_logy0
    log_theta_500 = jnp.log(pref_M500_theta) + lnM / 3.
    log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
    x_l0 = log_y0 - log_sigma_sz

    x_l0_min = x_l0[0]; x_l0_max = x_l0[-1]
    x_l0_linear = jnp.linspace(x_l0_min, x_l0_max, n_points)
    x_l1 = jnp.sqrt(jnp.exp(x_l0_linear)**2 + dof)
    residual = x_l1 - q_obs
    cpdf = gaussian_1d(residual, 1.0)

    cpdf = jnp.maximum(cpdf, 0.)
    cpdf = interp_uniform(x_l0, x_l0_min, x_l0_max, n_points, cpdf)
    return cpdf

# === Full per-cluster with HMF (with conv) ===
def per_cluster_with_conv(mn, mx, obs_v, hz):
    lnM = jnp.linspace(mn, mx, n_pts)
    hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)
    cpdf = bc_with_conv(lnM, obs_v, pref_logy0[0], pref_M500_theta[0],
                         sigma_sz_poly, alpha_szifi, dof,
                         jnp.float64(0.0), n_pts)
    return cpdf, hmf, lnM

def per_cluster_no_conv(mn, mx, obs_v, hz):
    lnM = jnp.linspace(mn, mx, n_pts)
    hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)
    cpdf = bc_no_conv(lnM, obs_v, pref_logy0[0], pref_M500_theta[0],
                       sigma_sz_poly, alpha_szifi, dof, n_pts)
    return cpdf, hmf, lnM

# === Merged: two observables in one per-cluster ===
def per_cluster_merged_no_conv(mn, mx, obs_q, obs_p, hz):
    lnM = jnp.linspace(mn, mx, n_pts)
    hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)
    cpdf_q = bc_no_conv(lnM, obs_q, pref_logy0[0], pref_M500_theta[0],
                         sigma_sz_poly, alpha_szifi, dof, n_pts)
    cpdf_p = bc_no_conv(lnM, obs_p, pref_logy0[0], pref_M500_theta[0],
                         sigma_sz_poly, alpha_szifi, dof, n_pts)
    return cpdf_q, cpdf_p, hmf, lnM

jit_with = jax.jit(jax.vmap(per_cluster_with_conv))
jit_no = jax.jit(jax.vmap(per_cluster_no_conv))
jit_merged = jax.jit(jax.vmap(per_cluster_merged_no_conv))

p_obs = jnp.ones(n_clusters) * 3.0

print("=== With FFT conv (scatter=0 but conv still computed) ===")
for i in range(5):
    t0 = time.time()
    r = jit_with(lnM_min, lnM_max, q_obs, hmf_z)
    jax.block_until_ready(r[0])
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms")

print("\n=== No FFT conv (scatter=0, conv skipped) ===")
for i in range(5):
    t0 = time.time()
    r = jit_no(lnM_min, lnM_max, q_obs, hmf_z)
    jax.block_until_ready(r[0])
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms")

print("\n=== Merged two obs, no conv ===")
for i in range(5):
    t0 = time.time()
    r = jit_merged(lnM_min, lnM_max, q_obs, p_obs, hmf_z)
    jax.block_until_ready(r[0])
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms")

print("\n=== Two separate no-conv calls ===")
for i in range(5):
    t0 = time.time()
    r1 = jit_no(lnM_min, lnM_max, q_obs, hmf_z)
    r2 = jit_no(lnM_min, lnM_max, p_obs, hmf_z)
    jax.block_until_ready(r2[0])
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms")
