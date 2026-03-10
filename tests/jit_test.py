"""Test: compare cached JIT vmap vs fresh closure JIT vmap."""
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time

from cosmocnc_jax.utils import interp_uniform, gaussian_1d, convolve_nd

# Simple backward conv function (like the old hardcoded version)
def backward_conv_q(lnM, q_obs, pref_logy0, pref_M500_theta,
                     sigma_sz_poly, alpha_szifi, dof,
                     sigma_lnq, n_points, apply_cutoff, q_cutoff):
    log_y0 = lnM * alpha_szifi + pref_logy0
    log_theta_500 = jnp.log(pref_M500_theta) + lnM / 3.
    log_sigma_sz = jnp.polyval(sigma_sz_poly, log_theta_500)
    x_l0 = log_y0 - log_sigma_sz

    x_l0_min = x_l0[0]; x_l0_max = x_l0[-1]
    x_l0_linear = jnp.linspace(x_l0_min, x_l0_max, n_points)
    x_l1 = jnp.sqrt(jnp.exp(x_l0_linear)**2 + dof)
    residual = x_l1 - q_obs
    cpdf = gaussian_1d(residual, 1.0)
    cpdf = jnp.where(apply_cutoff & (x_l1 < q_cutoff), 0., cpdf)

    dx = x_l0_linear[1] - x_l0_linear[0]
    x_kernel = x_l0_linear - jnp.mean(x_l0_linear) + 0.5 * dx
    kernel = gaussian_1d(x_kernel, jnp.maximum(sigma_lnq, 1e-30))
    cpdf_conv = convolve_nd(cpdf, kernel)
    cpdf = jnp.where(sigma_lnq > 1e-10, cpdf_conv, cpdf)
    cpdf = jnp.maximum(cpdf, 0.)
    cpdf = interp_uniform(x_l0, x_l0_min, x_l0_max, n_points, cpdf)
    return cpdf


# Test data
n_clusters = 16000
n_pts = 2048
n_lnM0 = 8192

lnM_min = jnp.ones(n_clusters) * (-2.0)
lnM_max = jnp.ones(n_clusters) * 4.0
q_obs = jnp.ones(n_clusters) * 10.0
hmf_z = jnp.ones((n_clusters, n_lnM0)) * 1e-5
pref_logy0 = jnp.ones(n_clusters) * 5.0
pref_M500_theta = jnp.ones(n_clusters) * 0.1
sigma_sz_poly = jnp.array([0.1, -0.2, 0.3, 0.4])
alpha_szifi = jnp.float64(1.2)
dof = jnp.float64(0.0)
sigma_lnq = jnp.float64(0.0)
lnM0_min = jnp.float64(-2.3)
lnM0_max = jnp.float64(4.6)

print("=" * 60)
print("Test 1: OLD approach - closure capture, create JIT each call")
print("=" * 60)

for trial in range(4):
    # Create fresh JIT function (like old code did each MCMC iteration)
    def _per_cluster(mn, mx, obs_v, hz, p_logy0, p_M500):
        lnM = jnp.linspace(mn, mx, n_pts)
        hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)
        cpdf = backward_conv_q(lnM, obs_v, p_logy0, p_M500,
                                sigma_sz_poly, alpha_szifi, dof,
                                sigma_lnq, n_pts, False, jnp.float64(5.))
        return cpdf, hmf, lnM

    vmap_fn = jax.jit(jax.vmap(_per_cluster))
    t0 = time.time()
    cpdf, hmf, lnM = vmap_fn(lnM_min, lnM_max, q_obs, hmf_z, pref_logy0, pref_M500_theta)
    jax.block_until_ready(cpdf)
    t1 = time.time()
    print(f"  [{trial}] {t1-t0:.4f}s  cpdf[0,:3]={cpdf[0,:3]}")

print("\n" + "=" * 60)
print("Test 2: NEW approach - cached JIT vmap, SR params as args")
print("=" * 60)

# Create cached JIT function (like new code at init)
def _per_cluster_cached(mn, mx, obs_v, hz, lnM0_mn, lnM0_mx, n_lnM0_arg,
                         prefactors, layer0_sr, layer1_sr,
                         scatter_sigma, apply_cut, cut_val):
    lnM = jnp.linspace(mn, mx, n_pts)
    hmf = interp_uniform(lnM, lnM0_mn, lnM0_mx, n_lnM0_arg, hz, left=0., right=0.)
    p_logy0, p_M500 = prefactors
    sr_poly, alpha_sr = layer0_sr
    dof_sr, = layer1_sr
    cpdf = backward_conv_q(lnM, obs_v, p_logy0, p_M500,
                            sr_poly, alpha_sr, dof_sr,
                            scatter_sigma, n_pts, apply_cut, cut_val)
    return cpdf, hmf, lnM

cached_vmap = jax.jit(jax.vmap(_per_cluster_cached, in_axes=(
    0, 0, 0, 0,  # mn, mx, obs_v, hz
    None, None, None,  # lnM0_min/max, n_lnM0
    (0, 0),  # prefactors
    None, None,  # layer0_sr, layer1_sr
    None, None, None  # scatter, apply_cut, cut_val
)))

for trial in range(4):
    t0 = time.time()
    cpdf2, hmf2, lnM2 = cached_vmap(
        lnM_min, lnM_max, q_obs, hmf_z,
        lnM0_min, lnM0_max, n_lnM0,
        (pref_logy0, pref_M500_theta),
        (sigma_sz_poly, alpha_szifi),
        (dof,),
        sigma_lnq, jnp.bool_(False), jnp.float64(5.))
    jax.block_until_ready(cpdf2)
    t1 = time.time()
    print(f"  [{trial}] {t1-t0:.4f}s  cpdf[0,:3]={cpdf2[0,:3]}")

print("\n" + "=" * 60)
print("Test 3: NEW approach but closure-capture SR params")
print("=" * 60)

def _make_closure_jit(sr_poly, alpha_sr, dof_sr, scatter_s, apply_c, cut_v):
    def _per_cluster_v3(mn, mx, obs_v, hz, p_logy0, p_M500):
        lnM = jnp.linspace(mn, mx, n_pts)
        hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)
        cpdf = backward_conv_q(lnM, obs_v, p_logy0, p_M500,
                                sr_poly, alpha_sr, dof_sr,
                                scatter_s, n_pts, apply_c, cut_v)
        return cpdf, hmf, lnM
    return jax.jit(jax.vmap(_per_cluster_v3))

for trial in range(4):
    fn = _make_closure_jit(sigma_sz_poly, alpha_szifi, dof, sigma_lnq, False, jnp.float64(5.))
    t0 = time.time()
    cpdf3, hmf3, lnM3 = fn(lnM_min, lnM_max, q_obs, hmf_z, pref_logy0, pref_M500_theta)
    jax.block_until_ready(cpdf3)
    t1 = time.time()
    print(f"  [{trial}] {t1-t0:.4f}s  cpdf[0,:3]={cpdf3[0,:3]}")
