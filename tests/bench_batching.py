"""Benchmark different JIT batching strategies for backward conv.

Tests:
1. Current: separate JIT per observable (2 dispatches)
2. Merged: all observables in one JIT (1 dispatch, shared lnM/HMF)
3. Skip-conv: eliminate FFT when scatter=0
4. Merged + skip-conv: both optimizations
5. Chunked vmap: process clusters in chunks
"""
import os
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
from cosmocnc_jax.utils import interp_uniform, gaussian_1d, convolve_nd, simpson

# ── Setup: full pipeline to get realistic data ──
SHARED_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim"], ["p_so_sim"]],
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False, "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16, "n_points": 16384,
    "n_points_data_lik": 2048, "sigma_mass_prior": 10,
    "downsample_hmf_bc": 2, "delta_m_with_ref": True,
    "scalrel_type_deriv": "numerical",
    "cosmo_param_density": "critical", "cosmo_model": "lcdm",
    "hmf_calc": "cnc", "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params = dict(nc.cnc_params)
nc.scal_rel_params = dict(nc.scal_rel_params)
nc.cosmo_params = dict(nc.cosmo_params)
nc.cnc_params.update(SHARED_PARAMS)
nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
nc.scal_rel_params.update({"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.})
nc.initialise()

# Warmup
print("Warmup...")
nc.get_hmf()
nc.get_cluster_abundance()
nc.get_number_counts()
ll = nc.get_log_lik()
jax.block_until_ready(ll)
print(f"  ll={float(np.asarray(ll)):.4f}")

# ── Extract the data we need for benchmarking ──
# Get cluster data from cached bc
obs_select_key = nc.cnc_params["obs_select"]
sr_sel = nc.scaling_relations[obs_select_key]
idx_bc = nc._bc_cached['idx_bc']
z_clusters = nc._bc_cached['z_clusters']
obs_data = nc._bc_cached['obs_data']
skyfracs_clusters = nc._bc_cached['skyfracs_clusters']
n_bc = len(idx_bc)
print(f"\nn_clusters = {n_bc}")

# Cosmo data
H0_jnp = jnp.float64(nc.cosmology.background_cosmology.H0.value)
from cosmocnc_jax.hmf import constants
D_CMB_jnp = jnp.float64(nc.cosmology.D_CMB)
gamma_jnp = jnp.float64(constants().gamma)

hmf_matrix_ds = nc.hmf_matrix[:,::nc.cnc_params["downsample_hmf_bc"]]
lnM0 = nc.ln_M[::nc.cnc_params["downsample_hmf_bc"]]
lnM0_min = lnM0[0]
lnM0_max = lnM0[-1]
n_lnM0 = lnM0.shape[0]
n_points_dl = int(nc.cnc_params["n_points_data_lik"])

# Cosmo interp
z_min_grid = nc.redshift_vec[0]
z_max_grid = nc.redshift_vec[-1]
n_z_grid = nc.redshift_vec.shape[0]
D_A_c, E_z_c, D_l_CMB_c, rho_c_c, hmf_z_c = nc._interp_cosmo_jit(
    z_clusters, nc.D_A, nc.E_z, nc.D_l_CMB, nc.rho_c,
    hmf_matrix_ds, z_min_grid, z_max_grid, n_z_grid)

# Mass range
ref_sr_params = nc.cnc_params.get("scal_rel_params_ref", nc.scal_rel_params)
ref_pref_sr = sr_sel.get_prefactor_sr_params(ref_sr_params)
ref_layer0_sr = sr_sel.get_layer_sr_params(0, ref_sr_params)
ref_layer1_sr = sr_sel.get_layer_sr_params(1, ref_sr_params)
ref_layer0_deriv_sr = sr_sel.get_layer_deriv_sr_params(0, ref_sr_params)
ref_layer1_deriv_sr = sr_sel.get_layer_deriv_sr_params(1, ref_sr_params)
ref_scatter_sigma = jnp.float64(sr_sel.get_scatter_sigma(ref_sr_params))
from cosmocnc_jax.cnc import _N_COARSE_MASS
lnM_coarse = jnp.linspace(lnM0_min, lnM0_max, _N_COARSE_MASS)
obs_sel_vals = obs_data[obs_select_key][0]
sigma_mass_prior = jnp.float64(nc.cnc_params["sigma_mass_prior"])

lnM_min, lnM_max = nc._mass_range_with_pref_jit(
    obs_sel_vals,
    E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
    H0_jnp, D_CMB_jnp, gamma_jnp,
    ref_pref_sr,
    ref_layer0_sr, ref_layer1_sr,
    ref_layer0_deriv_sr, ref_layer1_deriv_sr,
    ref_scatter_sigma, sigma_mass_prior,
    lnM0_min, lnM0_max, lnM_coarse)
jax.block_until_ready(lnM_min)

# ── Approach 1: Current (separate per-observable JIT calls) ──
print("\n=== Approach 1: Current (separate JIT per observable) ===")

def run_current():
    cpdf_list = []
    has_obs_list = []
    hmf_interp = None
    lnM_grid = None
    for obs_name in nc._bc_obs_list:
        sr = nc.scaling_relations[obs_name]
        obs_vals, has_obs = obs_data[obs_name]
        pref_sr = sr.get_prefactor_sr_params(nc.scal_rel_params)
        layer0_sr = sr.get_layer_sr_params(0, nc.scal_rel_params)
        layer1_sr = sr.get_layer_sr_params(1, nc.scal_rel_params)
        scatter_sigma = jnp.float64(sr.get_scatter_sigma(nc.scal_rel_params))
        if obs_name == obs_select_key:
            obs_apply_cutoff = False  # simplified
            obs_cutoff_val = jnp.float64(-jnp.inf)
        else:
            obs_apply_cutoff = False
            obs_cutoff_val = jnp.float64(-jnp.inf)
        cpdf_obs, hmf_obs, lnM_obs = nc._bc_jit_vmaps[obs_name](
            lnM_min, lnM_max, obs_vals, hmf_z_c,
            E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
            H0_jnp, D_CMB_jnp, gamma_jnp,
            pref_sr, layer0_sr, layer1_sr,
            scatter_sigma, obs_apply_cutoff, obs_cutoff_val,
            lnM0_min, lnM0_max, n_lnM0)
        cpdf_list.append(cpdf_obs)
        has_obs_list.append(has_obs)
        if hmf_interp is None:
            hmf_interp = hmf_obs
            lnM_grid = lnM_obs
    log_liks, cwh = nc._combine_integrate_jit(
        tuple(cpdf_list), tuple(has_obs_list),
        hmf_interp, skyfracs_clusters, lnM_grid)
    return log_liks

# Warmup
r = run_current()
jax.block_until_ready(r)

for i in range(5):
    t0 = time.time()
    r = run_current()
    jax.block_until_ready(r)
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms  ll={float(jnp.sum(r)):.4f}")

# ── Approach 2: Merged all-observable JIT ──
# Build a single JIT function that processes ALL observables per cluster.
print("\n=== Approach 2: Merged all-obs single JIT ===")

# Get the layer functions from scaling relations
bc_fns = {}
pref_fns = {}
for obs_name in nc._bc_obs_list:
    sr = nc.scaling_relations[obs_name]
    bc_fns[obs_name] = nc._bc_fns[obs_name]
    pref_fns[obs_name] = sr.get_prefactor_fn_unified()

obs_list = list(nc._bc_obs_list)
n_obs = len(obs_list)

def _make_merged_bc(bc_fn_list, pref_fn_list, n_psr_list, n_pts):
    """Build a single per-cluster function for all observables."""
    n_o = len(bc_fn_list)

    def per_cluster(mn, mx, obs_vals, hz,
                    E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                    H0, D_CMB, gamma,
                    all_pref_sr, all_layer0_sr, all_layer1_sr,
                    all_scatter_sigma, all_apply_cut, all_cut_val,
                    lnM0_min, lnM0_max, n_lnM0):
        # Shared: lnM grid and HMF interp (computed ONCE)
        lnM = jnp.linspace(mn, mx, n_pts)
        hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)

        cpdfs = []
        for i in range(n_o):
            prefactors = pref_fn_list[i](E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                                          H0, D_CMB, gamma, *all_pref_sr[i])
            layer0_args = prefactors + all_layer0_sr[i]
            cpdf = bc_fn_list[i](lnM, obs_vals[i], layer0_args, all_layer1_sr[i],
                                  all_scatter_sigma[i], n_pts,
                                  all_apply_cut[i], all_cut_val[i])
            cpdfs.append(cpdf)
        return tuple(cpdfs) + (hmf, lnM)
    return per_cluster

bc_fn_l = [bc_fns[o] for o in obs_list]
pref_fn_l = [pref_fns[o] for o in obs_list]
n_psr_l = [nc._n_pref_sr[o] for o in obs_list]

merged_per_cluster = _make_merged_bc(bc_fn_l, pref_fn_l, n_psr_l, n_points_dl)

# Build vmap axes for merged function
# obs_vals: stacked (n_obs, n_clusters) -> vmap axis 0 on outer
# all_pref_sr: tuple of tuples, each (None,) * n_psr -> None
# all_layer0_sr, all_layer1_sr: None
# all_scatter_sigma, all_apply_cut, all_cut_val: None
merged_vmap_in = (
    0, 0,  # mn, mx
    1,     # obs_vals: (n_obs, n_clusters) -> vmap over axis 1 (clusters)
    0,     # hz
    0, 0, 0, 0,  # cosmo per-cluster
    None, None, None,  # H0, D_CMB, gamma
    tuple([tuple([None]*n) for n in n_psr_l]),  # all_pref_sr
    tuple([None]*n_obs),  # all_layer0_sr
    tuple([None]*n_obs),  # all_layer1_sr
    tuple([None]*n_obs),  # all_scatter_sigma
    tuple([None]*n_obs),  # all_apply_cut
    tuple([None]*n_obs),  # all_cut_val
    None, None, None,  # lnM0 bounds
)

merged_jit = jax.jit(jax.vmap(merged_per_cluster, in_axes=merged_vmap_in))

# Prepare merged inputs
all_obs_vals = jnp.stack([obs_data[o][0] for o in obs_list])  # (n_obs, n_clusters)
all_has_obs = [obs_data[o][1] for o in obs_list]
all_pref_sr = tuple(nc.scaling_relations[o].get_prefactor_sr_params(nc.scal_rel_params) for o in obs_list)
all_layer0_sr = tuple(nc.scaling_relations[o].get_layer_sr_params(0, nc.scal_rel_params) for o in obs_list)
all_layer1_sr = tuple(nc.scaling_relations[o].get_layer_sr_params(1, nc.scal_rel_params) for o in obs_list)
all_scatter = tuple(jnp.float64(nc.scaling_relations[o].get_scatter_sigma(nc.scal_rel_params)) for o in obs_list)
all_apply_cut = tuple(False for _ in obs_list)
all_cut_val = tuple(jnp.float64(-jnp.inf) for _ in obs_list)

def run_merged():
    result = merged_jit(
        lnM_min, lnM_max, all_obs_vals, hmf_z_c,
        E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
        H0_jnp, D_CMB_jnp, gamma_jnp,
        all_pref_sr, all_layer0_sr, all_layer1_sr,
        all_scatter, all_apply_cut, all_cut_val,
        lnM0_min, lnM0_max, n_lnM0)
    # result = (cpdf_0, cpdf_1, ..., hmf, lnM)
    cpdf_list = [result[i] for i in range(n_obs)]
    hmf_interp = result[n_obs]
    lnM_grid = result[n_obs + 1]
    log_liks, cwh = nc._combine_integrate_jit(
        tuple(cpdf_list), tuple(all_has_obs),
        hmf_interp, skyfracs_clusters, lnM_grid)
    return log_liks

# Warmup
print("  Compiling merged JIT...")
r = run_merged()
jax.block_until_ready(r)
print(f"  Compiled. ll={float(jnp.sum(r)):.4f}")

for i in range(5):
    t0 = time.time()
    r = run_merged()
    jax.block_until_ready(r)
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms  ll={float(jnp.sum(r)):.4f}")


# ── Approach 3: Skip-conv (no FFT when scatter=0) ──
print("\n=== Approach 3: Skip-conv (no FFT when scatter=0) ===")

from cosmocnc_jax.cnc import build_backward_conv_1d

def build_backward_conv_1d_noconv(layer0_fn, layer1_fn, layer0_returns_aux=False):
    """Same as build_backward_conv_1d but WITHOUT the FFT convolution."""
    def backward_conv_1d(lnM, obs_val, layer0_args, layer1_args,
                          sigma_scatter_0, n_points, apply_cutoff, cutoff_val):
        if layer0_returns_aux:
            x_l0, _aux = layer0_fn(lnM, *layer0_args)
        else:
            x_l0 = layer0_fn(lnM, *layer0_args)

        x_l0_min = x_l0[0]
        x_l0_max = x_l0[-1]
        x_l0_linear = jnp.linspace(x_l0_min, x_l0_max, n_points)

        if len(layer1_args) > 0:
            x_l1 = layer1_fn(x_l0_linear, *layer1_args)
        else:
            x_l1 = layer1_fn(x_l0_linear)
        residual = x_l1 - obs_val
        cpdf = gaussian_1d(residual, 1.0)
        cpdf = jnp.where(apply_cutoff & (x_l1 < cutoff_val), 0., cpdf)
        cpdf = jnp.maximum(cpdf, 0.)
        cpdf = interp_uniform(x_l0, x_l0_min, x_l0_max, n_points, cpdf)
        return cpdf
    return backward_conv_1d

# Build no-conv versions
bc_fns_noconv = {}
for obs_name in obs_list:
    sr = nc.scaling_relations[obs_name]
    layer0_fn = sr.get_layer_fn(0)
    layer1_fn = sr.get_layer_fn(1)
    layer0_returns_aux = sr.get_layer_returns_aux(0)
    bc_fns_noconv[obs_name] = build_backward_conv_1d_noconv(
        layer0_fn, layer1_fn, layer0_returns_aux=layer0_returns_aux)

# Merged + no-conv
bc_fn_noconv_l = [bc_fns_noconv[o] for o in obs_list]
merged_noconv_per_cluster = _make_merged_bc(bc_fn_noconv_l, pref_fn_l, n_psr_l, n_points_dl)
merged_noconv_jit = jax.jit(jax.vmap(merged_noconv_per_cluster, in_axes=merged_vmap_in))

def run_merged_noconv():
    result = merged_noconv_jit(
        lnM_min, lnM_max, all_obs_vals, hmf_z_c,
        E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
        H0_jnp, D_CMB_jnp, gamma_jnp,
        all_pref_sr, all_layer0_sr, all_layer1_sr,
        all_scatter, all_apply_cut, all_cut_val,
        lnM0_min, lnM0_max, n_lnM0)
    cpdf_list = [result[i] for i in range(n_obs)]
    hmf_interp = result[n_obs]
    lnM_grid = result[n_obs + 1]
    log_liks, cwh = nc._combine_integrate_jit(
        tuple(cpdf_list), tuple(all_has_obs),
        hmf_interp, skyfracs_clusters, lnM_grid)
    return log_liks

print("  Compiling merged+noconv JIT...")
r = run_merged_noconv()
jax.block_until_ready(r)
print(f"  Compiled. ll={float(jnp.sum(r)):.4f}")

for i in range(5):
    t0 = time.time()
    r = run_merged_noconv()
    jax.block_until_ready(r)
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms  ll={float(jnp.sum(r)):.4f}")


# ── Approach 4: Merged + noconv + integrate all in one JIT ──
print("\n=== Approach 4: Everything in one JIT (bc + combine + integrate) ===")

padding_frac = nc.cnc_params.get("padding_fraction", 0.)
n_drop_int = int(padding_frac * n_points_dl) if padding_frac > 1e-5 else 0

def _make_all_in_one(bc_fn_list, pref_fn_list, n_psr_list, n_pts, n_o, n_drop):
    def per_cluster(mn, mx, obs_vals, has_obs_vals, hz, skyfrac,
                    E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                    H0, D_CMB, gamma,
                    all_pref_sr, all_layer0_sr, all_layer1_sr,
                    all_scatter_sigma, all_apply_cut, all_cut_val,
                    lnM0_min, lnM0_max, n_lnM0):
        lnM = jnp.linspace(mn, mx, n_pts)
        hmf = interp_uniform(lnM, lnM0_min, lnM0_max, n_lnM0, hz, left=0., right=0.)

        cpdf_product = jnp.ones(n_pts)
        for i in range(n_o):
            prefactors = pref_fn_list[i](E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
                                          H0, D_CMB, gamma, *all_pref_sr[i])
            layer0_args = prefactors + all_layer0_sr[i]
            cpdf = bc_fn_list[i](lnM, obs_vals[i], layer0_args, all_layer1_sr[i],
                                  all_scatter_sigma[i], n_pts,
                                  all_apply_cut[i], all_cut_val[i])
            cpdf_product = cpdf_product * jnp.where(has_obs_vals[i], cpdf, 1.)

        cwh = cpdf_product * hmf * 4. * jnp.pi * skyfrac
        if n_drop > 0:
            log_lik = jnp.log(jnp.maximum(simpson(cwh[n_drop:-n_drop], x=lnM[n_drop:-n_drop]), 1e-300))
        else:
            log_lik = jnp.log(jnp.maximum(simpson(cwh, x=lnM), 1e-300))
        return log_lik
    return per_cluster

# With conv (current bc_fns)
allinone_per_cluster = _make_all_in_one(bc_fn_l, pref_fn_l, n_psr_l, n_points_dl, n_obs, n_drop_int)

all_has_obs_arr = jnp.stack([obs_data[o][1] for o in obs_list])  # (n_obs, n_clusters)

allinone_vmap_in = (
    0, 0,  # mn, mx
    1, 1,  # obs_vals, has_obs_vals: (n_obs, n_clusters) -> vmap over axis 1
    0, 0,  # hz, skyfrac
    0, 0, 0, 0,  # cosmo per-cluster
    None, None, None,  # H0, D_CMB, gamma
    tuple([tuple([None]*n) for n in n_psr_l]),
    tuple([None]*n_obs),
    tuple([None]*n_obs),
    tuple([None]*n_obs),
    tuple([None]*n_obs),
    tuple([None]*n_obs),
    None, None, None,
)

allinone_jit = jax.jit(jax.vmap(allinone_per_cluster, in_axes=allinone_vmap_in))

def run_allinone():
    log_liks = allinone_jit(
        lnM_min, lnM_max, all_obs_vals, all_has_obs_arr, hmf_z_c, skyfracs_clusters,
        E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
        H0_jnp, D_CMB_jnp, gamma_jnp,
        all_pref_sr, all_layer0_sr, all_layer1_sr,
        all_scatter, all_apply_cut, all_cut_val,
        lnM0_min, lnM0_max, n_lnM0)
    return log_liks

print("  Compiling all-in-one JIT (with conv)...")
r = run_allinone()
jax.block_until_ready(r)
print(f"  Compiled. ll={float(jnp.sum(r)):.4f}")

for i in range(5):
    t0 = time.time()
    r = run_allinone()
    jax.block_until_ready(r)
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms  ll={float(jnp.sum(r)):.4f}")


# ── Approach 5: All-in-one + noconv ──
print("\n=== Approach 5: Everything in one JIT + no conv ===")

allinone_noconv = _make_all_in_one(bc_fn_noconv_l, pref_fn_l, n_psr_l, n_points_dl, n_obs, n_drop_int)
allinone_noconv_jit = jax.jit(jax.vmap(allinone_noconv, in_axes=allinone_vmap_in))

def run_allinone_noconv():
    log_liks = allinone_noconv_jit(
        lnM_min, lnM_max, all_obs_vals, all_has_obs_arr, hmf_z_c, skyfracs_clusters,
        E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
        H0_jnp, D_CMB_jnp, gamma_jnp,
        all_pref_sr, all_layer0_sr, all_layer1_sr,
        all_scatter, all_apply_cut, all_cut_val,
        lnM0_min, lnM0_max, n_lnM0)
    return log_liks

print("  Compiling all-in-one+noconv JIT...")
r = run_allinone_noconv()
jax.block_until_ready(r)
print(f"  Compiled. ll={float(jnp.sum(r)):.4f}")

for i in range(5):
    t0 = time.time()
    r = run_allinone_noconv()
    jax.block_until_ready(r)
    print(f"  [{i}] {(time.time()-t0)*1000:.1f}ms  ll={float(jnp.sum(r)):.4f}")


# ── Approach 6: Chunked vmap (all-in-one + noconv, process in chunks) ──
print("\n=== Approach 6: Chunked batching (all-in-one + noconv) ===")

for chunk_size in [256, 512, 1024, 2048, 4096, n_bc]:
    # Build chunked version with lax.map over chunks
    n_chunks = (n_bc + chunk_size - 1) // chunk_size
    # Pad arrays to be divisible
    pad_n = n_chunks * chunk_size - n_bc
    if pad_n > 0:
        lnM_min_p = jnp.pad(lnM_min, (0, pad_n), constant_values=0.)
        lnM_max_p = jnp.pad(lnM_max, (0, pad_n), constant_values=1.)
        obs_vals_p = jnp.pad(all_obs_vals, ((0,0),(0, pad_n)), constant_values=0.)
        has_obs_p = jnp.pad(all_has_obs_arr, ((0,0),(0, pad_n)), constant_values=False)
        hz_p = jnp.pad(hmf_z_c, ((0, pad_n),(0,0)), constant_values=0.)
        sky_p = jnp.pad(skyfracs_clusters, (0, pad_n), constant_values=0.)
        Ez_p = jnp.pad(E_z_c, (0, pad_n), constant_values=1.)
        DA_p = jnp.pad(D_A_c, (0, pad_n), constant_values=1.)
        Dl_p = jnp.pad(D_l_CMB_c, (0, pad_n), constant_values=1.)
        rc_p = jnp.pad(rho_c_c, (0, pad_n), constant_values=1.)
    else:
        lnM_min_p, lnM_max_p = lnM_min, lnM_max
        obs_vals_p, has_obs_p = all_obs_vals, all_has_obs_arr
        hz_p, sky_p = hmf_z_c, skyfracs_clusters
        Ez_p, DA_p, Dl_p, rc_p = E_z_c, D_A_c, D_l_CMB_c, rho_c_c

    # Reshape to (n_chunks, chunk_size, ...)
    def reshape_1d(x): return x.reshape(n_chunks, chunk_size)
    def reshape_2d(x): return x.reshape(n_chunks, chunk_size, x.shape[-1])
    def reshape_obs(x): return x.reshape(x.shape[0], n_chunks, chunk_size)

    lnM_min_c = reshape_1d(lnM_min_p)
    lnM_max_c = reshape_1d(lnM_max_p)
    obs_vals_c = reshape_obs(obs_vals_p)  # (n_obs, n_chunks, chunk_size)
    has_obs_c = reshape_obs(has_obs_p)
    hz_c = reshape_2d(hz_p)
    sky_c = reshape_1d(sky_p)
    Ez_c2 = reshape_1d(Ez_p)
    DA_c2 = reshape_1d(DA_p)
    Dl_c2 = reshape_1d(Dl_p)
    rc_c2 = reshape_1d(rc_p)

    # vmap over chunk_size, lax.map over n_chunks
    chunked_vmap = jax.vmap(allinone_noconv, in_axes=allinone_vmap_in)

    @jax.jit
    def run_chunked(lnM_min_c, lnM_max_c, obs_vals_c, has_obs_c, hz_c, sky_c,
                    Ez_c, DA_c, Dl_c, rc_c):
        def process_chunk(i):
            return chunked_vmap(
                lnM_min_c[i], lnM_max_c[i], obs_vals_c[:, i], has_obs_c[:, i],
                hz_c[i], sky_c[i],
                Ez_c[i], DA_c[i], Dl_c[i], rc_c[i],
                H0_jnp, D_CMB_jnp, gamma_jnp,
                all_pref_sr, all_layer0_sr, all_layer1_sr,
                all_scatter, all_apply_cut, all_cut_val,
                lnM0_min, lnM0_max, n_lnM0)
        log_liks = jax.lax.map(process_chunk, jnp.arange(n_chunks))
        return log_liks.reshape(-1)[:n_bc]

    # Warmup
    r = run_chunked(lnM_min_c, lnM_max_c, obs_vals_c, has_obs_c, hz_c, sky_c,
                    Ez_c2, DA_c2, Dl_c2, rc_c2)
    jax.block_until_ready(r)

    times = []
    for i in range(5):
        t0 = time.time()
        r = run_chunked(lnM_min_c, lnM_max_c, obs_vals_c, has_obs_c, hz_c, sky_c,
                        Ez_c2, DA_c2, Dl_c2, rc_c2)
        jax.block_until_ready(r)
        times.append(time.time()-t0)

    mean_ms = np.mean(times[1:]) * 1000  # skip first (may have extra overhead)
    print(f"  chunk_size={chunk_size:5d}  n_chunks={n_chunks:3d}  mean={mean_ms:.1f}ms  ll={float(jnp.sum(r)):.4f}")

print("\nDone!")
