"""Full pipeline profiling: identify every bottleneck."""
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_MAX_THREADS"] = "10"
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

# Detailed profiling with block_until_ready at each step
print("\n=== Detailed profiling (5 evals) ===")
for i in range(5):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = 0.805 + 0.003 * i

    t0 = time.time()
    nc.update_params(cp, dict(nc.scal_rel_params))
    jax.block_until_ready(jnp.array(0.))  # sync
    t1 = time.time()

    nc.get_hmf()
    jax.block_until_ready(nc.hmf_matrix)
    t2 = time.time()

    nc.get_cluster_abundance()
    jax.block_until_ready(nc.abundance_tensor)
    t3 = time.time()

    nc.get_number_counts()
    t4 = time.time()

    ll = nc.get_log_lik()
    jax.block_until_ready(ll)
    t5 = time.time()

    print(f"[{i}] update={t1-t0:.4f}  hmf={t2-t1:.4f}  abund={t3-t2:.4f}  ncounts={t4-t3:.4f}  loglik={t5-t4:.4f}  total={t5-t0:.4f}  ll={float(np.asarray(ll)):.4f}")

# Now profile INSIDE get_log_lik_data with manual instrumentation
print("\n=== Internal get_log_lik_data profiling ===")
# Monkey-patch to add timing
import types

original_get_log_lik_data = nc.get_log_lik_data

def profiled_get_log_lik_data(self):
    indices_no_z = self.catalogue.indices_no_z
    indices_obs_select = self.catalogue.indices_obs_select
    indices_other_obs = self.catalogue.indices_other_obs
    log_lik_data = 0.
    obs_key = self.cnc_params["obs_select"]

    # Skip no_z and obs_select paths (empty for this config)

    if self.cnc_params["data_lik_from_abundance"] == False:
        indices_other_obs = np.concatenate((indices_other_obs, indices_obs_select))
        self.indices_bc = indices_other_obs

    if len(indices_other_obs) > 0:
        t = {}
        t0 = time.time()

        # --- Data preparation (Python) ---
        hmf_matrix_ds = self.hmf_matrix[:,::self.cnc_params["downsample_hmf_bc"]]
        lnM0 = self.ln_M[::self.cnc_params["downsample_hmf_bc"]]
        n_bc = len(indices_other_obs)
        H0 = self.cosmology.background_cosmology.H0.value
        gamma = self.scaling_relations[list(self.scaling_relations.keys())[0]].const.gamma if hasattr(self.scaling_relations[list(self.scaling_relations.keys())[0]], "const") else 0.
        obs_select_key = self.cnc_params["obs_select"]
        sr_sel = self.scaling_relations[obs_select_key]

        idx_bc = np.asarray(indices_other_obs, dtype=int)
        z_clusters = jnp.asarray(self.catalogue.catalogue["z"])[idx_bc]

        # Build obs_data
        obs_data = {}
        for obs_name in self._bc_obs_list:
            obs_vals = jnp.asarray(self.catalogue.catalogue[obs_name])[idx_bc]
            has_obs = ~jnp.isnan(obs_vals)
            obs_vals = jnp.where(has_obs, obs_vals, 0.)
            obs_data[obs_name] = (obs_vals, has_obs)

        patch_key = obs_select_key
        patch_clusters = jnp.asarray(self.catalogue.catalogue_patch[patch_key])[idx_bc].astype(jnp.int32)
        skyfracs_arr = jnp.array(self.scal_rel_selection.skyfracs)
        skyfracs_clusters = skyfracs_arr[patch_clusters]

        sigma_mass_prior = jnp.float64(self.cnc_params["sigma_mass_prior"])
        n_points_dl = int(self.cnc_params["n_points_data_lik"])

        scal_rel_params_ref = self.cnc_params.get("scal_rel_params_ref", self.scal_rel_params)

        apply_cutoff_cfg = self.cnc_params["apply_obs_cutoff"]
        if apply_cutoff_cfg != False and apply_cutoff_cfg.get(str([obs_select_key]), False) == True:
            apply_cutoff = True
            cutoff_val = jnp.float64(self.scal_rel_params.get("q_cutoff", 0.0))
        else:
            apply_cutoff = False
            cutoff_val = jnp.float64(-jnp.inf)

        cosmo_quantities = {
            "E_z": self.E_z, "H0": jnp.float64(H0),
            "D_A": self.D_A, "D_CMB": jnp.float64(self.cosmology.D_CMB),
            "D_l_CMB": self.D_l_CMB, "rho_c": self.rho_c,
            "gamma": jnp.float64(gamma),
        }

        t["data_prep"] = time.time() - t0; t0 = time.time()

        # --- Stage 1: Cosmo interp ---
        z_min_grid = self.redshift_vec[0]
        z_max_grid = self.redshift_vec[-1]
        n_z_grid = self.redshift_vec.shape[0]
        D_A_c, E_z_c, D_l_CMB_c, rho_c_c, hmf_z_c = self._interp_cosmo_jit(
            z_clusters, self.D_A, self.E_z, self.D_l_CMB, self.rho_c,
            hmf_matrix_ds, z_min_grid, z_max_grid, n_z_grid)
        jax.block_until_ready(hmf_z_c)
        t["cosmo_interp"] = time.time() - t0; t0 = time.time()

        # --- Stage 2: Mass range (with prefactors inside JIT) ---
        if self.cnc_params["delta_m_with_ref"] == True:
            ref_sr_params = scal_rel_params_ref
        else:
            ref_sr_params = self.scal_rel_params
        ref_pref_sr = sr_sel.get_prefactor_sr_params(ref_sr_params)
        ref_layer0_sr = sr_sel.get_layer_sr_params(0, ref_sr_params)
        ref_layer1_sr = sr_sel.get_layer_sr_params(1, ref_sr_params)
        ref_layer0_deriv_sr = sr_sel.get_layer_deriv_sr_params(0, ref_sr_params)
        ref_layer1_deriv_sr = sr_sel.get_layer_deriv_sr_params(1, ref_sr_params)
        ref_scatter_sigma = jnp.float64(sr_sel.get_scatter_sigma(ref_sr_params))
        lnM_coarse = jnp.linspace(lnM0[0], lnM0[-1], 128)
        obs_sel_vals = obs_data[obs_select_key][0]
        H0_jnp = jnp.float64(H0)
        D_CMB_jnp = jnp.float64(self.cosmology.D_CMB)
        gamma_jnp = jnp.float64(gamma)

        lnM_min, lnM_max = self._mass_range_with_pref_jit(
            obs_sel_vals,
            E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
            H0_jnp, D_CMB_jnp, gamma_jnp,
            ref_pref_sr,
            ref_layer0_sr, ref_layer1_sr,
            ref_layer0_deriv_sr, ref_layer1_deriv_sr,
            ref_scatter_sigma, sigma_mass_prior,
            lnM0[0], lnM0[-1], lnM_coarse)
        jax.block_until_ready(lnM_min)
        t["mass_range_with_pref"] = time.time() - t0; t0 = time.time()

        # --- Stage 3: All-in-one backward conv + combine + integrate ---
        lnM0_min = lnM0[0]
        lnM0_max = lnM0[-1]
        n_lnM0 = lnM0.shape[0]

        all_obs_vals = jnp.stack([obs_data[o][0] for o in self._bc_obs_list])
        all_has_obs = jnp.stack([obs_data[o][1] for o in self._bc_obs_list])
        all_pref_sr = tuple(
            self.scaling_relations[o].get_prefactor_sr_params(self.scal_rel_params)
            for o in self._bc_obs_list)
        all_layer0_sr = tuple(
            self.scaling_relations[o].get_layer_sr_params(0, self.scal_rel_params)
            for o in self._bc_obs_list)
        all_layer1_sr = tuple(
            self.scaling_relations[o].get_layer_sr_params(1, self.scal_rel_params)
            for o in self._bc_obs_list)
        all_scatter = tuple(
            jnp.float64(self.scaling_relations[o].get_scatter_sigma(self.scal_rel_params))
            for o in self._bc_obs_list)
        all_apply_cut = tuple(
            apply_cutoff if o == obs_select_key else False
            for o in self._bc_obs_list)
        all_cut_val = tuple(
            cutoff_val if o == obs_select_key else jnp.float64(-jnp.inf)
            for o in self._bc_obs_list)

        log_liks, cwh, lnM_grid = self._allinone_bc_jit(
            lnM_min, lnM_max, all_obs_vals, all_has_obs, hmf_z_c, skyfracs_clusters,
            E_z_c, D_A_c, D_l_CMB_c, rho_c_c,
            H0_jnp, D_CMB_jnp, gamma_jnp,
            all_pref_sr, all_layer0_sr, all_layer1_sr,
            all_scatter, all_apply_cut, all_cut_val,
            lnM0_min, lnM0_max, n_lnM0)
        jax.block_until_ready(log_liks)
        t["bc_all_in_one"] = time.time() - t0

        print("  Internal stages:")
        total_internal = sum(t.values())
        for k, v in t.items():
            pct = 100 * v / total_internal
            print(f"    {k:20s}: {v*1000:7.1f}ms  ({pct:4.1f}%)")
        print(f"    {'TOTAL':20s}: {total_internal*1000:7.1f}ms")

# Run profiled version
print("\n=== Profiled get_log_lik_data (3 evals) ===")
for i in range(3):
    cp = dict(nc.cosmo_params)
    cp["sigma_8"] = 0.808 + 0.002 * i
    nc.update_params(cp, dict(nc.scal_rel_params))
    nc.get_hmf()
    nc.get_cluster_abundance()
    nc.get_number_counts()
    jax.block_until_ready(nc.abundance_tensor)

    print(f"\n[{i}]")
    profiled_get_log_lik_data(nc)
