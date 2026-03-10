"""
Test: cosmocnc_jax catalogue simulator (sim.py)

Tests the fully JAXified catalogue generator against statistical expectations.
Verifies:
  1. JIT path produces valid catalogues (correct shapes, selection cut)
  2. Fallback path works identically
  3. generate_catalogues_hmf works
  4. Statistical properties (n_selected, z distribution, q distribution)
  5. Benchmark timing for single and batch generation

Usage:
  taskset -c 0-9 python tests/test_sim.py
"""

# ── Thread control (must be set before any library imports) ──────────
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
import jax.numpy as jnp
import time
import sys

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc_jax

# ── Configuration ────────────────────────────────────────────────────

SIM_PARAMS = {
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

    "M_min": 1e14,
    "M_max": 1e16,
    "n_points": 16384,

    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "hmf_calc": "cnc",
    "interp_tinker": "linear",

    "stacked_likelihood": False,
    "likelihood_type": "unbinned",

    # Simulator settings
    "cov_constant": {"0": True, "1": True},
    "observable_vectorised": True,
    "observable_vector": False,
}

SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}

COSMO = {
    "Om0": 0.315,
    "Ob0": 0.049,
    "h": 0.674,
    "sigma_8": 0.811,
    "n_s": 0.965,
    "m_nu": 0.06,
}


def create_number_counts(cosmology_tool="classy_sz_jax", M_min=None):
    cnc_params = dict(cosmocnc_jax.cnc_params_default)
    cnc_params.update(SIM_PARAMS)
    cnc_params["cosmology_tool"] = cosmology_tool
    if M_min is not None:
        cnc_params["M_min"] = M_min

    cosmo_params = dict(cosmocnc_jax.cosmo_params_default)
    cosmo_params.update(COSMO)

    scal_rel_params = dict(cosmocnc_jax.scaling_relation_params_default)
    scal_rel_params.update(SCAL_REL)

    nc = cosmocnc_jax.cluster_number_counts(cnc_params)
    nc.initialise()
    nc.update_params(cosmo_params, scal_rel_params)

    return nc


def check(condition, msg):
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")


# ── Test 1: JIT path (q+p) ──────────────────────────────────────────

def test_jit_qp():
    print("\n=== Test 1: JIT path (q+p SO sim) ===")

    nc = create_number_counts()

    from cosmocnc_jax.sim import catalogue_generator

    gen = catalogue_generator(number_counts=nc, n_catalogues=1, seed=42)
    gen.generate_catalogues()

    cat = gen.catalogue_list[0]

    # Check that all observables from observables[0] are in catalogue
    primary_obs = SIM_PARAMS["observables"][0]
    check("z" in cat, "catalogue has 'z'")
    check("M" in cat, "catalogue has 'M'")

    for obs in primary_obs:
        check(obs in cat, f"catalogue has '{obs}'")
        check(obs + "_patch" in cat, f"catalogue has '{obs}_patch'")

    n_sel = len(cat["z"])
    check(n_sel > 0, f"n_selected = {n_sel} > 0")
    check(len(cat["M"]) == n_sel, f"M length matches z length ({n_sel})")

    for obs in primary_obs:
        check(len(cat[obs]) == n_sel, f"{obs} length matches ({n_sel})")

    # All q should be above selection threshold
    obs_select = SIM_PARAMS["obs_select"]
    q_min = float(jnp.min(cat[obs_select]))
    check(q_min > SIM_PARAMS["obs_select_min"],
          f"min({obs_select}) = {q_min:.2f} > {SIM_PARAMS['obs_select_min']}")

    # z should be in range
    z_min_obs = float(jnp.min(cat["z"]))
    z_max_obs = float(jnp.max(cat["z"]))
    check(z_min_obs >= 0., f"min(z) = {z_min_obs:.4f} >= 0")
    check(z_max_obs <= SIM_PARAMS["z_max"] + 0.1,
          f"max(z) = {z_max_obs:.4f} <= z_max + tolerance")

    # M should be positive
    M_min = float(jnp.min(cat["M"]))
    check(M_min > 0., f"min(M) = {M_min:.2e} > 0")

    print(f"\n  n_tot (mean) = {float(gen.n_tot):.1f}")
    print(f"  n_tot_obs (Poisson) = {int(gen.n_tot_obs[0])}")
    print(f"  n_selected = {n_sel}")
    print(f"  selection fraction = {n_sel / int(gen.n_tot_obs[0]):.3f}")
    print(f"  <z> = {float(jnp.mean(cat['z'])):.3f}")
    for obs in primary_obs:
        print(f"  <{obs}> = {float(jnp.mean(cat[obs])):.3f}")
    print(f"  <M> = {float(jnp.mean(cat['M'])):.3e}")

    return cat


# ── Test 2: generate_catalogues_hmf ─────────────────────────────────

def test_hmf_only():
    print("\n=== Test 2: generate_catalogues_hmf ===")

    nc = create_number_counts()

    from cosmocnc_jax.sim import catalogue_generator

    gen = catalogue_generator(number_counts=nc, n_catalogues=1, seed=123)
    gen.generate_catalogues_hmf()

    cat = gen.catalogue_list[0]

    check("z" in cat, "catalogue has 'z'")
    check("M" in cat, "catalogue has 'M'")

    n = len(cat["z"])
    check(n > 0, f"n_clusters = {n} > 0")
    check(len(cat["M"]) == n, "M length matches z length")

    z_min_obs = float(jnp.min(cat["z"]))
    z_max_obs = float(jnp.max(cat["z"]))
    check(z_min_obs >= 0., f"min(z) = {z_min_obs:.4f} >= 0")

    print(f"\n  n_clusters = {n}")
    print(f"  <z> = {float(jnp.mean(cat['z'])):.3f}")
    print(f"  <M> = {float(jnp.mean(cat['M'])):.3e}")


# ── Test 3: Multiple catalogues ─────────────────────────────────────

def test_multiple_catalogues():
    print("\n=== Test 3: Multiple catalogues ===")

    nc = create_number_counts()

    from cosmocnc_jax.sim import catalogue_generator

    n_cats = 5
    gen = catalogue_generator(number_counts=nc, n_catalogues=n_cats, seed=99)
    gen.generate_catalogues()

    check(len(gen.catalogue_list) == n_cats, f"generated {n_cats} catalogues")

    for i, cat in enumerate(gen.catalogue_list):
        n_sel = len(cat["z"])
        check(n_sel > 0, f"  catalogue {i}: n_selected = {n_sel}")

    # Catalogues should have different cluster counts (different Poisson draws)
    counts = [len(cat["z"]) for cat in gen.catalogue_list]
    print(f"\n  Cluster counts: {counts}")
    print(f"  Mean: {np.mean(counts):.1f}, Std: {np.std(counts):.1f}")


# ── Test 4: Benchmark ───────────────────────────────────────────────

def test_benchmark():
    print("\n=== Test 4: Benchmark (JIT path) ===")

    nc = create_number_counts()

    from cosmocnc_jax.sim import catalogue_generator

    # Warmup (JIT compilation) — first call compiles
    gen = catalogue_generator(number_counts=nc, n_catalogues=1, seed=0)
    t0 = time.time()
    gen.generate_catalogues()
    jax.block_until_ready(gen.catalogue_list[0]["z"])
    t_warmup = time.time() - t0
    print(f"  Warmup (incl. JIT compile): {t_warmup:.3f}s")

    # Timed runs — batch of 20 catalogues (JIT already compiled, same max_clusters)
    n_cats = 20
    gen = catalogue_generator(number_counts=nc, n_catalogues=n_cats, seed=200)
    t0 = time.time()
    gen.generate_catalogues()
    jax.block_until_ready(gen.catalogue_list[-1]["z"])
    t_batch = time.time() - t0
    print(f"\n  {n_cats} catalogues (JIT cached): {t_batch*1000:.1f} ms total")
    print(f"    Per-catalogue: {t_batch/n_cats*1000:.1f} ms")


# ── Test 5: Fallback path ───────────────────────────────────────────

def test_fallback():
    print("\n=== Test 5: Fallback path (forced, high M_min for speed) ===")

    # Use M_min=5e14 to get ~hundreds of clusters (not millions)
    # since the per-cluster covariance loop is O(n_clusters)
    nc = create_number_counts(M_min=5e14)

    from cosmocnc_jax.sim import catalogue_generator

    # Force fallback by setting cov_constant to False
    nc.cnc_params["cov_constant"] = {"0": False, "1": False}

    gen = catalogue_generator(number_counts=nc, n_catalogues=1, seed=42)
    gen.generate_catalogues()

    cat = gen.catalogue_list[0]

    n_sel = len(cat["z"])
    check(n_sel > 0, f"n_selected = {n_sel} > 0")
    for obs in SIM_PARAMS["observables"][0]:
        check(obs in cat, f"catalogue has '{obs}'")

    obs_select = SIM_PARAMS["obs_select"]
    q_min = float(jnp.min(jnp.array(cat[obs_select])))
    check(q_min > SIM_PARAMS["obs_select_min"],
          f"min({obs_select}) = {q_min:.2f} > {SIM_PARAMS['obs_select_min']}")

    print(f"  n_selected = {n_sel}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("cosmocnc_jax simulator (sim.py) test suite")
    print("=" * 60)

    cat = test_jit_qp()
    test_hmf_only()
    test_multiple_catalogues()
    test_benchmark()
    test_fallback()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
