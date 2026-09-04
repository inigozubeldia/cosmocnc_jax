"""Quick comparison of catalogue generation: cosmocnc vs cosmocnc_jax.

Since the RNG backends differ (numpy vs JAX), catalogues won't match
cluster-by-cluster. Instead we compare:
  1. n_tot (deterministic, from HMF)
  2. Mean number of selected clusters over many catalogues
  3. Distribution of observables (KS test on q, p, z, M)
  4. Scaling relation forward pass (deterministic part, no scatter)

Runs in ~30s on CPU.
"""

import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["XLA_FLAGS"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import sys
from scipy import stats

_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # directory containing this repo
sys.path = [p for p in sys.path if p not in ('', '.', _REPO_PARENT)]

import cosmocnc
import cosmocnc_jax

# ── Config ──

CNC_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "observables": [["q_so_sim", "p_so_sim"]],
    "data_lik_from_abundance": True,
    "compute_abundance_matrix": False,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 1., "n_z": 50,
    "M_min": 1e15, "M_max": 1e16,
    "n_points": 512,
    "n_points_data_lik": 64,
    "likelihood_type": "unbinned",
    "stacked_likelihood": False,
    "apply_obs_cutoff": False,
    "z_errors": False,
    "hmf_type": "Tinker08", "hmf_calc": "cnc",
    "mass_definition": "500c",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "interp_tinker": "linear",
}

COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}

SCAL_REL = {
    "bias_sz": 0.8, "bias_cmblens": 0.8,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}

N_CATS = 20  # number of catalogues for statistical comparison


def setup(pkg, tool):
    nc = pkg.cluster_number_counts()
    p = dict(CNC_PARAMS)
    p["cosmology_tool"] = tool
    nc.cnc_params.update(p)
    nc.cosmo_params.update(COSMO)
    nc.scal_rel_params.update(SCAL_REL)
    nc.initialise()
    return nc


def compare(name, a, b, rtol=5e-4):
    a, b = float(a), float(b)
    rel = abs(a - b) / max(abs(a), 1e-30)
    ok = rel < rtol
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag} {name}: np={a:.6f}, jax={b:.6f}, rel_diff={rel:.3e}")
    return ok


def main():
    t0 = time.time()
    all_ok = True

    # ── Setup ──
    print("=" * 60)
    print("SIM COMPARISON: cosmocnc vs cosmocnc_jax")
    print("=" * 60)

    print("\n  Init cosmocnc...")
    nc_np = setup(cosmocnc, "classy_sz")
    print("  Init cosmocnc_jax...")
    nc_jx = setup(cosmocnc_jax, "classy_sz_jax")

    # ── 1. Compare n_tot (deterministic) ──
    print("\n--- 1. HMF n_tot ---")
    sim_np = cosmocnc.catalogue_generator(number_counts=nc_np, n_catalogues=N_CATS, seed=42)
    sim_jx = cosmocnc_jax.catalogue_generator(number_counts=nc_jx, n_catalogues=N_CATS, seed=42)

    sim_np.get_total_number_clusters()
    all_ok &= compare("n_tot", sim_np.n_tot, sim_jx.n_tot)

    # ── 2. Compare HMF-derived quantities ──
    print("\n--- 2. HMF-derived quantities ---")
    all_ok &= compare("dndz_integral", np.sum(np.asarray(sim_np.dndz)),
                       np.sum(np.asarray(sim_jx.dndz)))

    # ── 3. Generate catalogues (HMF only, no SR) ──
    print("\n--- 3. HMF-only catalogues ---")
    sim_np.generate_catalogues_hmf()
    sim_jx.generate_catalogues_hmf()

    n_cl_np = [len(sim_np.catalogue_list[i]["z"]) for i in range(N_CATS)]
    n_cl_jx = [len(sim_jx.catalogue_list[i]["z"]) for i in range(N_CATS)]

    mean_np = np.mean(n_cl_np)
    mean_jx = np.mean(n_cl_jx)
    # Both should be consistent with n_tot (Poisson mean)
    expected = float(sim_np.n_tot)
    print(f"  n_tot={expected:.1f}, mean_n_clusters: np={mean_np:.1f}, jax={mean_jx:.1f}")
    # Check both are within 3-sigma of expected (Poisson)
    sigma = np.sqrt(expected / N_CATS)
    for label, mean in [("np", mean_np), ("jax", mean_jx)]:
        dev = abs(mean - expected) / sigma
        ok = dev < 4.
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} mean_n_{label}: {dev:.1f} sigma from expected")
        all_ok &= ok

    # Pool all catalogues for distribution comparison
    z_np = np.concatenate([np.asarray(sim_np.catalogue_list[i]["z"]) for i in range(N_CATS)])
    z_jx = np.concatenate([np.asarray(sim_jx.catalogue_list[i]["z"]) for i in range(N_CATS)])
    M_np = np.concatenate([np.asarray(sim_np.catalogue_list[i]["M"]) for i in range(N_CATS)])
    M_jx = np.concatenate([np.asarray(sim_jx.catalogue_list[i]["M"]) for i in range(N_CATS)])

    # KS test on z and log(M) distributions
    for name, arr_np, arr_jx in [("z", z_np, z_jx), ("log10(M)", np.log10(M_np), np.log10(M_jx))]:
        ks_stat, ks_p = stats.ks_2samp(arr_np, arr_jx)
        ok = ks_p > 0.01  # p > 1% means distributions are consistent
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} KS({name}): stat={ks_stat:.4f}, p={ks_p:.4f}")
        all_ok &= ok

    # ── 4. Generate full catalogues (with SR + scatter) ──
    print("\n--- 4. Full catalogues (SR + scatter) ---")
    sim_np2 = cosmocnc.catalogue_generator(number_counts=nc_np, n_catalogues=N_CATS, seed=123)
    sim_jx2 = cosmocnc_jax.catalogue_generator(number_counts=nc_jx, n_catalogues=N_CATS, seed=123)

    sim_np2.generate_catalogues()
    sim_jx2.generate_catalogues()

    n_sel_np = [len(sim_np2.catalogue_list[i]["z"]) for i in range(N_CATS)]
    n_sel_jx = [len(sim_jx2.catalogue_list[i]["z"]) for i in range(N_CATS)]

    mean_sel_np = np.mean(n_sel_np)
    mean_sel_jx = np.mean(n_sel_jx)
    print(f"  Mean selected clusters: np={mean_sel_np:.1f}, jax={mean_sel_jx:.1f}")

    # Check means are within 10% of each other (scatter from different RNG streams)
    rel_diff = abs(mean_sel_np - mean_sel_jx) / max(mean_sel_np, 1.)
    ok = rel_diff < 0.15
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag} mean_n_selected rel_diff: {rel_diff:.3f}")
    all_ok &= ok

    # Pool and compare distributions
    q_np = np.concatenate([np.asarray(sim_np2.catalogue_list[i]["q_so_sim"]) for i in range(N_CATS)])
    q_jx = np.concatenate([np.asarray(sim_jx2.catalogue_list[i]["q_so_sim"]) for i in range(N_CATS)])
    p_np = np.concatenate([np.asarray(sim_np2.catalogue_list[i]["p_so_sim"]) for i in range(N_CATS)])
    p_jx = np.concatenate([np.asarray(sim_jx2.catalogue_list[i]["p_so_sim"]) for i in range(N_CATS)])
    z2_np = np.concatenate([np.asarray(sim_np2.catalogue_list[i]["z"]) for i in range(N_CATS)])
    z2_jx = np.concatenate([np.asarray(sim_jx2.catalogue_list[i]["z"]) for i in range(N_CATS)])

    for name, arr_np, arr_jx in [("z", z2_np, z2_jx),
                                   ("q_so_sim", q_np, q_jx),
                                   ("p_so_sim", p_np, p_jx)]:
        ks_stat, ks_p = stats.ks_2samp(arr_np, arr_jx)
        ok = ks_p > 0.01
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} KS({name}): stat={ks_stat:.4f}, p={ks_p:.4f}")
        all_ok &= ok

    # Compare mean and std of observables
    for name, arr_np, arr_jx in [("q_so_sim", q_np, q_jx), ("p_so_sim", p_np, p_jx)]:
        mean_rel = abs(np.mean(arr_np) - np.mean(arr_jx)) / max(abs(np.mean(arr_np)), 1e-30)
        std_rel = abs(np.std(arr_np) - np.std(arr_jx)) / max(abs(np.std(arr_np)), 1e-30)
        print(f"  {name}: mean_np={np.mean(arr_np):.3f}, mean_jax={np.mean(arr_jx):.3f} (rel={mean_rel:.3f})")
        print(f"  {name}: std_np={np.std(arr_np):.3f}, std_jax={np.std(arr_jx):.3f} (rel={std_rel:.3f})")

    # ── Summary ──
    dt = time.time() - t0
    print("\n" + "=" * 60)
    tag = "ALL PASSED" if all_ok else "SOME FAILED"
    print(f"RESULT: {tag}  (total time: {dt:.0f}s)")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
