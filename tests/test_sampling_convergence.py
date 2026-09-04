"""Test convergence of 2D logit-space sampling vs n_z.

Generates catalogues from the HMF at various n_z values and compares
binned (z, lnM) distributions to the theoretical prediction.
Reports bias in units of Poisson sigma per bin.

Usage:
    taskset -c 0-9 python tests/test_sampling_convergence.py
"""

import os
_N_THREADS = "10"
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = _N_THREADS
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from cosmocnc_jax.sim import catalogue_generator

# ── Config ──

COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}

SCAL_REL = {
    "corr_lnq_lnp": 0.,
    "bias_sz": 0.8,
    "dof": 0.,
}

BASE_CNC = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 0., "obs_select_max": 1e10,
    "z_min": 0.01, "z_max": 3.,
    "M_min": 1e14, "M_max": 1e16,
    "n_points": 8192,
    "sigma_mass_prior": 10,
    "hmf_type": "Tinker08", "hmf_calc": "cnc",
    "mass_definition": "500c",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
    "interp_tinker": "linear",
    "observables": [["q_so_sim"]],
    "stacked_likelihood": False,
    "cosmology_tool": "classy_sz_jax",
}

N_Z_VALUES = [5000, 2000, 1000, 500, 200]
N_POINTS_VALUES = [8192, 16384, 32768]
N_CATALOGUES = 50  # average over many to beat down Poisson noise
N_Z_BINS = 25
N_M_BINS = 20

z_bins_edges = np.linspace(0.01, 3., N_Z_BINS + 1)
ln_M_bins_edges = np.linspace(np.log(1), np.log(100), N_M_BINS + 1)  # M in units of 1e14 Msun
z_bins_centres = 0.5 * (z_bins_edges[:-1] + z_bins_edges[1:])
ln_M_bins_centres = 0.5 * (ln_M_bins_edges[:-1] + ln_M_bins_edges[1:])


def integrate_bins(bins_edges, x, dn_dx):
    """Integrate dn/dx in each bin using fine interpolation."""
    from scipy import integrate as sci_int
    n_x = np.zeros(len(bins_edges) - 1)
    for i in range(len(bins_edges) - 1):
        x_fine = np.linspace(bins_edges[i], bins_edges[i+1], 1000)
        dn_fine = np.interp(x_fine, x, dn_dx)
        n_x[i] = sci_int.simpson(dn_fine, x=x_fine)
    return n_x


def make_nc(n_z_val, n_points_val, reuse_nc=None):
    """Create or update a number_counts instance with given n_z and n_points."""
    if reuse_nc is None:
        nc = cosmocnc_jax.cluster_number_counts()
        p = dict(BASE_CNC)
        p["n_z"] = n_z_val
        p["n_points"] = n_points_val
        nc.cnc_params.update(p)
        nc.cosmo_params.update(COSMO)
        nc.scal_rel_params.update(SCAL_REL)
        nc.initialise()
    else:
        nc = reuse_nc
        nc.cnc_params["n_z"] = n_z_val
        nc.cnc_params["n_points"] = n_points_val
    nc.get_hmf()
    return nc


def run_for_nz(n_z_val, nc):
    """Run N_CATALOGUES catalogues at given n_z, return binned counts."""
    from scipy import integrate as sci_int

    print(f"\n{'='*60}")
    print(f"  n_z = {n_z_val}")
    print(f"{'='*60}")

    t0 = time.time()

    # Theoretical predictions from HMF
    sky_frac = np.sum(nc.scaling_relations[nc.cnc_params["obs_select"]].skyfracs)
    hmf_matrix = np.asarray(nc.hmf_matrix) * 4. * np.pi * sky_frac
    lnM = np.asarray(nc.ln_M)
    z = np.asarray(nc.redshift_vec)

    dn_dz = sci_int.simpson(hmf_matrix, x=lnM, axis=1)
    dn_dlnM = sci_int.simpson(hmf_matrix, x=z, axis=0)
    N_tot_theory = sci_int.simpson(dn_dz, x=z)

    n_z_theory = integrate_bins(z_bins_edges, z, dn_dz)
    n_M_theory = integrate_bins(ln_M_bins_edges, lnM, dn_dlnM)

    print(f"  N_tot (theory) = {N_tot_theory:.1f}")

    # Generate catalogues — time per single catalogue
    # Warmup: generate 1 catalogue to trigger JIT compilation
    gen_warmup = catalogue_generator(number_counts=nc, n_catalogues=1, seed=0)
    gen_warmup.generate_catalogues_hmf()

    # Time a single catalogue (post-JIT)
    t_single_start = time.time()
    gen_single = catalogue_generator(number_counts=nc, n_catalogues=1, seed=99)
    gen_single.generate_catalogues_hmf()
    t_single = time.time() - t_single_start

    # Now generate the full batch for statistics
    gen = catalogue_generator(number_counts=nc, n_catalogues=N_CATALOGUES, seed=42)
    gen.generate_catalogues_hmf()

    t_gen = time.time() - t0
    print(f"  Generated {N_CATALOGUES} catalogues in {t_gen:.1f}s")
    print(f"  Time per single catalogue (post-JIT): {t_single*1000:.1f} ms")

    # Bin the catalogues
    z_hists = np.zeros((N_CATALOGUES, N_Z_BINS))
    M_hists = np.zeros((N_CATALOGUES, N_M_BINS))

    for i, cat in enumerate(gen.catalogue_list):
        z_arr = np.asarray(cat["z"])
        lnM_arr = np.log(np.asarray(cat["M"]))
        z_hists[i], _ = np.histogram(z_arr, bins=z_bins_edges)
        M_hists[i], _ = np.histogram(lnM_arr, bins=ln_M_bins_edges)

    # Average over catalogues
    z_mean = np.mean(z_hists, axis=0)
    M_mean = np.mean(M_hists, axis=0)

    # Standard error of the mean (from catalogue-to-catalogue scatter)
    z_sem = np.std(z_hists, axis=0) / np.sqrt(N_CATALOGUES)
    M_sem = np.std(M_hists, axis=0) / np.sqrt(N_CATALOGUES)

    # Poisson sigma for a single catalogue
    z_poisson = np.sqrt(np.maximum(n_z_theory, 1.))
    M_poisson = np.sqrt(np.maximum(n_M_theory, 1.))

    # Bias = (mean sampled - theory), in units of single-catalogue Poisson sigma
    z_bias_sigma = (z_mean - n_z_theory) / z_poisson
    M_bias_sigma = (M_mean - n_M_theory) / M_poisson

    # Also express significance of the bias itself (bias / SEM)
    z_bias_signif = np.where(z_sem > 0, (z_mean - n_z_theory) / z_sem, 0.)
    M_bias_signif = np.where(M_sem > 0, (M_mean - n_M_theory) / M_sem, 0.)

    # Debug: print mass ranges
    print(f"\n  Mass debug: nc.ln_M range = [{lnM[0]:.4f}, {lnM[-1]:.4f}]")
    print(f"  Mass debug: histogram bin range = [{ln_M_bins_edges[0]:.4f}, {ln_M_bins_edges[-1]:.4f}]")
    if len(gen.catalogue_list) > 0:
        cat0_lnM = np.log(np.asarray(gen.catalogue_list[0]["M"]))
        print(f"  Mass debug: catalogue lnM range = [{np.min(cat0_lnM):.4f}, {np.max(cat0_lnM):.4f}]")
    print(f"  Mass debug: dn_dlnM range = [{np.min(dn_dlnM):.6e}, {np.max(dn_dlnM):.6e}]")
    print(f"  Mass debug: theory bins = {n_M_theory[:3]} ... {n_M_theory[-3:]}")
    print(f"  Mass debug: sampled mean = {M_mean[:3]} ... {M_mean[-3:]}")
    print(f"  Mass debug: total theory = {np.sum(n_M_theory):.1f}, sampled = {np.sum(M_mean):.1f}")

    print(f"\n  Redshift bins:")
    print(f"    Max |bias/Poisson_sigma|  = {np.max(np.abs(z_bias_sigma)):.4f}")
    print(f"    Mean |bias/Poisson_sigma| = {np.mean(np.abs(z_bias_sigma)):.4f}")
    print(f"    Max |bias/SEM|            = {np.max(np.abs(z_bias_signif)):.2f}")

    print(f"\n  Mass bins:")
    print(f"    Max |bias/Poisson_sigma|  = {np.max(np.abs(M_bias_sigma)):.4f}")
    print(f"    Mean |bias/Poisson_sigma| = {np.mean(np.abs(M_bias_sigma)):.4f}")
    print(f"    Max |bias/SEM|            = {np.max(np.abs(M_bias_signif)):.2f}")

    return {
        "n_z": n_z_val,
        "z_bias_sigma": z_bias_sigma,
        "M_bias_sigma": M_bias_sigma,
        "z_bias_signif": z_bias_signif,
        "M_bias_signif": M_bias_signif,
        "z_mean": z_mean,
        "M_mean": M_mean,
        "n_z_theory": n_z_theory,
        "n_M_theory": n_M_theory,
        "N_tot_theory": N_tot_theory,
        "time": t_gen,
        "time_single": t_single,
    }


def main():
    print("Sampling convergence test: logit-space 2D inverse CDF")
    print(f"N_CATALOGUES = {N_CATALOGUES}, N_Z_BINS = {N_Z_BINS}, N_M_BINS = {N_M_BINS}")
    print(f"n_z values: {N_Z_VALUES}")
    print(f"n_points values: {N_POINTS_VALUES}")

    # results[(n_z, n_points)] = {...}
    results = {}
    nc = None
    for n_pts in N_POINTS_VALUES:
        for n_z_val in N_Z_VALUES:
            key = (n_z_val, n_pts)
            print(f"\n>>> Running n_z={n_z_val}, n_points={n_pts}")
            nc = make_nc(n_z_val, n_pts, reuse_nc=nc)
            results[key] = run_for_nz(n_z_val, nc)
            results[key]['n_points'] = n_pts

    # ── Summary table ──
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"{'n_pts':>7}  {'n_z':>6}  {'max|z_bias/σ|':>14}  {'mean|z_bias/σ|':>15}  "
          f"{'max|M_bias/σ|':>14}  {'mean|M_bias/σ|':>15}  {'t/cat(ms)':>10}")
    print("-" * 95)

    for n_pts in N_POINTS_VALUES:
        for n_z_val in sorted(N_Z_VALUES):
            r = results[(n_z_val, n_pts)]
            print(f"{n_pts:>7}  {n_z_val:>6}  "
                  f"{np.max(np.abs(r['z_bias_sigma'])):>14.4f}  "
                  f"{np.mean(np.abs(r['z_bias_sigma'])):>15.4f}  "
                  f"{np.max(np.abs(r['M_bias_sigma'])):>14.4f}  "
                  f"{np.mean(np.abs(r['M_bias_sigma'])):>15.4f}  "
                  f"{r['time_single']*1000:>10.1f}")
        print("-" * 95)

    print("σ = single-catalogue Poisson sigma. Bias << 1σ means negligible.")

    # ── Plots ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_npts = {8192: 'tab:blue', 16384: 'tab:orange', 32768: 'tab:green'}

    # Top-left: max |z_bias/sigma| vs n_z for each n_points
    ax = axes[0, 0]
    for n_pts in N_POINTS_VALUES:
        nz_sorted = sorted(N_Z_VALUES)
        vals = [np.max(np.abs(results[(nz, n_pts)]['z_bias_sigma'])) for nz in nz_sorted]
        ax.plot(nz_sorted, vals, 'o-', color=colors_npts[n_pts], label=f'n_pts={n_pts}')
    ax.axhline(0.1, ls='--', color='gray', label='0.1σ threshold')
    ax.set_xlabel('n_z')
    ax.set_ylabel('max |z bias / Poisson σ|')
    ax.set_title('Redshift bias convergence')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: max |M_bias/sigma| vs n_z for each n_points
    ax = axes[0, 1]
    for n_pts in N_POINTS_VALUES:
        nz_sorted = sorted(N_Z_VALUES)
        vals = [np.max(np.abs(results[(nz, n_pts)]['M_bias_sigma'])) for nz in nz_sorted]
        ax.plot(nz_sorted, vals, 's-', color=colors_npts[n_pts], label=f'n_pts={n_pts}')
    ax.axhline(0.1, ls='--', color='gray', label='0.1σ threshold')
    ax.set_xlabel('n_z')
    ax.set_ylabel('max |M bias / Poisson σ|')
    ax.set_title('Mass bias convergence')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: z bias profile at highest n_z for each n_points
    ax = axes[1, 0]
    nz_best = max(N_Z_VALUES)
    for n_pts in N_POINTS_VALUES:
        r = results[(nz_best, n_pts)]
        ax.plot(z_bins_centres, r['z_bias_sigma'],
                color=colors_npts[n_pts], label=f'n_pts={n_pts}')
    ax.axhline(0, ls='-', color='k', lw=0.5)
    ax.axhline(0.1, ls='--', color='gray', alpha=0.5)
    ax.axhline(-0.1, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('z')
    ax.set_ylabel('bias / Poisson σ')
    ax.set_title(f'Redshift bias profile (n_z={nz_best})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: mass bias profile at highest n_z for each n_points
    ax = axes[1, 1]
    for n_pts in N_POINTS_VALUES:
        r = results[(nz_best, n_pts)]
        ax.plot(ln_M_bins_centres, r['M_bias_sigma'],
                color=colors_npts[n_pts], label=f'n_pts={n_pts}')
    ax.axhline(0, ls='-', color='k', lw=0.5)
    ax.axhline(0.1, ls='--', color='gray', alpha=0.5)
    ax.axhline(-0.1, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('ln M')
    ax.set_ylabel('bias / Poisson σ')
    ax.set_title(f'Mass bias profile (n_z={nz_best})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = '/scratch/scratch-izubeldia/planck_cosmology/figures/sampling_convergence.pdf'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    print(f"\nPlot saved to {outpath}")


if __name__ == "__main__":
    main()
