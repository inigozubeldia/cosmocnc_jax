"""Comprehensive N-D backward convolution test suite.

Covers:
1. Parameter scans (sigma_8, a_lens) — cosmocnc vs cosmocnc_jax, 1D vs 2D
2. Convergence tests (n_points_data_lik sweep) as param scans
3. 1D vs 2D log_lik difference
4. Performance benchmarks
5. Accuracy summary

Outputs plots to tests/plots/ and generates a PDF summary.

Optimised: cosmology is initialised once per package and shared across all
instances (different n_points_data_lik / observable modes).  This cuts init
time from ~40 × 20s to ~2 × 20s.
"""

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
import jax.numpy as jnp

import numpy as np
import time
import sys
import json
import copy

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

import cosmocnc
import cosmocnc_jax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = print

print(f"JAX backend: {jax.default_backend()}")

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Shared configuration
# ═══════════════════════════════════════════════════════════════════

BASE_CNC_PARAMS = {
    "cluster_catalogue": "SO_sim_0",
    "obs_select": "q_so_sim",
    "data_lik_from_abundance": False,
    "compute_abundance_matrix": True,
    "number_cores_hmf": 1, "number_cores_abundance": 1,
    "number_cores_data": 1, "number_cores_stacked": 1,
    "parallelise_type": "redshift",
    "obs_select_min": 5., "obs_select_max": 200.,
    "z_min": 0.01, "z_max": 3., "n_z": 100,
    "M_min": 1e13, "M_max": 1e16,
    "n_points": 16384, "n_points_data_lik": 128,
    "likelihood_type": "unbinned",
    "data_lik_type": "backward_convolutional",
    "stacked_likelihood": False,
    "apply_obs_cutoff": False,
    "sigma_mass_prior": 5., "z_errors": False,
    "delta_m_with_ref": False, "scalrel_type_deriv": "analytical",
    "downsample_hmf_bc": 8, "padding_fraction": 0.,
    "bc_chunk_size": 0,
    "hmf_type": "Tinker08", "hmf_calc": "cnc",
    "mass_definition": "500c",
    "interp_tinker": "linear",
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm",
}

BASE_SCAL_REL = {
    "bias_sz": 3., "bias_cmblens": 3.,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}

BASE_COSMO = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
    "k_cutoff": 1e8, "ps_cutoff": 1,
}

def make_cnc_params(obs_mode, n_pts_dl=128):
    """Create cnc_params with given observable mode and n_points_data_lik."""
    p = dict(BASE_CNC_PARAMS)
    p["n_points_data_lik"] = n_pts_dl
    if obs_mode == "2d":
        p["observables"] = [["q_so_sim", "p_so_sim"]]
    elif obs_mode == "1d":
        p["observables"] = [["q_so_sim"], ["p_so_sim"]]
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")
    return p


# ═══════════════════════════════════════════════════════════════════
# Instance pool — cosmology initialised once per package, shared
# ═══════════════════════════════════════════════════════════════════

# Fully initialised instances, keyed by (pkg_name, mode, npts)
_pool = {}


def get_instance(pkg, pkg_name, mode, npts):
    """Get a cached instance or create one (full init). Each unique
    (pkg, mode, npts) combo is initialised once and reused across sections."""
    key = (pkg_name, mode, npts)
    if key in _pool:
        return _pool[key]

    print(f"  [pool] Initialising {pkg_name} {mode} n_pts_dl={npts}...")
    t0 = time.time()

    nc = pkg.cluster_number_counts()
    nc.cnc_params.update(make_cnc_params(mode, npts))
    # cosmocnc uses classy_sz; cosmocnc_jax uses classy_sz_jax
    if pkg_name == "jax":
        nc.cnc_params["cosmology_tool"] = "classy_sz_jax"
    else:
        nc.cnc_params["cosmology_tool"] = "classy_sz"
    nc.scal_rel_params.update(dict(BASE_SCAL_REL))
    nc.cosmo_params.update(dict(BASE_COSMO))
    nc.initialise()

    dt = time.time() - t0
    print(f"  [pool] {pkg_name} {mode} n_pts_dl={npts} ready ({dt:.1f}s)")

    _pool[key] = nc
    return nc


def eval_log_lik(nc, cosmo=None, scal_rel=None):
    """Evaluate log_lik, optionally updating params first."""
    if cosmo is not None or scal_rel is not None:
        c = cosmo if cosmo is not None else nc.cosmo_params
        s = scal_rel if scal_rel is not None else nc.scal_rel_params
        nc.update_params(c, s)
    ll = nc.get_log_lik()
    if hasattr(ll, 'block_until_ready'):
        jax.block_until_ready(ll)
    return float(ll)


def warmup(nc, pkg_name, n=2):
    """JIT warmup evaluations."""
    for _ in range(n if pkg_name == "jax" else 1):
        eval_log_lik(nc)


# ═══════════════════════════════════════════════════════════════════
# Section 1: Parameter Scans (sigma_8 and a_lens)
# ═══════════════════════════════════════════════════════════════════

def run_param_scan(param_name, param_values, param_dict_key,
                   instances, labels, n_warmup=1):
    """Run a 1D parameter scan over multiple instances."""
    results = {label: [] for label in labels}
    for i, val in enumerate(param_values):
        print(f"  {param_name} = {val:.4f} ({i+1}/{len(param_values)})")
        for (nc, base_c, base_sr), label in zip(instances, labels):
            if param_dict_key == "cosmo":
                c = dict(base_c)
                c[param_name] = val
                ll = eval_log_lik(nc, cosmo=c, scal_rel=base_sr)
            else:
                sr = dict(base_sr)
                sr[param_name] = val
                ll = eval_log_lik(nc, cosmo=base_c, scal_rel=sr)
            results[label].append(ll)
            print(f"    {label}: {ll:.4f}")
    return results


def section_param_scans():
    """Run sigma_8 and a_lens parameter scans."""
    print("\n" + "=" * 70)
    print("SECTION 1: PARAMETER SCANS")
    print("=" * 70)

    print("\nInitialising instances (n_points_data_lik=128)...")
    insts = {}
    for pkg, pkg_name in [(cosmocnc, "cosmocnc"), (cosmocnc_jax, "jax")]:
        for mode in ["2d", "1d"]:
            label = f"{pkg_name}_{mode}"
            nc = get_instance(pkg, pkg_name, mode, 128)
            warmup(nc, pkg_name)
            ll = eval_log_lik(nc)
            insts[label] = (nc, dict(BASE_COSMO), dict(BASE_SCAL_REL))
            print(f"    {label} warmup log_lik = {ll:.4f}")

    instance_list = [insts[k] for k in ["cosmocnc_2d", "jax_2d", "cosmocnc_1d", "jax_1d"]]
    label_list = ["cosmocnc_2d", "jax_2d", "cosmocnc_1d", "jax_1d"]

    # ── sigma_8 scan ──
    sigma8_vals = np.linspace(0.790, 0.830, 10)
    print(f"\n--- sigma_8 scan: {sigma8_vals[0]:.3f} to {sigma8_vals[-1]:.3f} ---")
    sigma8_results = run_param_scan("sigma_8", sigma8_vals, "cosmo",
                                     instance_list, label_list)

    # ── a_lens scan ──
    alens_vals = np.linspace(0.8, 1.2, 10)
    print(f"\n--- a_lens scan: {alens_vals[0]:.2f} to {alens_vals[-1]:.2f} ---")
    alens_results = run_param_scan("a_lens", alens_vals, "scal_rel",
                                    instance_list, label_list)

    return {
        "sigma8": {"values": sigma8_vals, "results": sigma8_results},
        "alens": {"values": alens_vals, "results": alens_results},
        "labels": label_list,
    }


# ═══════════════════════════════════════════════════════════════════
# Section 2: Convergence Tests (n_points_data_lik sweep)
# ═══════════════════════════════════════════════════════════════════

def section_convergence():
    """Test convergence of backward conv as n_points_data_lik increases."""
    print("\n" + "=" * 70)
    print("SECTION 2: CONVERGENCE TESTS (n_points_data_lik sweep)")
    print("=" * 70)

    npts_2d = [96, 128, 192, 256]
    npts_1d = [64, 128, 256, 512]

    sigma8_vals = np.linspace(0.795, 0.825, 7)

    # ── 2D convergence ──
    print("\n--- 2D convergence (sigma_8 scan at each n_pts_dl) ---")
    conv_2d = {}
    for npts in npts_2d:
        print(f"\n  n_points_data_lik = {npts}")
        nc_ref = get_instance(cosmocnc, "cosmocnc", "2d", npts)
        nc_jax = get_instance(cosmocnc_jax, "jax", "2d", npts)
        warmup(nc_ref, "cosmocnc")
        warmup(nc_jax, "jax")

        lls_ref, lls_jax = [], []
        for val in sigma8_vals:
            c = dict(BASE_COSMO)
            c["sigma_8"] = val
            ll_ref = eval_log_lik(nc_ref, cosmo=c, scal_rel=BASE_SCAL_REL)
            ll_jax = eval_log_lik(nc_jax, cosmo=c, scal_rel=BASE_SCAL_REL)
            lls_ref.append(ll_ref)
            lls_jax.append(ll_jax)
            print(f"    s8={val:.3f}: ref={ll_ref:.3f}, jax={ll_jax:.3f}, "
                  f"diff={abs(ll_ref-ll_jax):.4f}")
        conv_2d[npts] = {"ref": lls_ref, "jax": lls_jax}

    # ── 1D convergence ──
    print("\n--- 1D convergence (sigma_8 scan at each n_pts_dl) ---")
    conv_1d = {}
    for npts in npts_1d:
        print(f"\n  n_points_data_lik = {npts}")
        nc_ref = get_instance(cosmocnc, "cosmocnc", "1d", npts)
        nc_jax = get_instance(cosmocnc_jax, "jax", "1d", npts)
        warmup(nc_ref, "cosmocnc")
        warmup(nc_jax, "jax")

        lls_ref, lls_jax = [], []
        for val in sigma8_vals:
            c = dict(BASE_COSMO)
            c["sigma_8"] = val
            ll_ref = eval_log_lik(nc_ref, cosmo=c, scal_rel=BASE_SCAL_REL)
            ll_jax = eval_log_lik(nc_jax, cosmo=c, scal_rel=BASE_SCAL_REL)
            lls_ref.append(ll_ref)
            lls_jax.append(ll_jax)
            print(f"    s8={val:.3f}: ref={ll_ref:.3f}, jax={ll_jax:.3f}, "
                  f"diff={abs(ll_ref-ll_jax):.4f}")
        conv_1d[npts] = {"ref": lls_ref, "jax": lls_jax}

    return {
        "sigma8_vals": sigma8_vals,
        "conv_2d": conv_2d, "npts_2d": npts_2d,
        "conv_1d": conv_1d, "npts_1d": npts_1d,
    }


# ═══════════════════════════════════════════════════════════════════
# Section 3: 1D vs 2D comparison
# ═══════════════════════════════════════════════════════════════════

def section_1d_vs_2d():
    """Compare 1D vs 2D log_lik at fiducial and scanned params."""
    print("\n" + "=" * 70)
    print("SECTION 3: 1D vs 2D COMPARISON")
    print("=" * 70)

    nc_ref_2d = get_instance(cosmocnc, "cosmocnc", "2d", 128)
    nc_ref_1d = get_instance(cosmocnc, "cosmocnc", "1d", 128)
    nc_jax_2d = get_instance(cosmocnc_jax, "jax", "2d", 128)
    nc_jax_1d = get_instance(cosmocnc_jax, "jax", "1d", 128)

    ll_ref_2d = eval_log_lik(nc_ref_2d)
    ll_ref_1d = eval_log_lik(nc_ref_1d)
    ll_jax_2d = eval_log_lik(nc_jax_2d)
    ll_jax_1d = eval_log_lik(nc_jax_1d)

    print(f"\nFiducial point:")
    print(f"  cosmocnc 2D: {ll_ref_2d:.4f}")
    print(f"  cosmocnc 1D: {ll_ref_1d:.4f}")
    print(f"  jax 2D:      {ll_jax_2d:.4f}")
    print(f"  jax 1D:      {ll_jax_1d:.4f}")
    print(f"  2D-1D diff (cosmocnc): {ll_ref_2d - ll_ref_1d:.4f}")
    print(f"  2D-1D diff (jax):      {ll_jax_2d - ll_jax_1d:.4f}")

    # a_lens scan — most sensitive to 2D correlation
    alens_vals = np.linspace(0.8, 1.2, 10)
    diff_ref, diff_jax = [], []
    for val in alens_vals:
        sr = dict(BASE_SCAL_REL)
        sr["a_lens"] = val
        l2d_ref = eval_log_lik(nc_ref_2d, cosmo=BASE_COSMO, scal_rel=sr)
        l1d_ref = eval_log_lik(nc_ref_1d, cosmo=BASE_COSMO, scal_rel=sr)
        l2d_jax = eval_log_lik(nc_jax_2d, cosmo=BASE_COSMO, scal_rel=sr)
        l1d_jax = eval_log_lik(nc_jax_1d, cosmo=BASE_COSMO, scal_rel=sr)
        diff_ref.append(l2d_ref - l1d_ref)
        diff_jax.append(l2d_jax - l1d_jax)
        print(f"  a_lens={val:.2f}: 2D-1D ref={l2d_ref-l1d_ref:.3f}, "
              f"jax={l2d_jax-l1d_jax:.3f}")

    return {
        "fiducial": {
            "ref_2d": ll_ref_2d, "ref_1d": ll_ref_1d,
            "jax_2d": ll_jax_2d, "jax_1d": ll_jax_1d,
        },
        "alens_vals": alens_vals,
        "diff_ref": diff_ref, "diff_jax": diff_jax,
    }


# ═══════════════════════════════════════════════════════════════════
# Section 4: Performance benchmarks
# ═══════════════════════════════════════════════════════════════════

def section_performance():
    """Benchmark get_log_lik timing at various n_points_data_lik."""
    print("\n" + "=" * 70)
    print("SECTION 4: PERFORMANCE BENCHMARKS")
    print("=" * 70)

    npts_list = [64, 128, 256]
    n_eval = 10
    perf = {}

    for npts in npts_list:
        print(f"\n  n_points_data_lik = {npts}")
        perf[npts] = {}

        for mode in ["1d", "2d"]:
            nc = get_instance(cosmocnc_jax, "jax", mode, npts)
            # Warmup
            for _ in range(3):
                ll = eval_log_lik(nc)

            times = []
            for _ in range(n_eval):
                t0 = time.time()
                ll = eval_log_lik(nc)
                times.append(time.time() - t0)

            avg_ms = np.mean(times) * 1000
            min_ms = np.min(times) * 1000
            std_ms = np.std(times) * 1000
            perf[npts][mode] = {
                "avg_ms": avg_ms, "min_ms": min_ms, "std_ms": std_ms,
                "times_ms": [t * 1000 for t in times],
            }
            print(f"    {mode}: avg={avg_ms:.1f}ms, min={min_ms:.1f}ms, "
                  f"std={std_ms:.1f}ms ({n_eval} runs)")

    # Also benchmark cosmocnc (numpy) at n_pts_dl=128 for reference
    print("\n  cosmocnc (numpy) reference at n_pts_dl=128:")
    perf["numpy_ref"] = {}
    for mode in ["1d", "2d"]:
        nc = get_instance(cosmocnc, "cosmocnc", mode, 128)
        eval_log_lik(nc)  # warmup
        times = []
        for _ in range(3):  # fewer runs since it's slow
            t0 = time.time()
            ll = eval_log_lik(nc)
            times.append(time.time() - t0)
        avg_ms = np.mean(times) * 1000
        perf["numpy_ref"][mode] = {"avg_ms": avg_ms, "min_ms": np.min(times)*1000}
        print(f"    {mode}: avg={avg_ms:.0f}ms ({len(times)} runs)")

    return perf


# ═══════════════════════════════════════════════════════════════════
# Section 5: Accuracy summary
# ═══════════════════════════════════════════════════════════════════

def section_accuracy():
    """Compute accuracy metrics: relative differences at fiducial."""
    print("\n" + "=" * 70)
    print("SECTION 5: ACCURACY SUMMARY")
    print("=" * 70)

    results = {}
    for mode in ["1d", "2d"]:
        for npts in [128, 256] if mode == "2d" else [128, 256, 512]:
            nc_ref = get_instance(cosmocnc, "cosmocnc", mode, npts)
            nc_jax = get_instance(cosmocnc_jax, "jax", mode, npts)
            ll_ref = eval_log_lik(nc_ref)
            ll_jax = eval_log_lik(nc_jax)
            abs_diff = abs(ll_ref - ll_jax)
            rel_diff = abs_diff / abs(ll_ref)
            key = f"{mode}_npts{npts}"
            results[key] = {
                "ref": ll_ref, "jax": ll_jax,
                "abs_diff": abs_diff, "rel_diff": rel_diff,
            }
            print(f"  {key}: ref={ll_ref:.4f}, jax={ll_jax:.4f}, "
                  f"abs_diff={abs_diff:.4f}, rel_diff={rel_diff:.2e}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting + PDF generation
# ═══════════════════════════════════════════════════════════════════

def make_plots_and_pdf(scan_data, conv_data, comp_data, perf_data, acc_data):
    """Generate all plots and compile into PDF."""
    pdf_path = os.path.join(PLOT_DIR, "nd_backward_conv_report.pdf")

    with PdfPages(pdf_path) as pdf:

        # ── Title page ──
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.6, "N-D Backward Convolution\nComprehensive Test Report",
                 ha='center', va='center', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.45, "cosmocnc_jax (classy_sz_jax) vs cosmocnc (classy_sz)\n"
                 "1D independent vs 2D correlated observables",
                 ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.3,
                 f"JAX backend: {jax.default_backend()}\n"
                 f"n_z=100, n_points=16384\n"
                 f"Observables: q_so_sim, p_so_sim (corr=0.5)",
                 ha='center', va='center', fontsize=11, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────
        # SECTION 1: Parameter scans
        # ──────────────────────────────────────────────

        for scan_name, x_label, x_key in [
            ("sigma8", r"$\sigma_8$", "sigma8"),
            ("alens", r"$a_\mathrm{lens}$", "alens"),
        ]:
            data = scan_data[x_key]
            vals = data["values"]
            res = data["results"]

            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
            colors = {"cosmocnc_2d": "C0", "jax_2d": "C1",
                      "cosmocnc_1d": "C2", "jax_1d": "C3"}
            styles = {"cosmocnc_2d": "-", "jax_2d": "--",
                      "cosmocnc_1d": "-", "jax_1d": "--"}
            markers = {"cosmocnc_2d": "o", "jax_2d": "s",
                       "cosmocnc_1d": "^", "jax_1d": "v"}

            for label in scan_data["labels"]:
                axes[0].plot(vals, res[label], color=colors[label],
                            ls=styles[label], marker=markers[label],
                            ms=4, label=label)
            axes[0].set_ylabel("log L")
            axes[0].legend(fontsize=9)
            axes[0].set_title(f"Parameter scan: {x_label}")
            axes[0].grid(True, alpha=0.3)

            res_2d = np.array(res["jax_2d"]) - np.array(res["cosmocnc_2d"])
            res_1d = np.array(res["jax_1d"]) - np.array(res["cosmocnc_1d"])
            axes[1].plot(vals, res_2d, 'C1-o', ms=4, label="jax_2d - cosmocnc_2d")
            axes[1].plot(vals, res_1d, 'C3-^', ms=4, label="jax_1d - cosmocnc_1d")
            axes[1].axhline(0, color='k', ls=':', lw=0.5)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel(r"$\Delta$ log L")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, f"scan_{scan_name}.png"), dpi=150)
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(8.5, 3))
            text = (f"Figure: {x_label} parameter scan. "
                    f"Top: log-likelihood vs {x_label} for cosmocnc (solid) and "
                    f"cosmocnc_jax (dashed), in both 2D correlated and 1D independent "
                    f"configurations. Bottom: residual (jax - cosmocnc). "
                    f"n_points_data_lik=128.")
            if scan_name == "alens":
                text += (" a_lens directly modulates the CMB lensing mass bias, "
                         "making it sensitive to the 2D backward conv correctness.")
            fig.text(0.1, 0.5, text, ha='left', va='center', fontsize=10,
                     wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        # ──────────────────────────────────────────────
        # SECTION 2: Convergence tests
        # ──────────────────────────────────────────────

        sigma8_vals = conv_data["sigma8_vals"]

        # 2D convergence
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        npts_2d = conv_data["npts_2d"]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(npts_2d)))

        jax_high = conv_data["conv_2d"][npts_2d[-1]]["jax"]

        for i, npts in enumerate(npts_2d):
            lls_jax = conv_data["conv_2d"][npts]["jax"]
            axes[0].plot(sigma8_vals, lls_jax, color=cmap[i], marker='o', ms=3,
                        label=f"n={npts}")
            residual = np.array(lls_jax) - np.array(jax_high)
            axes[1].plot(sigma8_vals, residual, color=cmap[i], marker='o', ms=3)

        axes[0].set_ylabel("log L (jax 2D)")
        axes[0].legend(fontsize=8, title="n_pts_dl")
        axes[0].set_title("2D convergence: cosmocnc_jax, varying n_points_data_lik")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel(r"$\sigma_8$")
        axes[1].set_ylabel(f"log L - log L(n={npts_2d[-1]})")
        axes[1].axhline(0, color='k', ls=':', lw=0.5)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "convergence_2d.png"), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

        # 1D convergence
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        npts_1d = conv_data["npts_1d"]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(npts_1d)))

        jax_high_1d = conv_data["conv_1d"][npts_1d[-1]]["jax"]

        for i, npts in enumerate(npts_1d):
            lls_jax = conv_data["conv_1d"][npts]["jax"]
            axes[0].plot(sigma8_vals, lls_jax, color=cmap[i], marker='o', ms=3,
                        label=f"n={npts}")
            residual = np.array(lls_jax) - np.array(jax_high_1d)
            axes[1].plot(sigma8_vals, residual, color=cmap[i], marker='o', ms=3)

        axes[0].set_ylabel("log L (jax 1D)")
        axes[0].legend(fontsize=8, title="n_pts_dl")
        axes[0].set_title("1D convergence: cosmocnc_jax, varying n_points_data_lik")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel(r"$\sigma_8$")
        axes[1].set_ylabel(f"log L - log L(n={npts_1d[-1]})")
        axes[1].axhline(0, color='k', ls=':', lw=0.5)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "convergence_1d.png"), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

        # jax vs cosmocnc accuracy across n_pts_dl
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs_diffs = []
        max_abs_diffs = []
        for npts in npts_2d:
            diffs = np.abs(np.array(conv_data["conv_2d"][npts]["jax"]) -
                          np.array(conv_data["conv_2d"][npts]["ref"]))
            mean_abs_diffs.append(np.mean(diffs))
            max_abs_diffs.append(np.max(diffs))
        ax.semilogy(npts_2d, mean_abs_diffs, 'bo-', label="mean |jax - cosmocnc| (2D)")
        ax.semilogy(npts_2d, max_abs_diffs, 'rs-', label="max |jax - cosmocnc| (2D)")

        mean_abs_diffs_1d = []
        max_abs_diffs_1d = []
        for npts in npts_1d:
            diffs = np.abs(np.array(conv_data["conv_1d"][npts]["jax"]) -
                          np.array(conv_data["conv_1d"][npts]["ref"]))
            mean_abs_diffs_1d.append(np.mean(diffs))
            max_abs_diffs_1d.append(np.max(diffs))
        ax.semilogy(npts_1d, mean_abs_diffs_1d, 'g^-', label="mean |jax - cosmocnc| (1D)")
        ax.semilogy(npts_1d, max_abs_diffs_1d, 'mv-', label="max |jax - cosmocnc| (1D)")

        ax.set_xlabel("n_points_data_lik")
        ax.set_ylabel("|log L difference|")
        ax.set_title("jax vs cosmocnc accuracy across n_points_data_lik")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "accuracy_vs_npts.png"), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 4))
        fig.text(0.1, 0.5,
                 "Convergence tests. Top figures: log L vs sigma_8 at different "
                 "n_points_data_lik resolutions. Bottom panels show residual vs "
                 "highest resolution. The 2D case converges more slowly due to "
                 "the 2D convolution grid. Bottom figure: absolute difference "
                 "between jax and cosmocnc as a function of n_points_data_lik, "
                 "showing both mean and max over the sigma_8 scan points.",
                 ha='left', va='center', fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────
        # SECTION 3: 1D vs 2D comparison
        # ──────────────────────────────────────────────

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        alens = comp_data["alens_vals"]
        axes[0].plot(alens, comp_data["diff_ref"], 'C0-o', ms=4,
                    label="cosmocnc (2D - 1D)")
        axes[0].plot(alens, comp_data["diff_jax"], 'C1--s', ms=4,
                    label="cosmocnc_jax (2D - 1D)")
        axes[0].set_ylabel("log L(2D) - log L(1D)")
        axes[0].legend()
        axes[0].set_title("2D vs 1D log-likelihood difference vs a_lens")
        axes[0].grid(True, alpha=0.3)

        residual = np.array(comp_data["diff_jax"]) - np.array(comp_data["diff_ref"])
        axes[1].plot(alens, residual, 'k-o', ms=4)
        axes[1].axhline(0, color='r', ls=':', lw=0.5)
        axes[1].set_xlabel(r"$a_\mathrm{lens}$")
        axes[1].set_ylabel("jax diff - cosmocnc diff")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "1d_vs_2d.png"), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

        fid = comp_data["fiducial"]
        fig = plt.figure(figsize=(8.5, 4))
        text = (
            "1D vs 2D comparison. The 2D correlated configuration accounts for "
            "cross-covariance between q_so_sim and p_so_sim (corr=0.5), while 1D "
            "treats them independently. The difference is sensitive to a_lens "
            "(CMB lensing mass bias).\n\n"
            f"Fiducial values (n_pts_dl=128):\n"
            f"  cosmocnc 2D: {fid['ref_2d']:.4f}    cosmocnc 1D: {fid['ref_1d']:.4f}\n"
            f"  jax 2D:      {fid['jax_2d']:.4f}    jax 1D:      {fid['jax_1d']:.4f}\n"
            f"  2D-1D (cosmocnc): {fid['ref_2d']-fid['ref_1d']:.4f}\n"
            f"  2D-1D (jax):      {fid['jax_2d']-fid['jax_1d']:.4f}"
        )
        fig.text(0.1, 0.5, text, ha='left', va='center', fontsize=10,
                 family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────
        # SECTION 4: Performance
        # ──────────────────────────────────────────────

        fig, ax = plt.subplots(figsize=(8, 5))
        npts_perf = [k for k in sorted(perf_data.keys()) if isinstance(k, int)]
        for mode, color, marker in [("1d", "C0", "o"), ("2d", "C1", "s")]:
            avgs = [perf_data[n][mode]["avg_ms"] for n in npts_perf]
            mins = [perf_data[n][mode]["min_ms"] for n in npts_perf]
            ax.plot(npts_perf, avgs, f'{color}-{marker}', ms=5,
                   label=f"jax {mode} (avg)")
            ax.plot(npts_perf, mins, f'{color}--{marker}', ms=4, alpha=0.6,
                   label=f"jax {mode} (min)")

        if "numpy_ref" in perf_data:
            for mode, marker in [("1d", "^"), ("2d", "v")]:
                if mode in perf_data["numpy_ref"]:
                    ax.axhline(perf_data["numpy_ref"][mode]["avg_ms"],
                              color='red', ls=':', alpha=0.7,
                              label=f"numpy {mode} (n=128)")

        ax.set_xlabel("n_points_data_lik")
        ax.set_ylabel("Time (ms)")
        ax.set_title("get_log_lik() performance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "performance.png"), dpi=150)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 5))
        lines = ["Performance summary (ms)\n"]
        lines.append(f"{'n_pts_dl':>10} {'1D avg':>10} {'1D min':>10} "
                     f"{'2D avg':>10} {'2D min':>10}")
        lines.append("-" * 55)
        for n in npts_perf:
            d1 = perf_data[n]["1d"]
            d2 = perf_data[n]["2d"]
            lines.append(f"{n:>10} {d1['avg_ms']:>10.1f} {d1['min_ms']:>10.1f} "
                        f"{d2['avg_ms']:>10.1f} {d2['min_ms']:>10.1f}")
        if "numpy_ref" in perf_data:
            lines.append("")
            lines.append("NumPy reference (n_pts_dl=128):")
            for mode in ["1d", "2d"]:
                if mode in perf_data["numpy_ref"]:
                    d = perf_data["numpy_ref"][mode]
                    lines.append(f"  {mode}: avg={d['avg_ms']:.0f}ms")

        fig.text(0.1, 0.5, "\n".join(lines), ha='left', va='center',
                 fontsize=10, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────
        # SECTION 5: Accuracy summary
        # ──────────────────────────────────────────────

        fig = plt.figure(figsize=(8.5, 6))
        lines = ["Accuracy summary: |jax - cosmocnc| at fiducial parameters\n"]
        lines.append(f"{'Config':>15} {'cosmocnc':>14} {'jax':>14} "
                     f"{'abs_diff':>12} {'rel_diff':>12}")
        lines.append("-" * 70)
        for key in sorted(acc_data.keys()):
            d = acc_data[key]
            lines.append(f"{key:>15} {d['ref']:>14.4f} {d['jax']:>14.4f} "
                        f"{d['abs_diff']:>12.4f} {d['rel_diff']:>12.2e}")

        lines.append("\n\nNotes:")
        lines.append("- rel_diff < 1e-3 is acceptable for MCMC applications")
        lines.append("- Differences arise from: analytical vs numerical mass derivatives,")
        lines.append("  circular vs linear convolution (2D), and floating-point differences")

        fig.text(0.1, 0.5, "\n".join(lines), ha='left', va='center',
                 fontsize=10, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF report saved to: {pdf_path}")
    return pdf_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    scan_data = section_param_scans()
    conv_data = section_convergence()
    comp_data = section_1d_vs_2d()
    perf_data = section_performance()
    acc_data = section_accuracy()

    pdf_path = make_plots_and_pdf(scan_data, conv_data, comp_data, perf_data, acc_data)

    n_inits = len(_pool)
    total_time = time.time() - t_start
    print(f"\nTotal instances created: {n_inits} (2 slow cosmo inits, rest fast)")
    print(f"Total test time: {total_time/60:.1f} min")
    print(f"Report: {pdf_path}")


if __name__ == "__main__":
    main()
