"""
3-way benchmark: cosmocnc (NumPy CPU) vs cosmocnc_jax (JAX CPU) vs cosmocnc_jax (JAX GPU)

Run with:
  source /scratch/scratch-izubeldia/cosmocncenv/bin/activate
  python tests/benchmark_3way.py
"""
import os
import sys
import subprocess
import json
import time

# ── Determine mode from env/argv ──
_MODE = os.environ.get("_BENCH_MODE", "main")

# ═══════════════════════════════════════════════════════════════════
# SUBPROCESS WORKER: runs a single JAX benchmark, outputs JSON
# ═══════════════════════════════════════════════════════════════════
if _MODE in ("jax_cpu", "jax_gpu"):
    _N = "10"
    os.environ["OMP_NUM_THREADS"] = _N
    os.environ["OPENBLAS_NUM_THREADS"] = _N
    os.environ["MKL_NUM_THREADS"] = _N
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if _MODE == "jax_cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=10"
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:  # jax_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["XLA_FLAGS"] = ""

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
    import cosmocnc_jax

    SHARED_PARAMS = {
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
    SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}

    nc = cosmocnc_jax.cluster_number_counts()
    nc.cnc_params = dict(nc.cnc_params)
    nc.scal_rel_params = dict(nc.scal_rel_params)
    nc.cosmo_params = dict(nc.cosmo_params)
    nc.cnc_params.update(SHARED_PARAMS)
    nc.scal_rel_params.update(SCAL_REL)

    t_init0 = time.time()
    nc.initialise()
    t_init = time.time() - t_init0

    # Warmup
    nc.get_number_counts()
    ll_warmup = nc.get_log_lik()

    # JIT warmup with parameter change
    n_evals = 10
    sigma_8_vec = np.linspace(0.805, 0.818, n_evals)
    scal = dict(nc.scal_rel_params)

    cp0 = dict(nc.cosmo_params)
    cp0["sigma_8"] = sigma_8_vec[0]
    nc.update_params(cp0, scal)
    _ = nc.get_log_lik()

    # Timed runs
    ll_arr = []
    times = []
    for i in range(n_evals):
        cp = dict(nc.cosmo_params)
        cp["sigma_8"] = sigma_8_vec[i]
        t0 = time.time()
        nc.update_params(cp, scal)
        t1 = time.time()
        ll = nc.get_log_lik()
        t2 = time.time()
        ll_arr.append(float(np.asarray(ll).ravel()[0]))
        times.append({"update": t1 - t0, "get_log_lik": t2 - t1, "total": t2 - t0})

    # Output JSON
    skip = 2
    result = {
        "mode": _MODE,
        "device": str(jax.devices()[0]),
        "init_time": t_init,
        "ll_values": ll_arr,
        "avg_update": float(np.mean([t["update"] for t in times[skip:]])),
        "avg_get_log_lik": float(np.mean([t["get_log_lik"] for t in times[skip:]])),
        "avg_total": float(np.mean([t["total"] for t in times[skip:]])),
        "per_eval": [{"update": t["update"], "get_log_lik": t["get_log_lik"],
                       "total": t["total"], "ll": ll_arr[i]}
                      for i, t in enumerate(times)],
    }
    print("__BENCH_JSON__" + json.dumps(result))
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════
# MAIN: orchestrate all three benchmarks
# ═══════════════════════════════════════════════════════════════════

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Main process: CPU only (leave GPU for subprocess)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
import cosmocnc
import cosmocnc_jax

SHARED_PARAMS = {
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
    "cosmo_param_density": "critical",
    "cosmo_model": "lcdm", "hmf_calc": "cnc",
    "interp_tinker": "linear",
    "stacked_likelihood": False, "likelihood_type": "unbinned",
}
SCAL_REL = {"corr_lnq_lnp": 0., "bias_sz": 0.8, "dof": 0.}
n_evals = 10
sigma_8_vec = np.linspace(0.805, 0.818, n_evals)
skip = 2

print("=" * 78)
print("  3-WAY BENCHMARK: cosmocnc (NumPy) vs cosmocnc_jax (CPU) vs cosmocnc_jax (GPU)")
print("=" * 78)


# ── 1. NumPy benchmark ──────────────────────────────────────────────
print("\n[1/3] cosmocnc (NumPy, CPU)...")
nc_np = cosmocnc.cluster_number_counts()
nc_np.cnc_params = dict(nc_np.cnc_params)
nc_np.scal_rel_params = dict(nc_np.scal_rel_params)
nc_np.cosmo_params = dict(nc_np.cosmo_params)
nc_np.cnc_params.update(SHARED_PARAMS)
nc_np.cnc_params["cosmology_tool"] = "classy_sz"
nc_np.scal_rel_params.update(SCAL_REL)

t0 = time.time()
nc_np.initialise()
t_init_np = time.time() - t0
print(f"  Init: {t_init_np:.1f}s")

# Warmup
nc_np.get_number_counts()
nc_np.get_log_lik()

ll_np = []
times_np = []
scal_np = dict(nc_np.scal_rel_params)
for i in range(n_evals):
    cp = dict(nc_np.cosmo_params)
    cp["sigma_8"] = sigma_8_vec[i]
    t0 = time.time()
    nc_np.update_params(cp, scal_np)
    t1 = time.time()
    ll = nc_np.get_log_lik()
    t2 = time.time()
    ll_np.append(float(ll))
    times_np.append({"update": t1 - t0, "get_log_lik": t2 - t1, "total": t2 - t0})
    if i < 3 or i == n_evals - 1:
        print(f"  [{i}] update={t1-t0:.3f}s  get_log_lik={t2-t1:.3f}s  total={t2-t0:.3f}s  ll={ll_np[-1]:.2f}")

avg_np = {k: np.mean([t[k] for t in times_np[skip:]]) for k in times_np[0]}
ll_np = np.array(ll_np)
print(f"  Average (last {n_evals-skip}): {avg_np['total']*1000:.0f}ms/eval")

# Free memory
del nc_np


# ── 2. JAX CPU benchmark (subprocess — no GPU) ──────────────────────
print(f"\n[2/3] cosmocnc_jax (JAX, CPU) — subprocess with CUDA_VISIBLE_DEVICES='' ...")
env_cpu = os.environ.copy()
env_cpu["_BENCH_MODE"] = "jax_cpu"
env_cpu["CUDA_VISIBLE_DEVICES"] = ""
env_cpu["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=10"

proc_cpu = subprocess.run(
    ["taskset", "-c", "0-9", sys.executable, __file__],
    env=env_cpu, capture_output=True, text=True, timeout=600
)

jax_cpu_result = None
for line in proc_cpu.stdout.split('\n'):
    if line.startswith("__BENCH_JSON__"):
        jax_cpu_result = json.loads(line[len("__BENCH_JSON__"):])
        break

if jax_cpu_result is None:
    print("  ERROR: CPU subprocess failed!")
    if proc_cpu.stderr:
        # Print last 10 lines of stderr
        stderr_lines = proc_cpu.stderr.strip().split('\n')
        for line in stderr_lines[-10:]:
            print(f"    {line}")
else:
    ll_jax_cpu = np.array(jax_cpu_result["ll_values"])
    print(f"  Device: {jax_cpu_result['device']}")
    print(f"  Init: {jax_cpu_result['init_time']:.1f}s")
    for ev in jax_cpu_result["per_eval"]:
        i = jax_cpu_result["per_eval"].index(ev)
        if i < 3 or i == n_evals - 1:
            print(f"  [{i}] update={ev['update']:.4f}s  get_log_lik={ev['get_log_lik']:.4f}s  total={ev['total']:.4f}s  ll={ev['ll']:.2f}")
    print(f"  Average (last {n_evals-skip}): {jax_cpu_result['avg_total']*1000:.1f}ms/eval")


# ── 3. JAX GPU benchmark (subprocess — with GPU) ────────────────────
print(f"\n[3/3] cosmocnc_jax (JAX, GPU) — subprocess with CUDA_VISIBLE_DEVICES=1 ...")
env_gpu = os.environ.copy()
env_gpu["_BENCH_MODE"] = "jax_gpu"
env_gpu["CUDA_VISIBLE_DEVICES"] = "1"
env_gpu["XLA_FLAGS"] = ""
env_gpu.pop("JAX_PLATFORMS", None)  # Remove CPU-only override so GPU is used

proc_gpu = subprocess.run(
    [sys.executable, __file__],
    env=env_gpu, capture_output=True, text=True, timeout=600
)

jax_gpu_result = None
for line in proc_gpu.stdout.split('\n'):
    if line.startswith("__BENCH_JSON__"):
        jax_gpu_result = json.loads(line[len("__BENCH_JSON__"):])
        break

if jax_gpu_result is None:
    print("  ERROR: GPU subprocess failed!")
    if proc_gpu.stderr:
        stderr_lines = proc_gpu.stderr.strip().split('\n')
        for line in stderr_lines[-10:]:
            print(f"    {line}")
else:
    ll_jax_gpu = np.array(jax_gpu_result["ll_values"])
    print(f"  Device: {jax_gpu_result['device']}")
    print(f"  Init: {jax_gpu_result['init_time']:.1f}s")
    for ev in jax_gpu_result["per_eval"]:
        i = jax_gpu_result["per_eval"].index(ev)
        if i < 3 or i == n_evals - 1:
            print(f"  [{i}] update={ev['update']:.4f}s  get_log_lik={ev['get_log_lik']:.4f}s  total={ev['total']:.4f}s  ll={ev['ll']:.2f}")
    print(f"  Average (last {n_evals-skip}): {jax_gpu_result['avg_total']*1000:.1f}ms/eval")


# ── Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("  RESULTS")
print("=" * 78)

headers = f"{'':30s} {'NumPy CPU':>12s}"
divider = f"{'-'*30} {'-'*12}"
row_np_update = f"  {'update_params':28s} {avg_np['update']:10.4f}s "
row_np_loglik = f"  {'get_log_lik':28s} {avg_np['get_log_lik']:10.4f}s "
row_np_total = f"  {'TOTAL per eval':28s} {avg_np['total']:10.4f}s "

if jax_cpu_result:
    headers += f" {'JAX CPU':>12s} {'CPU speedup':>12s}"
    divider += f" {'-'*12} {'-'*12}"
    su = avg_np['update'] / max(jax_cpu_result['avg_update'], 1e-10)
    sl = avg_np['get_log_lik'] / max(jax_cpu_result['avg_get_log_lik'], 1e-10)
    st = avg_np['total'] / max(jax_cpu_result['avg_total'], 1e-10)
    row_np_update += f" {jax_cpu_result['avg_update']:10.4f}s  {su:10.1f}x"
    row_np_loglik += f" {jax_cpu_result['avg_get_log_lik']:10.4f}s  {sl:10.1f}x"
    row_np_total += f" {jax_cpu_result['avg_total']:10.4f}s  {st:10.1f}x"

if jax_gpu_result:
    headers += f" {'JAX GPU':>12s} {'GPU speedup':>12s}"
    divider += f" {'-'*12} {'-'*12}"
    su = avg_np['update'] / max(jax_gpu_result['avg_update'], 1e-10)
    sl = avg_np['get_log_lik'] / max(jax_gpu_result['avg_get_log_lik'], 1e-10)
    st = avg_np['total'] / max(jax_gpu_result['avg_total'], 1e-10)
    row_np_update += f" {jax_gpu_result['avg_update']:10.4f}s  {su:10.1f}x"
    row_np_loglik += f" {jax_gpu_result['avg_get_log_lik']:10.4f}s  {sl:10.1f}x"
    row_np_total += f" {jax_gpu_result['avg_total']:10.4f}s  {st:10.1f}x"

print(f"\n{headers}")
print(divider)
print(row_np_update)
print(row_np_loglik)
print(row_np_total)

# Accuracy
print(f"\n  Numerical accuracy (log_lik vs NumPy):")
if jax_cpu_result:
    max_rel_cpu = np.max(np.abs(ll_np - ll_jax_cpu) / np.maximum(np.abs(ll_np), 1e-30))
    print(f"    JAX CPU: max_rel = {max_rel_cpu:.3e}")
if jax_gpu_result:
    max_rel_gpu = np.max(np.abs(ll_np - ll_jax_gpu) / np.maximum(np.abs(ll_np), 1e-30))
    print(f"    JAX GPU: max_rel = {max_rel_gpu:.3e}")
if jax_cpu_result and jax_gpu_result:
    max_rel_cpugpu = np.max(np.abs(ll_jax_cpu - ll_jax_gpu) / np.maximum(np.abs(ll_jax_cpu), 1e-30))
    print(f"    CPU vs GPU: max_rel = {max_rel_cpugpu:.3e}")

print()
