"""Component-level profiling of N-D backward conv pipeline."""
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
import time, sys
sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]
import cosmocnc_jax
from cosmocnc_jax.utils import _circular_convolve, eval_gaussian_nd, get_mesh, gaussian_1d

print(f"JAX backend: {jax.default_backend()}")

# ============================================================
# 1. Micro-benchmark: individual 2D operations at scale
# ============================================================
print("\n" + "=" * 60)
print("  MICRO-BENCHMARKS (vmapped over 1000 samples, n=128)")
print("=" * 60)

n = 128
batch = 1000
key = jax.random.PRNGKey(0)

# Create test data
x = jnp.linspace(-5, 5, n)
xx, yy = jnp.meshgrid(x, x)
signal_2d = jnp.exp(-(xx**2 + yy**2) / 2)
kernel_2d = jnp.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])

# Batch
signals = jnp.tile(signal_2d[None], (batch, 1, 1))
kernels = jnp.tile(kernel_2d[None], (batch, 1, 1))

def bench(label, fn, n_warmup=2, n_run=5):
    for _ in range(n_warmup):
        r = fn()
        jax.block_until_ready(r)
    times = []
    for _ in range(n_run):
        t0 = time.time()
        r = fn()
        jax.block_until_ready(r)
        times.append(time.time() - t0)
    avg = np.mean(times) * 1000
    print(f"  {label:50s} {avg:8.1f} ms")
    return avg

# 2D meshgrid creation
x_stack = jnp.tile(jnp.stack([x, x])[None], (batch, 1, 1))  # (batch, 2, n)
@jax.jit
@jax.vmap
def do_meshgrid(xs):
    return get_mesh(xs)
bench("get_mesh (2, 256) → (2, 256, 256)", lambda: do_meshgrid(x_stack))

# 2D Gaussian evaluation
x_mesh_single = get_mesh(jnp.stack([x, x]))  # (2, 256, 256)
x_meshes = jnp.tile(x_mesh_single[None], (batch, 1, 1, 1))
@jax.jit
@jax.vmap
def do_gaussian(xm):
    return eval_gaussian_nd(xm, cov=cov)
bench("eval_gaussian_nd (2, 256, 256)", lambda: do_gaussian(x_meshes))

# Circular convolution
@jax.jit
@jax.vmap
def do_conv(s, k):
    return _circular_convolve(s, k)
bench("circular_convolve (256, 256)", lambda: do_conv(signals, kernels))

# Bilinear interpolation (256 diagonal points from 256x256 grid)
fi_batch = jnp.tile(jnp.linspace(0.5, n-1.5, n)[None], (batch, 1))
@jax.jit
@jax.vmap
def do_bilinear(s, fi):
    i0 = jnp.floor(fi).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, n - 2)
    t = fi - i0
    return (s[i0, i0] * (1-t) * (1-t) +
            s[i0, i0+1] * (1-t) * t +
            s[i0+1, i0] * t * (1-t) +
            s[i0+1, i0+1] * t * t)
bench("bilinear interp (256 diag from 256x256)", lambda: do_bilinear(signals, fi_batch))

# 1D FFT convolution for comparison
signal_1d = jnp.tile(x[None], (batch, 1))
kernel_1d = jnp.tile(gaussian_1d(x, 0.3)[None], (batch, 1))
@jax.jit
@jax.vmap
def do_conv_1d(s, k):
    from cosmocnc_jax.utils import _fft_convolve_same
    return _fft_convolve_same(s, k)
bench("1D fft_convolve_same (256)", lambda: do_conv_1d(signal_1d, kernel_1d))

# Full 2D pipeline (meshgrid + gaussian + conv + interp)
@jax.jit
@jax.vmap
def do_full_2d(xs, obs_shift, kern):
    mesh = get_mesh(xs)
    shifted = mesh  # + obs_shift (simplified)
    pdf = eval_gaussian_nd(shifted, cov=cov)
    pdf_conv = _circular_convolve(pdf, kern)
    pdf_conv = jnp.maximum(pdf_conv, 0.)
    # Diagonal interpolation
    fi = jnp.linspace(0.5, n-1.5, n)
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, n-2)
    t = fi - i0
    return (pdf_conv[i0, i0]*(1-t)*(1-t) + pdf_conv[i0, i0+1]*(1-t)*t +
            pdf_conv[i0+1, i0]*t*(1-t) + pdf_conv[i0+1, i0+1]*t*t)

obs_shifts = jnp.zeros((batch, 2, 1, 1))
bench("full 2D pipeline (mesh+gauss+conv+interp)", lambda: do_full_2d(x_stack, obs_shifts, kernels))


# ============================================================
# 2. Full pipeline profiling
# ============================================================
print("\n" + "=" * 60)
print("  FULL PIPELINE (2D correlated, chunk=1000)")
print("=" * 60)

cnc_params = {
    "cluster_catalogue": "SO_sim_0",
    "observables": [["q_so_sim", "p_so_sim"]],
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
    "cosmology_tool": "classy_sz",
    "likelihood_type": "unbinned",
    "data_lik_type": "backward_convolutional",
    "stacked_likelihood": False,
    "apply_obs_cutoff": False,
    "sigma_mass_prior": 5., "z_errors": False,
    "delta_m_with_ref": False, "scalrel_type_deriv": "analytical",
    "downsample_hmf_bc": 8, "padding_fraction": 0.,
    "bc_chunk_size": 1000,
    "hmf_type": "Tinker08", "hmf_calc": "classy_sz",
    "sigma_R_type": "class_sz", "mass_definition": "500c",
}
scal_rel = {
    "bias_sz": 3., "bias_cmblens": 3.,
    "sigma_lnq_szifi": 0.2, "sigma_lnp": 0.2, "corr_lnq_lnp": 0.5,
    "A_szifi": -4.439, "alpha_szifi": 1.617, "a_lens": 1., "dof": 0.,
}
cosmo = {
    "Om0": 0.315, "Ob0": 0.04897, "h": 0.674,
    "sigma_8": 0.811, "n_s": 0.96, "m_nu": 0.06,
    "tau_reio": 0.0544, "w0": -1., "N_eff": 3.046,
}

nc = cosmocnc_jax.cluster_number_counts()
nc.cnc_params.update(cnc_params)
nc.scal_rel_params.update(scal_rel)
nc.cosmo_params.update(cosmo)
nc.initialise()

# Warmup
ll = nc.get_log_lik()
jax.block_until_ready(ll)

bench("get_log_lik (total)", lambda: jax.block_until_ready(nc.get_log_lik()))

# Chunk size sweep
print("\n" + "=" * 60)
print("  CHUNK SIZE SWEEP")
print("=" * 60)

for cs in [500, 1000, 2000, 5000, 0]:
    label = f"chunk={cs}" if cs > 0 else "full_vmap"
    nc.cnc_params["bc_chunk_size"] = cs
    # Need to re-warmup since chunk size change may trigger recompile
    try:
        ll = nc.get_log_lik()
        jax.block_until_ready(ll)
        bench(f"{label}", lambda: jax.block_until_ready(nc.get_log_lik()), n_warmup=1, n_run=3)
    except Exception as e:
        print(f"  {label:50s}  FAILED: {type(e).__name__}")
