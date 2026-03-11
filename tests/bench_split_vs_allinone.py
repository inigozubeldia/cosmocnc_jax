"""Benchmark: split-JIT vs all-in-one at different cluster counts.
Tests the isolated 2D backward conv kernel at batch sizes from 100 to 15000.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time

print(f"JAX backend: {jax.default_backend()}")

n = 128

cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
inv_00 = cov[1,1]/det; inv_01 = -cov[0,1]/det; inv_11 = cov[0,0]/det
norm = 1.0 / jnp.sqrt((2*jnp.pi)**2 * det)

# ── Approach 1: Split-JIT (separate vmap'd kernel) ──
def _circular_convolve(a, kernel):
    kernel_shifted = jnp.fft.ifftshift(kernel)
    A = jnp.fft.fftn(a)
    K = jnp.fft.fftn(kernel_shifted)
    return jnp.fft.ifftn(A * K).real

def per_cluster_2d(r0, r1, kc0, kc1, x_l0_0, x_l0_1, x_lin_0_start, x_lin_1_start, dx0, dx1):
    maha1 = (inv_00 * r0[:, None]**2 +
             inv_11 * r1[None, :]**2 +
             2 * inv_01 * r0[:, None] * r1[None, :])
    cpdf = norm * jnp.exp(-0.5 * maha1)

    maha0 = (inv_00 * kc0[:, None]**2 +
             inv_11 * kc1[None, :]**2 +
             2 * inv_01 * kc0[:, None] * kc1[None, :])
    kernel = norm * jnp.exp(-0.5 * maha0)

    cpdf = _circular_convolve(cpdf, kernel)
    cpdf = jnp.maximum(cpdf, 0.)

    fi0 = jnp.clip((x_l0_0 - x_lin_0_start) / dx0, 0., n - 1 - 1e-7)
    fi1 = jnp.clip((x_l0_1 - x_lin_1_start) / dx1, 0., n - 1 - 1e-7)
    i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n - 2)
    i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n - 2)
    t0 = fi0 - i0; t1 = fi1 - i1
    cpdf = (cpdf[i0, i1]*(1-t0)*(1-t1) + cpdf[i0, i1+1]*(1-t0)*t1 +
            cpdf[i0+1, i1]*t0*(1-t1) + cpdf[i0+1, i1+1]*t0*t1)
    return cpdf

split_jit = jax.jit(jax.vmap(per_cluster_2d))

# ── Approach 2: All-in-one (meshgrid + eval_gaussian_nd style, mimics current cnc.py) ──
def per_cluster_allinone(r0, r1, kc0, kc1, x_l0_0, x_l0_1, x_lin_0_start, x_lin_1_start, dx0, dx1):
    # Meshgrid approach (what XLA over-fuses in the all-in-one JIT)
    r0_g, r1_g = jnp.meshgrid(r0, r1, indexing='ij')
    x_mesh = jnp.stack([r0_g, r1_g])  # (2, n, n)
    # Full covariance Gaussian
    diff = x_mesh  # centered at 0
    inv_cov = jnp.linalg.inv(cov)
    maha = jnp.einsum('i...,ij,j...->...', diff, inv_cov, diff)
    cpdf = norm * jnp.exp(-0.5 * maha)

    kc0_g, kc1_g = jnp.meshgrid(kc0, kc1, indexing='ij')
    x_k = jnp.stack([kc0_g, kc1_g])
    diff_k = x_k
    maha_k = jnp.einsum('i...,ij,j...->...', diff_k, inv_cov, diff_k)
    kernel = norm * jnp.exp(-0.5 * maha_k)

    cpdf = _circular_convolve(cpdf, kernel)
    cpdf = jnp.maximum(cpdf, 0.)

    fi0 = jnp.clip((x_l0_0 - x_lin_0_start) / dx0, 0., n - 1 - 1e-7)
    fi1 = jnp.clip((x_l0_1 - x_lin_1_start) / dx1, 0., n - 1 - 1e-7)
    i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n - 2)
    i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n - 2)
    t0 = fi0 - i0; t1 = fi1 - i1
    cpdf = (cpdf[i0, i1]*(1-t0)*(1-t1) + cpdf[i0, i1+1]*(1-t0)*t1 +
            cpdf[i0+1, i1]*t0*(1-t1) + cpdf[i0+1, i1+1]*t0*t1)
    return cpdf

allinone_jit = jax.jit(jax.vmap(per_cluster_allinone))


def bench(fn, args, label, n_warmup=1, n_runs=5):
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append(time.time() - t0)
    avg = sum(times) / len(times) * 1000
    return avg


key = jax.random.PRNGKey(42)
x = jnp.linspace(-5, 5, n)
dx = x[1] - x[0]

print(f"\n{'Batch':>7s}  {'Split-JIT':>10s}  {'All-in-one':>10s}  {'Ratio':>6s}")
print("-" * 42)

for batch in [100, 300, 500, 1000, 3000, 5000, 10000, 15000]:
    r0s = jax.random.normal(key, (batch, n)) * 0.1 + x[None, :]
    r1s = jax.random.normal(key, (batch, n)) * 0.1 + x[None, :]
    kc0s = jnp.tile(x[None, :], (batch, 1))
    kc1s = jnp.tile(x[None, :], (batch, 1))
    x_l0_0s = jnp.tile(x[None, :], (batch, 1))
    x_l0_1s = jnp.tile(x[None, :], (batch, 1))
    starts_0 = jnp.full(batch, x[0])
    starts_1 = jnp.full(batch, x[0])
    dxs0 = jnp.full(batch, dx)
    dxs1 = jnp.full(batch, dx)

    args = (r0s, r1s, kc0s, kc1s, x_l0_0s, x_l0_1s, starts_0, starts_1, dxs0, dxs1)

    try:
        t_split = bench(split_jit, args, "split")
    except Exception as e:
        t_split = -1
        print(f"  Split FAILED at batch={batch}: {e}")

    try:
        t_aio = bench(allinone_jit, args, "allinone")
    except Exception as e:
        t_aio = -1
        print(f"  All-in-one FAILED at batch={batch}: {e}")

    if t_split > 0 and t_aio > 0:
        ratio = t_aio / t_split
        print(f"{batch:7d}  {t_split:8.1f}ms  {t_aio:8.1f}ms  {ratio:5.1f}x")
    elif t_split > 0:
        print(f"{batch:7d}  {t_split:8.1f}ms  {'FAIL':>10s}  {'N/A':>6s}")
    elif t_aio > 0:
        print(f"{batch:7d}  {'FAIL':>10s}  {t_aio:8.1f}ms  {'N/A':>6s}")
