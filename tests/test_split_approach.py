"""Test: split 2D backward conv into separate JIT from the rest."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time

print(f"JAX backend: {jax.default_backend()}")

# Simulate the full 2D backward conv pipeline for n=128
n = 128
batch = 15000

cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
inv_00 = cov[1,1]/det; inv_01 = -cov[0,1]/det; inv_11 = cov[0,0]/det
norm = 1.0 / jnp.sqrt((2*jnp.pi)**2 * det)

def _circular_convolve(a, kernel):
    kernel_shifted = jnp.fft.ifftshift(kernel)
    A = jnp.fft.fftn(a)
    K = jnp.fft.fftn(kernel_shifted)
    return jnp.fft.ifftn(A * K).real

# Full 2D pipeline per cluster (outer product Gaussian + circular conv + bilinear interp)
def per_cluster_2d(r0, r1, kc0, kc1, x_l0_0, x_l0_1, x_lin_0_start, x_lin_1_start, dx0, dx1):
    # Layer-1 Gaussian (outer product — no meshgrid)
    maha1 = (inv_00 * r0[:, None]**2 +
             inv_11 * r1[None, :]**2 +
             2 * inv_01 * r0[:, None] * r1[None, :])
    cpdf = norm * jnp.exp(-0.5 * maha1)

    # Layer-0 kernel (outer product)
    maha0 = (inv_00 * kc0[:, None]**2 +
             inv_11 * kc1[None, :]**2 +
             2 * inv_01 * kc0[:, None] * kc1[None, :])
    kernel = norm * jnp.exp(-0.5 * maha0)

    # Circular convolution
    cpdf = _circular_convolve(cpdf, kernel)
    cpdf = jnp.maximum(cpdf, 0.)

    # Bilinear interpolation at diagonal points
    fi0 = jnp.clip((x_l0_0 - x_lin_0_start) / dx0, 0., n - 1 - 1e-7)
    fi1 = jnp.clip((x_l0_1 - x_lin_1_start) / dx1, 0., n - 1 - 1e-7)
    i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n - 2)
    i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n - 2)
    t0 = fi0 - i0; t1 = fi1 - i1
    cpdf = (cpdf[i0, i1]*(1-t0)*(1-t1) + cpdf[i0, i1+1]*(1-t0)*t1 +
            cpdf[i0+1, i1]*t0*(1-t1) + cpdf[i0+1, i1+1]*t0*t1)
    return cpdf

# Vmapped + JIT'd (separate from any other computation)
bc_2d_vmap = jax.jit(jax.vmap(per_cluster_2d))

# Create test data
key = jax.random.PRNGKey(42)
x = jnp.linspace(-5, 5, n)
r0s = jax.random.normal(key, (batch, n)) * 0.1 + x[None, :]
r1s = jax.random.normal(key, (batch, n)) * 0.1 + x[None, :]
kc0s = jnp.tile(x[None, :], (batch, 1))
kc1s = jnp.tile(x[None, :], (batch, 1))
x_l0_0s = jnp.tile(x[None, :], (batch, 1))
x_l0_1s = jnp.tile(x[None, :], (batch, 1))
starts_0 = jnp.full(batch, x[0])
starts_1 = jnp.full(batch, x[0])
dx = x[1] - x[0]
dxs0 = jnp.full(batch, dx)
dxs1 = jnp.full(batch, dx)

print(f"\n--- 2D backward conv: {batch} clusters × {n}×{n} ---")

# Warmup
try:
    result = bc_2d_vmap(r0s, r1s, kc0s, kc1s, x_l0_0s, x_l0_1s,
                         starts_0, starts_1, dxs0, dxs1)
    jax.block_until_ready(result)
    print(f"  Warmup OK, shape={result.shape}")

    # Steady state
    times = []
    for _ in range(5):
        t0 = time.time()
        result = bc_2d_vmap(r0s, r1s, kc0s, kc1s, x_l0_0s, x_l0_1s,
                             starts_0, starts_1, dxs0, dxs1)
        jax.block_until_ready(result)
        times.append(time.time() - t0)
    avg = sum(times)/len(times)*1000
    print(f"  Avg: {avg:.0f}ms for {batch} clusters")
    print(f"  Per 1000: {avg/batch*1000:.1f}ms")
except Exception as e:
    print(f"  FAILED: {e}")

