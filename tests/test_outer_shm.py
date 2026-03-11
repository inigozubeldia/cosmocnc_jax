"""Test if outer product Gaussian works at different grid sizes."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time

print(f"JAX backend: {jax.default_backend()}")

cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
inv_00 = cov[1,1]/det
inv_01 = -cov[0,1]/det
inv_11 = cov[0,0]/det
norm = 1.0 / jnp.sqrt((2*jnp.pi)**2 * det)

for n in [64, 96, 100, 112, 120, 128, 192, 256]:
    x = jnp.linspace(-5, 5, n)
    batch = 1000

    @jax.jit
    @jax.vmap
    def outer_gaussian(r0, r1):
        maha = (inv_00 * r0[:, None]**2 +
                inv_11 * r1[None, :]**2 +
                2 * inv_01 * r0[:, None] * r1[None, :])
        return norm * jnp.exp(-0.5 * maha)

    r0s = jnp.tile(x[None], (batch, 1))
    r1s = jnp.tile(x[None], (batch, 1))

    try:
        result = outer_gaussian(r0s, r1s)
        jax.block_until_ready(result)
        # Steady state
        t0 = time.time()
        for _ in range(5):
            result = outer_gaussian(r0s, r1s)
            jax.block_until_ready(result)
        dt = (time.time() - t0) / 5 * 1000
        print(f"  n={n:4d}: OK  {dt:.1f}ms/1000  shape={result.shape}")
    except Exception as e:
        print(f"  n={n:4d}: FAILED — {type(e).__name__}: {str(e)[:80]}")
