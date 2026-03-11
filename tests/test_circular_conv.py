"""Quick test: circular convolution correctness + GPU compatibility."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(f"JAX backend: {jax.default_backend()}")

def _circular_convolve(a, kernel):
    kernel_shifted = jnp.fft.ifftshift(kernel)
    A = jnp.fft.fftn(a)
    K = jnp.fft.fftn(kernel_shifted)
    return jnp.fft.ifftn(A * K).real

def _circular_convolve_1d_ffts(a, kernel):
    """2D circular conv using sequential 1D FFTs (avoids cuFFT 2D plans)."""
    kernel_shifted = jnp.fft.ifftshift(kernel)
    # Sequential 1D complex FFTs
    A = jnp.fft.fft(a, axis=1)
    A = jnp.fft.fft(A, axis=0)
    K = jnp.fft.fft(kernel_shifted, axis=1)
    K = jnp.fft.fft(K, axis=0)
    C = A * K
    result = jnp.fft.ifft(C, axis=0)
    result = jnp.fft.ifft(result, axis=1)
    return result.real

# Test 1: correctness on small array
print("\n--- Correctness test ---")
n = 64
x = jnp.linspace(-5, 5, n)
xx, yy = jnp.meshgrid(x, x)
signal = jnp.exp(-(xx**2 + yy**2) / 2)
kernel = jnp.exp(-(xx**2 + yy**2) / (2 * 0.5**2))

res_fftn = _circular_convolve(signal, kernel)
res_seq = _circular_convolve_1d_ffts(signal, kernel)
print(f"  fftn vs sequential 1D: max diff = {float(jnp.max(jnp.abs(res_fftn - res_seq))):.2e}")

# Test 2: vmap compatibility (the real test)
print("\n--- vmap test (batch=100) ---")
batch = 100
signals = jnp.tile(signal[None], (batch, 1, 1))  # (100, 64, 64)
kernels = jnp.tile(kernel[None], (batch, 1, 1))

# Test sequential 1D approach with vmap
vmap_conv = jax.vmap(_circular_convolve_1d_ffts)
try:
    result = vmap_conv(signals, kernels)
    jax.block_until_ready(result)
    print(f"  sequential 1D vmap: OK, shape={result.shape}")
except Exception as e:
    print(f"  sequential 1D vmap: FAILED — {e}")

# Test fftn approach with vmap
vmap_conv2 = jax.vmap(_circular_convolve)
try:
    result2 = vmap_conv2(signals, kernels)
    jax.block_until_ready(result2)
    print(f"  fftn vmap: OK, shape={result2.shape}")
except Exception as e:
    print(f"  fftn vmap: FAILED — {e}")

# Test 3: large batch (5000)
print("\n--- vmap test (batch=5000, n=256) ---")
n2 = 256
x2 = jnp.linspace(-5, 5, n2)
xx2, yy2 = jnp.meshgrid(x2, x2)
signal2 = jnp.exp(-(xx2**2 + yy2**2) / 2)
kernel2 = jnp.exp(-(xx2**2 + yy2**2) / (2 * 0.5**2))
signals2 = jnp.tile(signal2[None], (5000, 1, 1))
kernels2 = jnp.tile(kernel2[None], (5000, 1, 1))

import time

for label, fn in [("sequential_1d", _circular_convolve_1d_ffts),
                  ("fftn", _circular_convolve)]:
    vfn = jax.jit(jax.vmap(fn))
    try:
        r = vfn(signals2, kernels2)
        jax.block_until_ready(r)
        # Steady state
        t0 = time.time()
        r = vfn(signals2, kernels2)
        jax.block_until_ready(r)
        dt = time.time() - t0
        print(f"  {label}: {dt*1000:.0f}ms for 5000×256×256")
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
