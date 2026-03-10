"""Test: precomputed FFTLog matrix vs calling TophatVar repeatedly."""
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
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

sys.path = [p for p in sys.path if p not in ('', '.', '/scratch/scratch-izubeldia')]

from mcfit import TophatVar

# Get realistic k grid from classy_sz
from classy_sz import Class as Class_sz
classy = Class_sz()
classy.set({
    'H0': 67.4, 'omega_b': 0.022246, 'omega_cdm': 0.12085,
    'tau_reio': 0.0544, 'n_s': 0.96, 'sigma8': 0.811, 'm_ncdm': 0.06,
    'output': 'tSZ_1h', 'HMF_prescription_NCDM': 1, 'no_spline_in_tinker': 1,
})
classy.compute_class_szfast()

pvd = {
    'H0': 67.4, 'omega_b': 0.022246, 'omega_cdm': 0.12085,
    'tau_reio': 0.0544, 'n_s': 0.96,
    'ln10^{10}A_s': np.log(np.exp(classy.get_current_derived_parameters(["ln10^{10}A_s"])["ln10^{10}A_s"])/1e10 * 1e10),
    'm_ncdm': 0.06,
}

# Get P(k) for multiple redshifts
n_z = 100
z_arr = np.linspace(0.01, 3.0, n_z)
pk_list = []
for z in z_arr:
    pk, k = classy.get_pkl_at_z(float(z), params_values_dict=pvd)
    pk_list.append(np.asarray(pk))
k_arr = np.asarray(k)
pk_batch = np.stack(pk_list)
print(f"k_arr shape: {k_arr.shape}, pk_batch shape: {pk_batch.shape}")

# Create TophatVar objects
tv0 = TophatVar(k_arr, lowring=True, deriv=0, backend='jax')
tv1 = TophatVar(k_arr, lowring=True, deriv=1, backend='jax')

# === Method 1: Direct call (current approach) ===
print("\n=== Method 1: Direct tv0(pk) calls ===")
def _single_z_direct(pk):
    _, var = tv0(pk, extrap=True)
    _, dvar = tv1(pk * jnp.asarray(k_arr), extrap=True)
    sigma = jnp.sqrt(var)
    dsigma = dvar / (2.0 * sigma)
    return sigma, dsigma

vmap_direct = jax.vmap(_single_z_direct)
pk_jax = jnp.asarray(pk_batch)
k_jax = jnp.asarray(k_arr)

# Warmup
sigma_direct, dsigma_direct = vmap_direct(pk_jax)

t0 = time.time()
for _ in range(10):
    sigma_direct, dsigma_direct = vmap_direct(pk_jax)
sigma_direct.block_until_ready()
t_direct = (time.time() - t0) / 10
print(f"  Time: {t_direct*1000:.1f}ms per call")
print(f"  sigma shape: {sigma_direct.shape}")

# === Method 2: Matrix approach ===
print("\n=== Method 2: Precomputed matrix ===")

# Try matrix() method
try:
    M0 = tv0.matrix(full=True, keeppads=False)
    print(f"  tv0.matrix() shape: {np.asarray(M0).shape}")
except Exception as e:
    print(f"  tv0.matrix() failed: {e}")

# Try with keeppads=True
try:
    M0_pad = tv0.matrix(full=True, keeppads=True)
    print(f"  tv0.matrix(keeppads=True) shape: {np.asarray(M0_pad).shape}")
except Exception as e:
    print(f"  tv0.matrix(keeppads=True) failed: {e}")

# Try factored form
try:
    a, b, C = tv0.matrix(full=False, keeppads=True)
    print(f"  a shape: {np.asarray(a).shape}, b shape: {np.asarray(b).shape}, C shape: {np.asarray(C).shape}")
except Exception as e:
    print(f"  factored form failed: {e}")

# Try manually constructing the matrix
print("\n=== Method 3: Manual matrix construction ===")
# The idea: call tv0 with identity-like inputs to build the matrix
n_k = len(k_arr)
R_vec = jnp.asarray(tv0.y)
n_R = len(R_vec)
print(f"  n_k={n_k}, n_R={n_R}")

# Check if we can extract the matrix by calling tv0 on basis vectors
# Actually, let's just check: does tv0(pk, extrap=False) give a linear result?
# That is, does tv0(a*pk1 + b*pk2) = a*tv0(pk1) + b*tv0(pk2)?

pk1 = pk_jax[0]
pk2 = pk_jax[50]

_, var1 = tv0(pk1, extrap=True)
_, var2 = tv0(pk2, extrap=True)
_, var_sum = tv0(0.7*pk1 + 0.3*pk2, extrap=True)
var_linear = 0.7*var1 + 0.3*var2

err = float(jnp.max(jnp.abs(var_sum - var_linear)))
rel = float(jnp.max(jnp.abs(var_sum - var_linear) / jnp.maximum(jnp.abs(var_sum), 1e-30)))
print(f"  Linearity test (extrap=True): max_abs={err:.3e}, max_rel={rel:.3e}")

_, var1_ne = tv0(pk1, extrap=False)
_, var2_ne = tv0(pk2, extrap=False)
_, var_sum_ne = tv0(0.7*pk1 + 0.3*pk2, extrap=False)
var_linear_ne = 0.7*var1_ne + 0.3*var2_ne

err_ne = float(jnp.max(jnp.abs(var_sum_ne - var_linear_ne)))
rel_ne = float(jnp.max(jnp.abs(var_sum_ne - var_linear_ne) / jnp.maximum(jnp.abs(var_sum_ne), 1e-30)))
print(f"  Linearity test (extrap=False): max_abs={err_ne:.3e}, max_rel={rel_ne:.3e}")

# Build matrix manually using extrap=False (zero-padding = linear)
print("\n=== Method 4: Build transform matrix manually ===")
# For extrap=False (zero pad), the transform is linear, so we can extract M

# Get internal parameters
N_padded = tv0.N  # Padded size
Npad = tv0.Npad   # Left padding
Npad_ = N_padded - n_k - Npad  # Right padding
print(f"  N_padded={N_padded}, Npad_left={Npad}, Npad_right={Npad_}")

# The transform with zero padding:
# 1. Pad: f_padded = [0...0, xfac*pk, 0...0]  (N_padded,)
# 2. FFT: F = rfft(f_padded)
# 3. Multiply: G = F * u
# 4. IFFT: g = hfft(G)
# 5. Unpad: var = yfac * g[Npad:Npad+n_k]

# This can be expressed as: var = yfac * IFFT(u * FFT(pad(xfac * pk)))
# Which IS linear in pk.

# Extract xfac and yfac from tv0 internals
xfac = jnp.asarray(tv0._xfac_)  # shape (n_k,) = x^(-q) * prefac
yfac = jnp.asarray(tv0.yfac)     # shape (n_R,)
u_kernel = jnp.asarray(tv0._u)   # shape (N_padded//2 + 1,) = Mellin kernel in freq domain
print(f"  xfac shape: {xfac.shape}, yfac shape: {yfac.shape}, u_kernel shape: {u_kernel.shape}")

# Build the matrix: M[i,j] = transform of e_j evaluated at R_i
# var = yfac * unpad(IFFT(u * FFT(pad(xfac * pk))))
# Since this is linear in pk, we can write var = M @ pk

# Actually, let me use a smarter approach: precompute the action
# For each basis vector e_j, compute tv0(e_j, extrap=False)
# This gives us column j of the matrix

# But that's N_k calls. Better to use the matrix formula directly.
# The full matrix (padded domain) is a circulant matrix with eigenvalues u.
# Let's construct it.

# DFT matrix approach: M = yfac * Unpad @ IDFT @ diag(u) @ DFT @ Pad @ diag(xfac)
# Where Pad inserts zeros, Unpad extracts the relevant rows

# Let's just do it in one shot with the matrix method if it works

# Alternative: pre-multiply pk by xfac, then use convolution matrix
# f = xfac * pk   (n_k,)
# f_padded = pad(f, Npad, Npad_)   (N_padded,)
# g_padded = circular_conv(f_padded, kernel)  (N_padded,)
# var = yfac * g_padded[Npad:Npad+n_k]

# The circular convolution matrix is: C[i,j] = c[(i-j) % N]
# where c = IFFT(u), the convolution kernel in real space

# Let's build this matrix directly
c_real = jnp.fft.hfft(u_kernel, n=N_padded) / N_padded  # Real-space kernel (N_padded,)
print(f"  c_real shape: {c_real.shape}")

# Build circulant matrix (N_padded × N_padded)
# C[i,j] = c[(i-j) % N]
idx = (jnp.arange(N_padded)[:, None] - jnp.arange(N_padded)[None, :]) % N_padded
C = c_real[idx]
print(f"  C (circulant) shape: {C.shape}")

# Now the full transform (with zero padding) is:
# 1. Multiply by xfac: f = xfac * pk    (n_k,)
# 2. Zero-pad: f_padded = [0]*Npad + f + [0]*Npad_   (N_padded,)
# 3. Circulant multiply: g_padded = C @ f_padded     (N_padded,)
# 4. Unpad: g = g_padded[Npad:Npad+n_k]              (n_k,)
# 5. Multiply by yfac: var = yfac * g                 (n_k,)

# Combining: var = yfac * C[Npad:Npad+n_k, Npad:Npad+n_k] @ (xfac * pk)
# M = diag(yfac) @ C_sub @ diag(xfac)  where C_sub = C[Npad:Npad+n_k, Npad:Npad+n_k]

C_sub = C[Npad:Npad+n_k, Npad:Npad+n_k]
M0_matrix = jnp.diag(yfac) @ C_sub @ jnp.diag(xfac)
print(f"  M0_matrix shape: {M0_matrix.shape}")

# Test: compare M0 @ pk with tv0(pk, extrap=False)
var_matrix = M0_matrix @ pk1
_, var_call = tv0(pk1, extrap=False)
err_m = float(jnp.max(jnp.abs(var_matrix - var_call)))
rel_m = float(jnp.max(jnp.abs(var_matrix - var_call) / jnp.maximum(jnp.abs(var_call), 1e-30)))
print(f"  Matrix vs tv0(extrap=False): max_abs={err_m:.3e}, max_rel={rel_m:.3e}")

# Now compare extrap=True vs extrap=False to see if we lose accuracy
_, var_extrap = tv0(pk1, extrap=True)
err_e = float(jnp.max(jnp.abs(var_matrix - var_extrap)))
rel_e = float(jnp.max(jnp.abs(var_matrix - var_extrap) / jnp.maximum(jnp.abs(var_extrap), 1e-30)))
print(f"  Matrix (no extrap) vs tv0(extrap=True): max_abs={err_e:.3e}, max_rel={rel_e:.3e}")

# Build M1 for derivative too
xfac1 = jnp.asarray(tv1._xfac_)
yfac1 = jnp.asarray(tv1.yfac)
u_kernel1 = jnp.asarray(tv1._u)
c_real1 = jnp.fft.hfft(u_kernel1, n=tv1.N) / tv1.N
idx1 = (jnp.arange(tv1.N)[:, None] - jnp.arange(tv1.N)[None, :]) % tv1.N
C1 = c_real1[idx1]
C1_sub = C1[tv1.Npad:tv1.Npad+n_k, tv1.Npad:tv1.Npad+n_k]
M1_matrix = jnp.diag(yfac1) @ C1_sub @ jnp.diag(xfac1)

# Test M1
dvar_matrix = M1_matrix @ (pk1 * k_jax)
_, dvar_call = tv1(pk1 * k_jax, extrap=False)
err_d = float(jnp.max(jnp.abs(dvar_matrix - dvar_call)))
print(f"  M1 vs tv1(extrap=False): max_abs={err_d:.3e}")

# === Batch timing comparison ===
print("\n=== Timing: batch matrix multiply vs vmap FFTLog ===")

M0_jax = jnp.asarray(M0_matrix)
M1_jax = jnp.asarray(M1_matrix)

@jax.jit
def batch_sigma_matrix(pk_batch):
    var_batch = pk_batch @ M0_jax.T          # (n_z, n_R)
    dvar_batch = (pk_batch * k_jax) @ M1_jax.T  # (n_z, n_R)
    sigma = jnp.sqrt(var_batch)
    dsigma = dvar_batch / (2.0 * sigma)
    return sigma, dsigma

# Warmup
sigma_mat, dsigma_mat = batch_sigma_matrix(pk_jax)

t0 = time.time()
for _ in range(100):
    sigma_mat, dsigma_mat = batch_sigma_matrix(pk_jax)
dsigma_mat.block_until_ready()
t_matrix = (time.time() - t0) / 100

print(f"  vmap FFTLog:        {t_direct*1000:.2f}ms")
print(f"  Matrix multiply:    {t_matrix*1000:.2f}ms")
print(f"  Speedup:            {t_direct/t_matrix:.1f}x")

# Accuracy
err_s = float(jnp.max(jnp.abs(sigma_mat - sigma_direct)))
rel_s = float(jnp.max(jnp.abs(sigma_mat - sigma_direct) / jnp.maximum(jnp.abs(sigma_direct), 1e-30)))
print(f"  sigma accuracy:     max_abs={err_s:.3e}, max_rel={rel_s:.3e}")

err_ds = float(jnp.max(jnp.abs(dsigma_mat - dsigma_direct)))
rel_ds = float(jnp.max(jnp.abs(dsigma_mat - dsigma_direct) / jnp.maximum(jnp.abs(dsigma_direct), 1e-30)))
print(f"  dsigma accuracy:    max_abs={err_ds:.3e}, max_rel={rel_ds:.3e}")
