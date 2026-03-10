import jax
import jax.numpy as jnp
import jax.scipy.signal as jax_signal
import numpy as np
import functools
import math
import time
import sys

import logging

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=level
    )

def set_verbosity(verbosity):
    levels = {
        'none': logging.CRITICAL,
        'minimal': logging.INFO,
        'extensive': logging.DEBUG
    }

    level = levels.get(verbosity, logging.INFO)
    configure_logging(level)


# =====================================================================
# Simpson's rule integration (JAX-compatible, matching scipy.integrate.simpson)
# =====================================================================

def simpson(y, x=None, axis=-1):
    """Composite Simpson's rule integration matching scipy.integrate.simpson.

    Parameters
    ----------
    y : jnp.ndarray
        Array of function values to integrate.
    x : jnp.ndarray, optional
        Array of sample points corresponding to y values.
    axis : int
        Axis along which to integrate.

    Returns
    -------
    float or jnp.ndarray
        Approximation of the integral.
    """
    y = jnp.asarray(y)

    if x is not None:
        x = jnp.asarray(x)

    # Move integration axis to position 0
    y = jnp.moveaxis(y, axis, 0)
    N = y.shape[0]

    if x is not None:
        if x.ndim == 1:
            h = jnp.diff(x)
        else:
            x = jnp.moveaxis(x, axis, 0)
            h = jnp.diff(x, axis=0)
    else:
        h = jnp.ones(N - 1)

    # For N points we have N-1 intervals
    # Use composite Simpson's rule: process pairs of intervals
    n_intervals = N - 1

    # If even number of intervals (odd N), standard Simpson's
    # If odd number of intervals (even N), use Simpson's for first n-1 intervals + trapezoidal for last

    def _simpson_even_intervals(y, h):
        """Simpson's rule for even number of intervals (pairs). Fully vectorized."""
        h0 = h[0::2]
        h1 = h[1::2]
        n = h0.shape[0]  # number of pairs
        y0 = y[0::2][:n]
        y1 = y[1::2][:n]
        y2 = y[2::2][:n]

        # Expand h dimensions to broadcast with y (h is along axis 0, y may have extra dims)
        extra_dims = y0.ndim - h0.ndim
        for _ in range(extra_dims):
            h0 = h0[..., jnp.newaxis]
            h1 = h1[..., jnp.newaxis]

        hsum = h0 + h1
        hprod = h0 * h1
        hprod_safe = jnp.where(hprod == 0, 1.0, hprod)
        h0_safe = jnp.where(h0 == 0, 1.0, h0)
        h1_safe = jnp.where(h1 == 0, 1.0, h1)

        terms = (hsum / 6.0) * (
            y0 * (2.0 - h1 / h0_safe) +
            y1 * (hsum * hsum / hprod_safe) +
            y2 * (2.0 - h0 / h1_safe)
        )

        return jnp.sum(terms, axis=0)

    if n_intervals % 2 == 0:
        # Even number of intervals -> standard Simpson's
        result = _simpson_even_intervals(y, h)
    else:
        if n_intervals == 1:
            # Only one interval, use trapezoidal
            h_0 = h[0]
            for _ in range(y.ndim - 1):
                h_0 = h_0[..., jnp.newaxis]
            result = 0.5 * h_0 * (y[0] + y[1])
        else:
            # Odd number of intervals: Simpson's for first n-2 intervals, trap for last
            result = _simpson_even_intervals(y[:-1], h[:-1])
            # Add trapezoidal rule for last interval (expand h[-1] for broadcasting)
            h_last = h[-1]
            for _ in range(y.ndim - 1):
                h_last = h_last[..., jnp.newaxis]
            result = result + 0.5 * h_last * (y[-2] + y[-1])

    return result


# =====================================================================
# Convolution functions (JAX-compatible)
# =====================================================================

def _fft_convolve_same(a, b):
    """FFT convolution with 'same' mode using jnp.fft directly.

    More GPU-friendly than jax.scipy.signal.fftconvolve: avoids cuFFT
    batched plan allocation issues with large vmap batch sizes.
    """
    n = a.shape[-1]
    m = b.shape[-1]
    # Pad to next power of 2 >= n+m-1 for efficient FFT
    fft_size = 1
    while fft_size < n + m - 1:
        fft_size *= 2
    A = jnp.fft.rfft(a, n=fft_size)
    B = jnp.fft.rfft(b, n=fft_size)
    result = jnp.fft.irfft(A * B, n=fft_size)
    # 'same' mode: return n elements centered on the full output
    start = (m - 1) // 2
    return result[start:start + n]


def convolve_1d(x, dn_dx, sigma=None, type="fft", kernel=None, sigma_min=0):
    """JIT-compatible 1D convolution with Gaussian kernel."""
    # Always compute the convolution (needed for JIT tracing)
    if kernel is None:
        # Use jnp.maximum to avoid zero sigma in gaussian
        sigma_safe = jnp.maximum(sigma, 1e-30)
        kernel = gaussian_1d(x - jnp.mean(x) + (x[1] - x[0]) * 0.5, sigma_safe)

    convolved = _fft_convolve_same(dn_dx, kernel) / jnp.sum(kernel)

    # Select convolved or original based on sigma > sigma_min (JIT-safe)
    return jnp.where(sigma > sigma_min, convolved, dn_dx)


def convolve_nd(distribution, kernel):

    convolved = _fft_convolve_same(distribution, kernel) / jnp.sum(kernel)

    return convolved


# =====================================================================
# Gaussian PDF functions (JAX-compatible)
# =====================================================================

def eval_gaussian_nd(x_mesh, cov=None, mean=None):

    shape = x_mesh.shape

    if mean is not None:
        # Subtract mean: mean shape is (n_dim,), broadcast to x_mesh shape
        mean = jnp.asarray(mean)
        if shape[0] > 1:
            x_mesh = x_mesh - mean.reshape(-1, *([1] * (len(shape) - 1)))
        else:
            x_mesh = x_mesh - mean.reshape(-1, *([1] * (len(shape) - 1)))

    if shape[0] > 1:

        x_mesh = x_mesh.reshape(*x_mesh.shape[:-2], -1)

        inv_cov = jnp.linalg.inv(cov)
        det_cov = jnp.linalg.det(cov)
        norm_factor = 1.0 / jnp.sqrt((2 * jnp.pi)**shape[0] * det_cov)
        mahalanobis = jnp.sum(jnp.dot(inv_cov, x_mesh) * x_mesh, axis=0)
        pdf = norm_factor * jnp.exp(-0.5 * mahalanobis)
        pdf = jnp.transpose(pdf.reshape(shape[1:]))

    else:

        pdf = gaussian_1d(x_mesh, jnp.sqrt(cov))[0, :]

    return pdf


def get_mesh(x):

    if x.shape[0] == 1:

        x_mesh = x

    elif x.shape[0] == 2:

        x_mesh = jnp.array(jnp.meshgrid(x[0, :], x[1, :]))

    elif x.shape[0] == 3:

        x_mesh = jnp.array(jnp.meshgrid(x[0, :], x[1, :], x[2, :]))

    return x_mesh


def gaussian_1d(x, sigma):

    return jnp.exp(-x**2 / (2. * sigma**2)) / (jnp.sqrt(2. * jnp.pi) * sigma)


# =====================================================================
# Interpolation functions (JAX-compatible)
# =====================================================================

class RegularGridInterpolator:
    """JAX-compatible regular grid interpolator (linear interpolation).

    Mimics scipy.interpolate.RegularGridInterpolator for 1D and 2D grids.
    """

    def __init__(self, points, values, method="linear", fill_value=None, bounds_error=True):
        self.points = tuple(jnp.asarray(p) for p in points)
        self.values = jnp.asarray(values)
        self.fill_value = fill_value
        self.bounds_error = bounds_error
        self.ndim = len(self.points)

    def __call__(self, xi):
        xi = jnp.asarray(xi)

        if self.ndim == 1:
            return self._interp_1d(xi)
        elif self.ndim == 2:
            return self._interp_2d(xi)
        elif self.ndim == 3:
            return self._interp_3d(xi)
        else:
            raise NotImplementedError("Only 1D, 2D, and 3D interpolation supported")

    def _interp_1d(self, xi):
        x = self.points[0]
        v = self.values
        xi_flat = xi.ravel()

        result = jnp.interp(xi_flat, x, v)

        if self.fill_value is not None:
            out_of_bounds = (xi_flat < x[0]) | (xi_flat > x[-1])
            result = jnp.where(out_of_bounds, self.fill_value, result)

        return result.reshape(xi.shape[:-1]) if xi.ndim > 1 else result

    def _interp_2d(self, xi):
        """Bilinear interpolation on a 2D regular grid."""
        x0 = self.points[0]
        x1 = self.points[1]
        values = self.values

        # xi shape: (..., 2) or (N, 2) or just (2,)
        if xi.ndim == 1:
            xi = xi[jnp.newaxis, :]

        original_shape = xi.shape[:-1]
        xi_flat = xi.reshape(-1, 2)

        xi0 = xi_flat[:, 0]
        xi1 = xi_flat[:, 1]

        # Find indices
        n0 = len(x0)
        n1 = len(x1)

        # Normalize to index space
        dx0 = x0[1] - x0[0]
        dx1 = x1[1] - x1[0]

        fi0 = (xi0 - x0[0]) / dx0
        fi1 = (xi1 - x1[0]) / dx1

        # Clamp to valid range
        fi0 = jnp.clip(fi0, 0, n0 - 1 - 1e-7)
        fi1 = jnp.clip(fi1, 0, n1 - 1 - 1e-7)

        i0 = jnp.floor(fi0).astype(jnp.int32)
        i1 = jnp.floor(fi1).astype(jnp.int32)

        i0 = jnp.clip(i0, 0, n0 - 2)
        i1 = jnp.clip(i1, 0, n1 - 2)

        # Fractional parts
        t0 = fi0 - i0
        t1 = fi1 - i1

        # Bilinear interpolation
        v00 = values[i0, i1]
        v01 = values[i0, i1 + 1]
        v10 = values[i0 + 1, i1]
        v11 = values[i0 + 1, i1 + 1]

        result = (v00 * (1 - t0) * (1 - t1) +
                  v01 * (1 - t0) * t1 +
                  v10 * t0 * (1 - t1) +
                  v11 * t0 * t1)

        if self.fill_value is not None:
            out_of_bounds = ((xi0 < x0[0]) | (xi0 > x0[-1]) |
                           (xi1 < x1[0]) | (xi1 > x1[-1]))
            result = jnp.where(out_of_bounds, self.fill_value, result)

        return result.reshape(original_shape)

    def _interp_3d(self, xi):
        """Trilinear interpolation on a 3D regular grid."""
        x0, x1, x2 = self.points
        values = self.values

        if xi.ndim == 1:
            xi = xi[jnp.newaxis, :]

        original_shape = xi.shape[:-1]
        xi_flat = xi.reshape(-1, 3)

        xi0, xi1, xi2 = xi_flat[:, 0], xi_flat[:, 1], xi_flat[:, 2]

        n0, n1, n2 = len(x0), len(x1), len(x2)
        dx0 = x0[1] - x0[0]
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]

        fi0 = jnp.clip((xi0 - x0[0]) / dx0, 0, n0 - 1 - 1e-7)
        fi1 = jnp.clip((xi1 - x1[0]) / dx1, 0, n1 - 1 - 1e-7)
        fi2 = jnp.clip((xi2 - x2[0]) / dx2, 0, n2 - 1 - 1e-7)

        i0 = jnp.clip(jnp.floor(fi0).astype(jnp.int32), 0, n0 - 2)
        i1 = jnp.clip(jnp.floor(fi1).astype(jnp.int32), 0, n1 - 2)
        i2 = jnp.clip(jnp.floor(fi2).astype(jnp.int32), 0, n2 - 2)

        t0 = fi0 - i0
        t1 = fi1 - i1
        t2 = fi2 - i2

        # Trilinear interpolation: 8 corner values
        result = (values[i0, i1, i2] * (1 - t0) * (1 - t1) * (1 - t2) +
                  values[i0, i1, i2 + 1] * (1 - t0) * (1 - t1) * t2 +
                  values[i0, i1 + 1, i2] * (1 - t0) * t1 * (1 - t2) +
                  values[i0, i1 + 1, i2 + 1] * (1 - t0) * t1 * t2 +
                  values[i0 + 1, i1, i2] * t0 * (1 - t1) * (1 - t2) +
                  values[i0 + 1, i1, i2 + 1] * t0 * (1 - t1) * t2 +
                  values[i0 + 1, i1 + 1, i2] * t0 * t1 * (1 - t2) +
                  values[i0 + 1, i1 + 1, i2 + 1] * t0 * t1 * t2)

        if self.fill_value is not None:
            oob = ((xi0 < x0[0]) | (xi0 > x0[-1]) |
                   (xi1 < x1[0]) | (xi1 > x1[-1]) |
                   (xi2 < x2[0]) | (xi2 > x2[-1]))
            result = jnp.where(oob, self.fill_value, result)

        return result.reshape(original_shape)


def interp1d_jax(x_new, x, y, kind="linear", fill_value=0.0):
    """JAX-compatible 1D interpolation (wrapper around jnp.interp with fill)."""
    result = jnp.interp(x_new, x, y, left=fill_value, right=fill_value)
    return result


def interp_along_axis0(z_eval, z_vec, matrix):
    """Interpolate a (n_z, n_points) matrix along axis 0 at a single z value.

    Pure JAX replacement for scipy.interpolate.interp1d along redshift axis.
    Returns a (n_points,) array.

    Uses direct linear interpolation with searchsorted instead of vmap over
    columns, which avoids a nested-vmap compilation blow-up in XLA.
    """
    n_z = z_vec.shape[0]
    # Clamp z_eval to grid range
    z_eval_c = jnp.clip(z_eval, z_vec[0], z_vec[-1])
    # Find bracketing index
    idx = jnp.searchsorted(z_vec, z_eval_c, side='right') - 1
    idx = jnp.clip(idx, 0, n_z - 2)
    # Linear interpolation weight
    dz = z_vec[idx + 1] - z_vec[idx]
    w = jnp.where(dz > 0, (z_eval_c - z_vec[idx]) / dz, 0.0)
    # Interpolate all columns at once: (1-w)*row[idx] + w*row[idx+1]
    return (1.0 - w) * matrix[idx] + w * matrix[idx + 1]


def interp_uniform(x_query, x_min, x_max, n_grid, y_values, left=None, right=None):
    """Interpolate from a uniform grid using direct index computation (no searchsorted).

    GPU-friendly: avoids binary search, uses regular memory access patterns.
    For scalar or array x_query, with y_values on a uniform grid
    x_grid = linspace(x_min, x_max, n_grid).

    Args:
        x_query: query points (scalar or array)
        x_min, x_max: grid bounds
        n_grid: number of grid points
        y_values: values on the uniform grid, shape (n_grid,) or (n_grid, ...)
        left: value for x < x_min (default: y_values[0])
        right: value for x > x_max (default: y_values[-1])
    """
    dx = (x_max - x_min) / (n_grid - 1)
    idx_float = (x_query - x_min) / dx
    idx = jnp.clip(jnp.floor(idx_float).astype(jnp.int32), 0, n_grid - 2)
    w = jnp.clip(idx_float - idx, 0.0, 1.0)
    result = (1.0 - w) * y_values[idx] + w * y_values[idx + 1]
    # Handle out-of-bounds
    if left is not None:
        result = jnp.where(x_query < x_min, left, result)
    if right is not None:
        result = jnp.where(x_query > x_max, right, result)
    return result


def interp_along_axis0_uniform(z_eval, z_min, z_max, n_z, matrix):
    """Interpolate a (n_z, n_points) matrix along axis 0 at a single z value.

    GPU-friendly version for uniform z grids (no searchsorted).
    """
    dz = (z_max - z_min) / (n_z - 1)
    z_eval_c = jnp.clip(z_eval, z_min, z_max)
    idx_float = (z_eval_c - z_min) / dz
    idx = jnp.clip(jnp.floor(idx_float).astype(jnp.int32), 0, n_z - 2)
    w = jnp.clip(idx_float - idx, 0.0, 1.0)
    return (1.0 - w) * matrix[idx] + w * matrix[idx + 1]


def build_cov_matrix_2obs(sigma_00, sigma_01, sigma_11):
    """Build a 2x2 covariance matrix from scatter values.

    Pure JAX function for use inside JIT.
    """
    return jnp.array([[sigma_00, sigma_01],
                       [sigma_01, sigma_11]])


# =====================================================================
# Array operation functions
# =====================================================================

def apodise(x_map):

    from jax.scipy.signal import windows
    # Use numpy for the tukey window generation since JAX doesn't have it
    import scipy.signal as scipy_signal
    window_1d = jnp.array(scipy_signal.windows.tukey(x_map.shape[0], alpha=0.1))
    window = [window_1d for i in range(0, len(x_map.shape))]
    window = functools.reduce(jnp.multiply, jnp.ix_(*window))

    return x_map * window


def extract_diagonal(tensor):

    if len(tensor.shape) == 2:

        diag = jnp.diag(tensor)

    elif len(tensor.shape) == 3:

        n = tensor.shape[0]
        idx = jnp.arange(n)
        diag = tensor[idx, idx, idx]

    return diag


# =====================================================================
# Cash statistic functions (JAX-compatible)
# =====================================================================

def get_cash_statistic(n_obs_vec, n_mean_vec):

    C = eval_cash_statistic(n_obs_vec, n_mean_vec)

    # eval_cash_statistic_expected already supports array inputs (jnp.where chains)
    C_mean, C_var = eval_cash_statistic_expected(n_mean_vec)

    indices_nonnan = ~jnp.isnan(C)

    C = jnp.where(indices_nonnan, C, 0.)
    C_var = jnp.where(indices_nonnan, C_var, 0.)
    C_mean = jnp.where(indices_nonnan, C_mean, 0.)

    C = jnp.sum(C)
    C_var = jnp.sum(C_var)
    C_mean = jnp.sum(C_mean)
    C_std = jnp.sqrt(C_var)

    return (C, C_mean, C_std)


def eval_cash_statistic(n_obs, n_mean):

    return 2. * (n_mean - n_obs + n_obs * jnp.log(n_obs / n_mean))


def eval_cash_statistic_expected(n_mean):

    # Mean - using jnp.where chains for piecewise function
    n_mean = jnp.asarray(n_mean, dtype=jnp.float64)

    C_mean = jnp.where(
        n_mean <= 0.5,
        -0.25 * n_mean**3 + 1.38 * n_mean**2 - 2 * n_mean * jnp.log(jnp.maximum(n_mean, 1e-30)),
        jnp.where(
            n_mean <= 2.,
            -0.00335 * n_mean**5 + 0.04259 * n_mean**4 - 0.27331 * n_mean**3 + 1.381 * n_mean**2 - 2. * n_mean * jnp.log(jnp.maximum(n_mean, 1e-30)),
            jnp.where(
                n_mean <= 5.,
                1.019275 + 0.1345 * n_mean**(0.461 - 0.9 * jnp.log(jnp.maximum(n_mean, 1e-30))),
                jnp.where(
                    n_mean <= 10.,
                    1.00624 + 0.604 / n_mean**1.68,
                    1. + 0.1649 / n_mean + 0.226 / n_mean**2
                )
            )
        )
    )

    # Variance
    C_var = jnp.where(
        n_mean <= 0.1,
        _cash_var_small(n_mean, C_mean),
        jnp.where(
            n_mean <= 0.2,
            -262. * n_mean**4 + 195. * n_mean**3 - 51.24 * n_mean**2 + 4.34 * n_mean + 0.77005,
            jnp.where(
                n_mean <= 0.3,
                4.23 * n_mean**2 - 2.8254 * n_mean + 1.12522,
                jnp.where(
                    n_mean <= 0.5,
                    -3.7 * n_mean**3 + 7.328 * n_mean**2 - 3.6926 * n_mean + 1.20641,
                    jnp.where(
                        n_mean <= 1.,
                        1.28 * n_mean**4 - 5.191 * n_mean**3 + 7.666 * n_mean**2 - 3.5446 * n_mean + 1.15431,
                        jnp.where(
                            n_mean <= 2.,
                            0.1125 * n_mean**4 - 0.641 * n_mean**3 + 0.859 * n_mean**2 + 1.0914 * n_mean - 0.05748,
                            jnp.where(
                                n_mean <= 3.,
                                0.089 * n_mean**3 - 0.872 * n_mean**2 + 2.8422 * n_mean - 0.67539,
                                jnp.where(
                                    n_mean <= 5.,
                                    2.12336 + 0.012202 * n_mean**(5.717 - 2.6 * jnp.log(jnp.maximum(n_mean, 1e-30))),
                                    jnp.where(
                                        n_mean <= 10.,
                                        2.05159 + 0.331 * n_mean**(1.343 - jnp.log(jnp.maximum(n_mean, 1e-30))),
                                        12. / n_mean**3 + 0.79 / n_mean**2 + 0.6747 / n_mean + 2.
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return (C_mean, C_var)


def _cash_var_small(n_mean, C_mean):
    """Helper for Cash variance computation when n_mean <= 0.1."""
    C_var = 0.
    for j in range(5):
        jf = float(j)
        log_j_over_n = jnp.where(j > 0, jnp.log(jf / jnp.maximum(n_mean, 1e-30)), 0.)
        term = jnp.exp(-n_mean) * n_mean**jf / math.factorial(j) * (n_mean - jf + jf * log_j_over_n)**2
        C_var = C_var + term
    C_var = 4. * C_var - C_mean**2
    return C_var


# =====================================================================
# Multiprocessing replacement (no-op in JAX version)
# =====================================================================

def launch_multiprocessing(function, n_cores):
    """In the JAX version, multiprocessing is replaced by vmap/vectorization.
    This function is kept for API compatibility but always runs single-core."""
    return function(0, {})


# =====================================================================
# Sampling functions
# =====================================================================

def rejection_sample_1d(x, pdf, n_samples):

    samples = np.zeros(n_samples)

    for i in range(0, n_samples):

        pdf_eval = 0.
        pdf_sample = 1.

        while pdf_sample > pdf_eval:

            x_sample = np.random.rand() * (x[-1] - x[0]) + x[0]
            pdf_sample = np.random.rand() * (pdf[-1] - pdf[0]) + pdf[0]
            pdf_eval = np.interp(x_sample, x, pdf)

        samples[i] = x_sample

    return samples


# =====================================================================
# Array tiling functions
# =====================================================================

def tile_1d_array(a, n_dim_output):

    grid = jnp.meshgrid(*([a] * n_dim_output), indexing='ij')
    custom_array = grid[0]

    return custom_array


def tile_1d_array_different_dim(original_array, n_dim_output, n_additional_dim):

    m = n_dim_output
    l = n_additional_dim

    result_shape = (l,) * m + (original_array.size,)
    result_array = original_array.reshape((1,) * m + original_array.shape)

    for i in range(m):
        result_array = jnp.repeat(result_array, l, axis=i)

    return result_array


# =====================================================================
# Interpolation helper functions
# =====================================================================

def interpolate_deep(x_interp, x, f):
    """Vectorized interpolation: for each row i, interpolate f[i,:] at x_interp[i]."""
    def _single_interp(xi, fi):
        return jnp.interp(xi, x, fi)

    return jax.vmap(_single_interp)(x_interp, f)


def sample_from_uniform(x_min, x_max, n=1):

    sample = np.random.uniform(low=x_min, high=x_max, size=n)

    if len(sample) == 1:

        sample = sample[0]

    return sample


def sample_from_gaussian(mu, sigma, n=1):

    sample = np.random.normal(loc=mu, scale=sigma, size=n)

    if len(sample) == 1:

        sample = sample[0]

    return sample
