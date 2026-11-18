"""Helpers that let the per-evaluation driver code run both on concrete values and under a
JAX trace (jax.jit / jax.vmap of the whole likelihood evaluation).

With concrete inputs every helper does exactly what the code it replaces did (float(x),
NumPy), so the ordinary evaluation path is unchanged. With traced inputs the value is kept
as a tracer and the JAX twin of the NumPy function is used.
"""
import numpy as np
import jax
import jax.numpy as jnp


def is_traced(*xs):
    """True if any argument is a JAX tracer."""
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def to_float(x):
    """float(x) for a concrete value; the tracer itself under a trace."""
    return x if isinstance(x, jax.core.Tracer) else float(x)


def xp_for(*xs):
    """NumPy for concrete arguments, jax.numpy if any argument is traced."""
    return jnp if is_traced(*xs) else np


def trapezoid(y, x, xp):
    """Trapezoidal rule with the given array module (NumPy: np.trapezoid, as before)."""
    if xp is np:
        return np.trapezoid(y, x)
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
