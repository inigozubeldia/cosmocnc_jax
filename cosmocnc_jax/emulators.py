"""Direct CosmoPowerJAX emulator interface for JIT-compatible calls.

Bypasses the Cython/Class_szfast/predict() overhead by calling
CosmoPowerJAX_custom._predict() directly with JAX arrays.

The key insight: CosmoPowerJAX_custom._predict() is pure JAX
(jnp.dot + activation functions + standardization). The NN weights
are captured in the closure as constants. This allows JIT compilation
of the full emulator call chain.

Usage:
    emulators, param_orders, z_interp = init_emulators('lcdm')
    predict_H, predict_DA, predict_pk, predict_der = make_predict_fns(...)
    find_lnAs = make_sigma8_solver(...)
"""
import jax
import jax.numpy as jnp
import numpy as np


def init_emulators(cosmo_model='lcdm'):
    """Load emulator objects and extract metadata. Call once at init time.

    Returns:
        emulators: dict of emulator objects {pkl, h, da, der}
        param_orders: dict of parameter name lists for each emulator
        z_interp: (5000,) z-grid for H/DA interpolation (0 to 20)
    """
    from classy_szfast.cosmopower_jax import (
        cp_pkl_nn_jax, cp_h_nn_jax, cp_da_nn_jax, cp_der_nn_jax
    )
    emulators = {
        'pkl': cp_pkl_nn_jax[cosmo_model],
        'h': cp_h_nn_jax[cosmo_model],
        'da': cp_da_nn_jax[cosmo_model],
        'der': cp_der_nn_jax[cosmo_model],
    }
    param_orders = {k: list(em.parameters) for k, em in emulators.items()}
    z_interp = jnp.linspace(0., 20., 5000)  # matches Class_szfast.cp_z_interp
    return emulators, param_orders, z_interp


def _call_emulator(emulator, input_vec):
    """Call emulator._predict() directly. Pure JAX, JIT-compatible.

    The emulator object is captured by closure. Its weights, biases,
    hyper_params, and standardization arrays become JAX constants in the
    compiled XLA graph. The ten_to_predictions check in _predict controls
    whether 10^x is applied.

    Args:
        emulator: CosmoPowerJAX_custom instance
        input_vec: shape (n_params,) or (n_samples, n_params) JAX array

    Returns:
        predictions: shape (n_samples, n_modes) or (n_modes,) if squeezed
    """
    if input_vec.ndim == 1:
        input_vec = input_vec[None, :]
    return emulator._predict(
        emulator.weights, emulator.hyper_params,
        emulator.param_train_mean, emulator.param_train_std,
        emulator.feature_train_mean, emulator.feature_train_std,
        input_vec)


def build_cosmo_vec(pvd, param_order):
    """Build ordered JAX array from params_values_dict matching emulator params.

    Args:
        pvd: dict with keys like 'omega_b', 'H0', etc.
        param_order: list of parameter names in emulator's expected order

    Returns:
        (n_params,) JAX array
    """
    return jnp.array([pvd[k] for k in param_order])


def extract_pk_power_fac(emulators, param_orders, classy, pvd):
    """Extract the PKL conversion factor by comparing direct vs Cython output.

    The PKL emulator outputs log10(D_l)-like values. The Cython path applies
    10^x * pk_power_fac to get P(k). We extract pk_power_fac by comparing.

    Also extracts the k-array from the Cython path.

    Args:
        emulators: dict from init_emulators
        param_orders: dict from init_emulators
        classy: Cython Class_sz object
        pvd: params_values_dict

    Returns:
        pk_power_fac: (n_k,) array, conversion from 10^(raw) to P(k)
        k_arr: (n_k,) array, wavenumber grid
    """
    z_test = 0.5

    # Get P(k) through Cython (ground truth)
    pk_cython, k_arr_np = classy.get_pkl_at_z(z_test, params_values_dict=pvd)
    pk_cython = np.asarray(pk_cython).ravel()
    k_arr = jnp.asarray(k_arr_np).ravel()

    # Get raw emulator output (same as predict() since ten_to_predictions=False)
    pkl_param_order = param_orders['pkl']
    # Build input: cosmo params + z
    cosmo_keys = [k for k in pkl_param_order if k != 'z_pk_save_nonclass']
    cosmo_vec = jnp.array([pvd[k] for k in cosmo_keys])
    input_vec = jnp.concatenate([cosmo_vec, jnp.array([z_test])])
    raw = np.asarray(_call_emulator(emulators['pkl'], input_vec)).ravel()

    # pk_cython = 10^raw * pk_power_fac  =>  pk_power_fac = pk_cython / 10^raw
    pk_from_raw = np.power(10., raw)
    pk_power_fac = pk_cython / pk_from_raw

    return jnp.asarray(pk_power_fac), k_arr


def make_predict_fns(emulators, param_orders, z_interp, pk_power_fac):
    """Create JIT-compiled prediction functions for each emulator.

    Args:
        emulators: dict from init_emulators
        param_orders: dict from init_emulators
        z_interp: (5000,) z-grid
        pk_power_fac: (n_k,) PKL conversion factor

    Returns:
        predict_H_at_z: (cosmo_vec_h, z_vec) -> H/c at z
        predict_DA_at_z: (cosmo_vec_da, z_vec) -> D_A at z
        predict_pk_batch: (cosmo_vec_pkl, z_vec) -> (n_z, n_k) P(k,z)
        predict_der: (cosmo_vec_der) -> derived params array
    """
    pkl_em = emulators['pkl']
    h_em = emulators['h']
    da_em = emulators['da']
    der_em = emulators['der']

    @jax.jit
    def predict_H_at_z(cosmo_vec_h, z_vec):
        """H/c at desired z. H emulator has ten_to_predictions=True.
        _predict().squeeze() returns (5000,) for single-sample input."""
        H_grid = _call_emulator(h_em, cosmo_vec_h)  # (5000,) — LINEAR H/c
        return jnp.interp(z_vec, z_interp, H_grid)

    @jax.jit
    def predict_DA_at_z(cosmo_vec_da, z_vec):
        """D_A at desired z. DA emulator has ten_to_predictions=False,
        returns LINEAR D_A. Convert to chi, interpolate, convert back."""
        DA_grid = _call_emulator(da_em, cosmo_vec_da)  # (5000,) — LINEAR D_A
        chi_grid = DA_grid * (1. + z_interp)  # chi = D_A * (1+z)
        chi_at_z = jnp.interp(z_vec, z_interp, chi_grid)
        return chi_at_z / (1. + z_vec)  # D_A = chi / (1+z)

    def _predict_pk_single_z(cosmo_vec_pkl, z):
        """P(k) at single z. PKL has ten_to_predictions=False,
        returns log10(D_l)-like values. Apply 10^x * pk_power_fac."""
        input_vec = jnp.concatenate([cosmo_vec_pkl, jnp.array([z])])
        raw = _call_emulator(pkl_em, input_vec)  # (n_k,) — log10 values
        return jnp.power(10., raw) * pk_power_fac  # -> P(k)

    predict_pk_batch = jax.jit(jax.vmap(_predict_pk_single_z, in_axes=(None, 0)))

    @jax.jit
    def predict_der(cosmo_vec_der):
        """Derived params. DER has ten_to_predictions=True (default),
        returns LINEAR derived quantities."""
        return _call_emulator(der_em, cosmo_vec_der)

    return predict_H_at_z, predict_DA_at_z, predict_pk_batch, predict_der


def make_sigma8_solver(der_emulator, lnAs_index):
    """Create JIT-compiled sigma8→lnAs Newton solver using jax.grad.

    Uses analytical derivatives through the DER NN forward pass,
    eliminating the need for finite-difference approximations.

    Args:
        der_emulator: CosmoPowerJAX_custom for derived params
        lnAs_index: index of 'ln10^{10}A_s' in the DER param order

    Returns:
        find_lnAs: JIT-compiled function (cosmo_vec_der, sigma8_target, lnAs_init) -> lnAs
    """
    def sigma8_from_lnAs(cosmo_vec_der, lnAs):
        """sigma8 as a pure function of lnAs (for jax.grad)."""
        vec = cosmo_vec_der.at[lnAs_index].set(lnAs)
        der = _call_emulator(der_emulator, vec)  # (n_der,) after squeeze
        return der[1]  # sigma8 is index 1

    grad_sigma8 = jax.grad(sigma8_from_lnAs, argnums=1)

    @jax.jit
    def find_lnAs(cosmo_vec_der, sigma8_target, lnAs_init):
        """Newton iteration: find lnAs such that sigma8(lnAs) = target.
        Uses fori_loop with fixed iterations (avoids while_loop dispatch overhead)."""
        def body(_, lnAs):
            s8 = sigma8_from_lnAs(cosmo_vec_der, lnAs)
            ds8 = grad_sigma8(cosmo_vec_der, lnAs)
            return lnAs - (s8 - sigma8_target) / ds8

        return jax.lax.fori_loop(0, 8, body, lnAs_init)

    return find_lnAs
