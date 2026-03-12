import jax
import jax.numpy as jnp
import functools
import numpy as np
import time
import logging


# =====================================================================
# Tinker08 parameter arrays (module-level constants for JIT)
# =====================================================================

TINKER08_DELTA_LOG = jnp.log10(jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.], dtype=jnp.float64))
TINKER08_DELTA_LIN = jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.], dtype=jnp.float64)
TINKER08_A = jnp.array([0.186,0.2,0.212,0.218,0.248,0.255,0.260,0.260,0.260], dtype=jnp.float64)
TINKER08_a = jnp.array([1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66], dtype=jnp.float64)
TINKER08_b = jnp.array([2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41], dtype=jnp.float64)
TINKER08_c = jnp.array([1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44], dtype=jnp.float64)


# =====================================================================
# Pure JAX functions for JIT compilation
# =====================================================================

def f_sigma_jit(sigma, redshift, Delta, tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                interp_log=True):
    """Pure JAX Tinker08 multiplicity function for JIT compilation.

    All arguments must be JAX arrays/scalars. No Python objects.
    interp_log: bool (static). If True, interpolate in log10(Delta) space (default).
                If False, interpolate in linear Delta space.
    """
    alpha = 10.**(-(0.75/jnp.log10(Delta/75.))**1.2)

    Delta_interp = jnp.where(interp_log, jnp.log10(Delta), Delta)

    A = jnp.interp(Delta_interp, tinker_Delta, tinker_A) * (1.+redshift)**(-0.14)
    a = jnp.interp(Delta_interp, tinker_Delta, tinker_a) * (1.+redshift)**(-0.06)
    b = jnp.interp(Delta_interp, tinker_Delta, tinker_b) * (1.+redshift)**(-alpha)
    c = jnp.interp(Delta_interp, tinker_Delta, tinker_c)

    return A*((sigma/b)**(-a)+1.)*jnp.exp(-c/sigma**2)


def get_sigma_M_from_arrays(M_vec, rho_m, R_vec, sigma_vec, dsigma_vec):
    """Pure JAX function to interpolate sigma(M) and dsigma/dR(M) from precomputed arrays."""
    R = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
    sigma = jnp.interp(R, R_vec, sigma_vec)
    dsigmadR = jnp.interp(R, R_vec, dsigma_vec)
    return sigma, dsigmadR, R


def compute_hmf_single_z(sigma, dsigmadR, R, M_vec, rho_m, redshift, Delta,
                          tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                          volume_element_val, interp_log=True):
    """Pure JAX: compute HMF for a single redshift from precomputed sigma arrays.

    Returns hmf in log-mass units (dn/dlnM * dV/dz * 4pi).
    volume_element_val: dV/dz/dOmega value for this redshift (already computed outside JIT).
    interp_log: bool (static). If True, interpolate Tinker08 params in log10(Delta).
    """
    dMdR = 4.*jnp.pi*rho_m*R**2

    fsigma = f_sigma_jit(sigma, redshift, Delta, tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                          interp_log=interp_log)

    hmf = -fsigma*rho_m/M_vec/dMdR*dsigmadR/sigma
    M_eval = M_vec/1e14
    hmf = hmf*1e14

    # log=True: multiply by M_eval (= M/1e14) for dn/dlnM
    hmf = hmf*M_eval

    # Apply volume element
    hmf = hmf * volume_element_val

    return hmf


# Vectorized version over redshift dimension
# interp_log is not vmapped (None) — it's a scalar bool shared across redshifts
_compute_hmf_vmap_log = jax.vmap(
    functools.partial(compute_hmf_single_z, interp_log=True),
    in_axes=(0, 0, 0, None, None, 0, 0,
             None, None, None, None, None, 0))

_compute_hmf_vmap_lin = jax.vmap(
    functools.partial(compute_hmf_single_z, interp_log=False),
    in_axes=(0, 0, 0, None, None, 0, 0,
             None, None, None, None, None, 0))


@functools.partial(jax.jit, static_argnums=(14,))
def compute_hmf_matrix_jit(sigma_matrix, dsigma_matrix, R_matrix, M_vec, rho_m,
                            redshift_vec, Delta_vec, volume_element_vec,
                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                            M_min_cutoff, interp_log=True):
    """JIT-compiled: compute full HMF matrix from precomputed sigma arrays.

    Args:
        sigma_matrix: (n_z, n_points) sigma values
        dsigma_matrix: (n_z, n_points) dsigma/dR values
        R_matrix: (n_z, n_points) R values
        M_vec: (n_points,) mass vector
        rho_m: scalar mean matter density
        redshift_vec: (n_z,) redshift values
        Delta_vec: (n_z,) overdensity values (w.r.t. mean) per redshift
        volume_element_vec: (n_z,) dV/dz/dOmega values
        tinker_*: Tinker08 parameter arrays
        M_min_cutoff: minimum mass cutoff (or -1 for no cutoff)
        interp_log: bool (static). If True, use log10(Delta) interpolation.

    Returns:
        hmf_matrix: (n_z, n_points)
    """
    if interp_log:
        hmf_matrix = _compute_hmf_vmap_log(sigma_matrix, dsigma_matrix, R_matrix,
                                            M_vec, rho_m, redshift_vec, Delta_vec,
                                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                                            volume_element_vec)
    else:
        hmf_matrix = _compute_hmf_vmap_lin(sigma_matrix, dsigma_matrix, R_matrix,
                                            M_vec, rho_m, redshift_vec, Delta_vec,
                                            tinker_Delta, tinker_A, tinker_a, tinker_b, tinker_c,
                                            volume_element_vec)

    # Apply M_min_cutoff if needed
    cutoff_mask = jnp.where(M_vec < M_min_cutoff, 0., 1.)
    hmf_matrix = hmf_matrix * cutoff_mask[jnp.newaxis, :]

    return hmf_matrix


class halo_mass_function:

    def __init__(self,
                 cosmology=None,
                 hmf_type="Tinker08",
                 mass_definition="500c",
                 M_min=1e13,M_max=1e16,
                 M_min_cutoff=None,
                 n_points=1000,
                 type_deriv="numerical",
                 hmf_calc="cnc",
                 extra_params=None,
                 logger = None,
                 interp_tinker=None):

        self.hmf_type = hmf_type
        self.mass_definition = mass_definition
        self.cosmology = cosmology
        self.h = self.cosmology.background_cosmology.H0.value/100.

        self.M_min = M_min
        self.M_max = M_max
        self.M_min_cutoff = M_min_cutoff
        self.n_points = n_points
        self.type_deriv = type_deriv
        self.hmf_calc = hmf_calc
        self.extra_params = extra_params

        self.other_params = {"interp_tinker":interp_tinker}

        self.logger = logging.getLogger(__name__)

        self.sigma_r_dict = {}

        self.const = constants()

        if self.hmf_type == "Tinker08":

            self.rho_c_0 = self.cosmology.background_cosmology.critical_density(0.).value*self.const.mpc**3/self.const.solar*1e3

        if self.hmf_calc == "hmf":

            import hmf as hmf_package

            if self.mass_definition[-1] == "c":

                md = "SOCritical"

            elif self.mass_definition[-1] == "m":

                md = "SOMean"

            self.massfunc_hmf = hmf_package.MassFunction(Mmax=np.log10(self.M_max*self.h),
                                                         Mmin=np.log10(self.M_min*self.h),
                                                         z=0.,
                                                         mdef_model=md,
                                                         mdef_params={"overdensity":float(self.mass_definition[0:-1])},
                                                         cosmo_model=self.cosmology.background_cosmology,
                                                         dlog10m=0.005,
                                                         sigma_8=cosmology.cosmo_params["sigma_8"],
                                                         n=cosmology.cosmo_params["n_s"])

    def eval_hmf(self,redshift,log=False,volume_element=False,save_sigma_r=False,load_sigma_r=False,
    M_min=None,M_max=None,n_points=None):

        if M_min is None:

            M_min = self.M_min

        if M_max is None:

            M_max = self.M_max

        if n_points is None:

            n_points = self.n_points

        if log == False:

            M_vec = jnp.linspace(M_min,M_max,n_points)

        elif log == True:

            M_vec = jnp.exp(jnp.linspace(jnp.log(M_min),jnp.log(M_max),n_points))

        if self.hmf_calc == "cnc":

            if self.hmf_type == "Tinker08":

                rho_m = self.rho_c_0*self.cosmology.cosmo_params["Om0"]

                if load_sigma_r is False:

                    k,ps = self.cosmology.power_spectrum.get_linear_power_spectrum(redshift)
                    k = jnp.asarray(k)
                    ps = jnp.asarray(ps)
                    sigma_r = sigma_R((k,ps),cosmology=self.cosmology)
                    sigma_r.get_derivative(type_deriv=self.type_deriv)

                elif load_sigma_r is True:

                    z_indices_key = np.array([float(index) for index in list(self.sigma_r_dict.keys())])
                    z_index = str(z_indices_key[np.argmin(np.abs(z_indices_key-redshift))])
                    sigma_r = self.sigma_r_dict[z_index]

                if save_sigma_r is True:

                    self.sigma_r_dict[str(redshift)] = sigma_r

                t0 = time.time()

                (sigma,dsigmadR) = sigma_r.get_sigma_M(M_vec,rho_m,get_deriv=True)

                self.sigma = sigma
                self.dsigmadR = dsigmadR
                self.R = sigma_r.R_eval

                dMdR = 4.*jnp.pi*rho_m*self.R**2

                if self.mass_definition[-1] == "c":

                    if self.cosmology.cnc_params["cosmology_tool"] == "cobaya_cosmo":

                        rescale = self.cosmology.Om(redshift)/(self.cosmology.H(redshift)/100.)**2

                    else:

                        rescale = self.cosmology.cosmo_params["Om0"]*(1.+redshift)**3/(self.cosmology.background_cosmology.H(redshift).value/(self.cosmology.cosmo_params["h"]*100.))**2

                elif self.mass_definition[-1] == "m":

                    rescale = 1

                Delta = float(self.mass_definition[0:-1])/rescale

                fsigma = f_sigma(sigma,redshift=redshift,hmf_type=self.hmf_type,
                Delta=Delta,mass_definition=self.mass_definition,
                other_params=self.other_params)
                self.fsigma = fsigma

                hmf = -fsigma*rho_m/M_vec/dMdR*dsigmadR/sigma
                M_eval = M_vec

                hmf = hmf*1e14
                M_eval = M_eval/1e14

                if log == True:

                    hmf = hmf*M_eval
                    M_eval = jnp.log(M_eval)

        elif self.hmf_calc == "hmf":

            self.massfunc_hmf.update(z=redshift)
            hmf = jnp.asarray(self.massfunc_hmf.dndm*1e14*self.h**4)
            M_eval = jnp.asarray(self.massfunc_hmf.m/self.h/1e14)

            hmf = jnp.interp(M_vec/1e14,M_eval,hmf)
            M_eval = M_vec/1e14

            if log == True:

                hmf = hmf*M_eval
                M_eval = jnp.log(M_eval)

        elif self.hmf_calc == "MiraTitan": #only works if log == True, note that returns a matrix instead of a vector

            t0 = time.time()

            if log == True:

                MT_emulator = self.extra_params["emulator"]

                M_vec = np.linspace(M_min,M_max,n_points)

                cosmology_emulator = {
                "h": self.h,
                "Ommh2": self.cosmology.cosmo_params["Om0"]*self.h**2,
                "Ombh2": self.cosmology.cosmo_params["Ob0"]*self.h**2,
                "Omnuh2": self.cosmology.Omega_nu*self.h**2,
                "sigma_8": self.cosmology.cosmo_params["sigma_8"],
                "n_s": self.cosmology.cosmo_params["n_s"],
                "w_0": -1.,
                "w_a": 0.
                }

                hmf = jnp.asarray(np.array(MT_emulator.predict(cosmology_emulator,redshift,M_vec*self.h))[0,:,:]*self.h**3)
                M_eval = jnp.log(M_vec/1e14)

                if volume_element == True:

                    for i in range(0,hmf.shape[0]):

                        hmf = hmf.at[i,:].set(hmf[i,:]*self.cosmology.background_cosmology.differential_comoving_volume(redshift[i]).value)

        if volume_element == True and self.hmf_calc != "MiraTitan":

            hmf = hmf*self.cosmology.background_cosmology.differential_comoving_volume(redshift).value

        if self.M_min_cutoff is not None:

            cutoff_mask = jnp.where(M_vec < self.M_min_cutoff, 0., 1.)
            if hmf.ndim == 2:
                hmf = hmf * cutoff_mask[jnp.newaxis, :]
            else:
                hmf = hmf * cutoff_mask

        return M_eval,hmf


class sigma_R:
    """Computes the variance of the linear density field smoothed with a top-hat filter.

    Uses mcfit.TophatVar (FFTLog algorithm) with JAX backend for differentiable computation.
    Pre-built TophatVar objects can be passed via _tv0/_tv1 to avoid redundant constructor calls
    when the same k grid is used across multiple redshifts.
    """

    def __init__(self, ps, cosmology=None, deriv=0, _tv0=None, _tv1=None):

        self.cosmology = cosmology
        (self.k, self.pk) = ps

        # Use mcfit with JAX backend (no numpy/JAX conversions needed)
        if _tv0 is None:
            from mcfit import TophatVar
            _tv0 = TophatVar(np.asarray(self.k), lowring=True, deriv=0, backend='jax')

        self.R_vec, self.var_vec = _tv0(self.pk, extrap=True)
        self.sigma_vec = jnp.sqrt(self.var_vec)
        self._tv1 = _tv1

    def get_derivative(self, type_deriv="analytical"):

        if type_deriv == "analytical":

            if self._tv1 is None:
                from mcfit import TophatVar
                self._tv1 = TophatVar(np.asarray(self.k), lowring=True, deriv=1, backend='jax')

            _, dvar = self._tv1(self.pk * self.k, extrap=True)
            self.dsigma_vec = dvar / (2.0 * self.sigma_vec)

        elif type_deriv == "numerical":

            self.dsigma_vec = jnp.gradient(self.sigma_vec, self.R_vec)

    def get_sigma_M(self, M_vec, rho_m, get_deriv=False):

        R = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
        self.R_eval = R

        sigma = jnp.interp(R, self.R_vec, self.sigma_vec)

        if get_deriv == False:

            ret = sigma

        elif get_deriv == True:

            dsigmadR = jnp.interp(R, self.R_vec, self.dsigma_vec)
            ret = (sigma, dsigmadR)

        return ret


def build_batch_sigma_fns(tv0, tv1, k_arr, type_deriv="analytical"):
    """Build cached vmapped functions for batch sigma computation.

    Call once per TophatVar pair (i.e., once per k grid). Returns functions
    that can be called repeatedly without re-tracing.

    Args:
        tv0: TophatVar(k, deriv=0, backend='jax') -- pre-built
        tv1: TophatVar(k, deriv=1, backend='jax') -- pre-built
        k_arr: (n_k,) JAX array of wavenumbers
        type_deriv: "analytical" or "numerical"

    Returns:
        (vmap_sigma_fn, vmap_interp_fn, R_vec) -- cached vmapped functions
    """
    R_vec = jnp.asarray(tv0.y)

    if type_deriv == "analytical":
        def _single_z(pk):
            _, var = tv0(pk, extrap=True)
            _, dvar = tv1(pk * k_arr, extrap=True)
            sigma_raw = jnp.sqrt(var)
            dsigma_raw = dvar / (2.0 * sigma_raw)
            return sigma_raw, dsigma_raw
    else:
        def _single_z(pk):
            _, var = tv0(pk, extrap=True)
            sigma_raw = jnp.sqrt(var)
            dsigma_raw = jnp.gradient(sigma_raw, R_vec)
            return sigma_raw, dsigma_raw

    vmap_sigma_fn = jax.jit(jax.vmap(_single_z))

    def _interp_to_M(sigma_row, dsigma_row, R_M):
        return jnp.interp(R_M, R_vec, sigma_row), jnp.interp(R_M, R_vec, dsigma_row)

    vmap_interp_fn = jax.jit(jax.vmap(_interp_to_M, in_axes=(0, 0, None)))

    return vmap_sigma_fn, vmap_interp_fn, R_vec


def batch_sigma_R_from_tophat(tv0, tv1, pk_batch, k_arr, M_vec, rho_m,
                               type_deriv="analytical",
                               _cached_fns=None):
    """Batch compute sigma(M) and dsigma/dR(M) for multiple power spectra via vmap.

    Args:
        tv0: TophatVar(k, deriv=0, backend='jax') -- pre-built
        tv1: TophatVar(k, deriv=1, backend='jax') -- pre-built
        pk_batch: (n_z, n_k) JAX array of power spectra
        k_arr: (n_k,) JAX array of wavenumbers
        M_vec: (n_M,) JAX array of masses
        rho_m: float, mean matter density
        type_deriv: "analytical" (FFTLog deriv=1) or "numerical" (jnp.gradient)
        _cached_fns: optional (vmap_sigma_fn, vmap_interp_fn, R_vec) from build_batch_sigma_fns

    Returns:
        sigma_matrix: (n_z, n_M)
        dsigma_matrix: (n_z, n_M)
        R_matrix: (n_z, n_M)
    """
    if _cached_fns is not None:
        vmap_sigma_fn, vmap_interp_fn, R_vec = _cached_fns
    else:
        vmap_sigma_fn, vmap_interp_fn, R_vec = build_batch_sigma_fns(
            tv0, tv1, k_arr, type_deriv)

    # Batch FFTLog transforms
    sigma_raw_batch, dsigma_raw_batch = vmap_sigma_fn(pk_batch)

    # Interpolate from R grid to mass-based R values
    R_M = (3. * M_vec / (4. * jnp.pi * rho_m))**(1./3.)
    sigma_matrix, dsigma_matrix = vmap_interp_fn(sigma_raw_batch, dsigma_raw_batch, R_M)
    R_matrix = jnp.broadcast_to(R_M[None, :], sigma_matrix.shape)

    return sigma_matrix, dsigma_matrix, R_matrix


#Delta is w.r.t. mean

def f_sigma(sigma, redshift=None, hmf_type="Tinker08", Delta=None, mass_definition="500c", other_params=None):

    params = hmf_params(hmf_type=hmf_type, mass_definition=mass_definition, other_params=other_params)

    if hmf_type == "Tinker08":

        alpha = 10.**(-(0.75/jnp.log10(Delta/75.))**1.2)

        A = params.get_param("A", Delta)*(1.+redshift)**(-0.14)
        a = params.get_param("a", Delta)*(1.+redshift)**(-0.06)
        b = params.get_param("b", Delta)*(1.+redshift)**(-alpha)
        c = params.get_param("c", Delta)

        f = A*((sigma/b)**(-a)+1.)*jnp.exp(-c/sigma**2)

    return f


class hmf_params:

    def __init__(self, hmf_type="Tinker08", mass_definition="500c", other_params=None):

        self.hmf_type = hmf_type
        self.mass_definition = mass_definition
        self.other_params = other_params

        if self.hmf_type == "Tinker08":

            if other_params["interp_tinker"] == "log":

                Delta = jnp.log10(jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.]))

            elif other_params["interp_tinker"] == "linear":

                Delta = jnp.array([200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.])

            A = jnp.array([0.186,0.2,0.212,0.218,0.248,0.255,0.260,0.260,0.260])
            a = jnp.array([1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66])
            b = jnp.array([2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41])
            c = jnp.array([1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44])

            self.params = {"A":A,"b":b,"a":a,"c":c,"Delta":Delta}

    def get_param(self, param, Delta):

        if self.hmf_type == "Tinker08":

            if self.other_params["interp_tinker"] == "log":

                ret = jnp.interp(jnp.log10(Delta), self.params["Delta"], self.params[param])

            elif self.other_params["interp_tinker"] == "linear":

                ret = jnp.interp(Delta, self.params["Delta"], self.params[param])

        return ret


class constants:

    def __init__(self):

        self.c_light = 2.997924581e8
        self.G = 6.674*1e-11
        self.solar = 1.98855*1e30
        self.mpc = 3.08567758149137*1e22
        self.gamma =  self.G/self.c_light**2*self.solar/self.mpc
