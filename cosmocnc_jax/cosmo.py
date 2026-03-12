import numpy as np
import jax.numpy as jnp

import sys
from .config import *
from .hmf import *
# scipy.integrate removed -- replaced with JAX trapezoid quadrature
import time

#for now only lcdm


class cosmology_model:

    def __init__(self,cosmo_params=None,cosmology_tool="classy_sz_jax",
    amplitude_parameter="sigma_8",cnc_params = None,logger = None):

        self.cnc_params = cnc_params

        self.logger = logging.getLogger(__name__)

        # if cosmo_params is None:

        #     cosmo_params = cosmo_params_default

        self.cosmo_params = cosmo_params
        self.amplitude_parameter = amplitude_parameter

        self.logger.info(f'Cosmology params: {self.cosmo_params}')
        # if self.cnc_params["cosmo_model"] != self.cnc_params["class_sz_cosmo_model"]:
        #     self.logger.warning(f'Cosmology model in cosmocnc params and classy_sz params do not match. Using classy_sz params.')
        #     self.cnc_params["class_sz_cosmo_model"] = self.cnc_params["cosmo_model"]
        #

        if cosmology_tool == "cobaya":

            cobaya_cosmology = cobaya_cosmo(self.cnc_params)

            self.power_spectrum = cobaya_cosmology
            self.background_cosmology = cobaya_cosmology
            self.background_cosmology.H0.value = cobaya_cosmology.H(0).value
            h = self.background_cosmology.H0.value/100.

            self.cosmo_params["Om0"] = cobaya_cosmology.Om(0.)
            self.cosmo_params["Ob0"] = cobaya_cosmology.Ob(0.)
            self.cosmo_params["sigma_8"] = cobaya_cosmology.sigma8(0.)
            self.cosmo_params["Onu0"] = cobaya_cosmology.Omega_nu_massive(0.)
            self.cosmo_params["n_s"] = cobaya_cosmology.ns
            self.cosmo_params["h"] = h


            self.z_CMB = cobaya_cosmology.z_cmb
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value


        elif cosmology_tool == "classy_sz_jax":

            from classy_sz import Class as Class_sz

            self.classy = Class_sz()

            # Build params_values_dict in classy_sz format
            self._build_params_values_dict()

            self.cosmo_model_dict = {'lcdm' : 0,
                                     'mnu'  : 1,
                                     'neff' : 2,
                                     'wcdm' : 3,
                                     'ede'  : 4,
                                     'mnu-3states' : 5,
                                     'ede-v2'  : 6,
                                     }

            # Set up classy_sz with JAX mode enabled
            classy_init_params = {
                'H0': self.cosmo_params["h"]*100.,
                'tau_reio':  self.cosmo_params["tau_reio"],
                'n_s': self.cosmo_params["n_s"],

                'output': self.cnc_params["class_sz_output"],

                'jax': 1,

                'HMF_prescription_NCDM': 1,
                'no_spline_in_tinker': 1,

                'M_min' : self.cnc_params["M_min"]*0.5,
                'M_max' : self.cnc_params["M_max"]*1.2,
                'z_min' : self.cnc_params["z_min"]*0.8,
                'z_max' : self.cnc_params["z_max"]*1.2,

                # Use a small fixed ndim_redshifts for classy_sz init
                # (the fine z-grid is only needed for HMF/simulator, which
                # uses JAX emulators directly, not classy_sz Cython)
                'ndim_redshifts' : min(self.cnc_params["n_z"], 100),
                'ndim_masses' : self.cnc_params["class_sz_ndim_masses"],
                'concentration_parameter': self.cnc_params["class_sz_concentration_parameter"],
                'cosmo_model': self.cosmo_model_dict[self.cnc_params['cosmo_model']],
                'mass_function' : self.cnc_params["class_sz_hmf"],

                'use_m500c_in_ym_relation' : self.cnc_params["class_sz_use_m500c_in_ym_relation"],
                'use_m200c_in_ym_relation' : self.cnc_params["class_sz_use_m200c_in_ym_relation"],
            }

            if self.cnc_params["cosmo_param_density"] == "critical":
                classy_init_params['omega_b'] = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
                classy_init_params['omega_cdm'] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2
                self.cosmo_params["Ob0h2"] = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = classy_init_params['omega_cdm']
            elif self.cnc_params["cosmo_param_density"] == "physical":
                classy_init_params['omega_b'] = self.cosmo_params["Ob0h2"]
                classy_init_params['omega_cdm'] = self.cosmo_params["Oc0h2"]
                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2
                self.cosmo_params["Om0"] = (self.cosmo_params["Oc0h2"]+self.cosmo_params["Ob0h2"])/self.cosmo_params["h"]**2
            elif self.cnc_params["cosmo_param_density"] == "mixed":
                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2
                classy_init_params['omega_b'] = self.cosmo_params["Ob0h2"]
                classy_init_params['omega_cdm'] = self.cosmo_params["Oc0h2"]

            if self.cnc_params['cosmo_model'] == "wcdm":
                classy_init_params['Omega_Lambda'] = 0.
                classy_init_params['w0_fld'] = self.cosmo_params["w0"]

            # Step 1: If sigma_8 is the amplitude parameter, first do a non-JAX init
            # to find A_s (find_As doesn't work in JAX mode due to emulator API mismatch)
            if self.amplitude_parameter == "sigma_8":
                classy_init_params_nonjax = dict(classy_init_params)
                classy_init_params_nonjax.pop('jax', None)
                classy_init_params_nonjax['sigma8'] = self.cosmo_params["sigma_8"]
                self.classy.set(classy_init_params_nonjax)
                self.logger.info('computing class_szfast (non-JAX, to find A_s from sigma_8)')
                self.classy.compute_class_szfast()
                self.As = np.exp(self.classy.get_current_derived_parameters(["ln10^{10}A_s"])["ln10^{10}A_s"])/1e10
                self.cosmo_params["A_s"] = self.As
                self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
                self.cosmo_params["sigma_8"] = self.sigma8
                self.logger.info(f'Found A_s={self.As:.6e} from sigma_8={self.sigma8:.6f}')

            elif self.amplitude_parameter == "A_s":
                self.As = self.cosmo_params["A_s"]

            # Step 2: Re-initialize with JAX mode using A_s
            self.classy = Class_sz()
            classy_init_params['ln10^{10}A_s'] = np.log(self.cosmo_params["A_s"]*1e10)
            self.classy.set(classy_init_params)

            self.logger.info('computing class_szfast (JAX mode)')

            # Monkey-patch calculate_sigma: the original uses in-place array
            # assignment (var[:,iz] = ...) which is incompatible with JAX arrays.
            # This version converts to NumPy first, computes, then stores results.
            from classy_szfast.classy_szfast import Class_szfast
            from mcfit import TophatVar
            _orig_calculate_sigma = Class_szfast.calculate_sigma

            def _calculate_sigma_numpy(csz_self, **kw):
                k = np.asarray(csz_self.cszfast_pk_grid_k)
                P = np.asarray(csz_self.cszfast_pk_grid_pk)
                var = P.copy()
                dvar = P.copy()
                for iz, zp in enumerate(csz_self.cszfast_pk_grid_z):
                    R, var[:, iz] = TophatVar(k, lowring=True)(P[:, iz], extrap=True)
                    dvar[:, iz] = np.gradient(var[:, iz], R)
                csz_self.cszfast_pk_grid_lnr = np.log(R)
                csz_self.cszfast_pk_grid_sigma2 = var
                csz_self.cszfast_pk_grid_sigma2_flat = var.flatten()
                csz_self.cszfast_pk_grid_lnsigma2_flat = 0.5 * np.log(var.flatten())
                csz_self.cszfast_pk_grid_dsigma2 = dvar
                csz_self.cszfast_pk_grid_dsigma2_flat = dvar.flatten()
                return 0

            Class_szfast.calculate_sigma = _calculate_sigma_numpy
            try:
                self.classy.compute_class_szfast()
            finally:
                Class_szfast.calculate_sigma = _orig_calculate_sigma

            self.logger.info('computing class_szfast done')

            # Rebuild pvd with correct A_s (found in step 1 or from cosmo_params)
            self._build_params_values_dict()

            # Initialize direct emulator interface (bypasses Cython for MCMC)
            from cosmocnc_jax.emulators import (
                init_emulators, extract_pk_power_fac, make_predict_fns,
                make_sigma8_solver
            )
            self._emu, self._emu_param_orders, self._z_interp = init_emulators(
                self.cnc_params['cosmo_model'])
            self._pk_power_fac, self._k_arr = extract_pk_power_fac(
                self._emu, self._emu_param_orders, self.classy, self._pvd)
            self._predict_H, self._predict_DA, self._predict_pk_batch, self._predict_der = \
                make_predict_fns(self._emu, self._emu_param_orders,
                                 self._z_interp, self._pk_power_fac)
            lnAs_index = self._emu_param_orders['der'].index('ln10^{10}A_s')
            self._find_lnAs = make_sigma8_solver(self._emu['der'], lnAs_index)
            self._lnAs_index = lnAs_index

            # Extract derived quantities using direct emulators
            self._extract_derived_from_jax()

            # Create JAX-compatible wrapper for background/power spectrum
            self.power_spectrum = classy_sz_jax_cosmo(self.classy, self.cosmo_params, self._pvd)
            self.background_cosmology = classy_sz_jax_cosmo(self.classy, self.cosmo_params, self._pvd)
            self.background_cosmology.H0.value = self.cosmo_params["h"]*100.

            # Mass conversion functions still go through C code (not in hot path)
            self.get_m500c_to_m200c_at_z_and_M = np.vectorize(self.classy.get_m500c_to_m200c_at_z_and_M)
            self.get_m200c_to_m500c_at_z_and_M = np.vectorize(self.classy.get_m200c_to_m500c_at_z_and_M)
            self.get_c200c_at_m_and_z = np.vectorize(self.classy.get_c200c_at_m_and_z_D08)
            self.get_dndlnM_at_z_and_M = np.vectorize(self.classy.get_dndlnM_at_z_and_M)
            self.get_delta_mean_from_delta_crit_at_z = np.vectorize(self.classy.get_delta_mean_from_delta_crit_at_z)

        print("cosmo params",self.cosmo_params)


    def _build_params_values_dict(self):
        """Build the params_values_dict in classy_sz emulator format from cosmo_params."""
        cp = self.cosmo_params
        h = cp["h"]
        self._pvd = {
            'H0': h * 100.,
            'omega_b': cp["Ob0"] * h**2,
            'omega_cdm': (cp["Om0"] - cp["Ob0"]) * h**2,
            'tau_reio': cp["tau_reio"],
            'n_s': cp["n_s"],
            'ln10^{10}A_s': np.log(cp.get("A_s", 2.1e-9) * 1e10),
            'm_ncdm': cp.get("m_nu", 0.06),
        }

    def _find_As_from_sigma8_jax(self, sigma8_target):
        """Find A_s from sigma_8 using JIT'd DER emulator Newton solver."""
        from cosmocnc_jax.emulators import build_cosmo_vec
        cosmo_vec_der = build_cosmo_vec(self._pvd, self._emu_param_orders['der'])
        lnAs_init = self._pvd['ln10^{10}A_s']
        lnAs = float(self._find_lnAs(
            cosmo_vec_der, jnp.float64(sigma8_target), jnp.float64(lnAs_init)))
        A_s = np.exp(lnAs) / 1e10
        return A_s, lnAs

    def _extract_derived_from_jax(self):
        """Extract derived cosmological quantities using direct DER emulator."""
        from cosmocnc_jax.emulators import build_cosmo_vec

        # If sigma_8 is the amplitude parameter, find A_s first
        if self.amplitude_parameter == "sigma_8":
            sigma8_target = self.cosmo_params["sigma_8"]
            A_s, lnAs = self._find_As_from_sigma8_jax(sigma8_target)
            self.cosmo_params["A_s"] = A_s
            self.As = A_s
            self._pvd['ln10^{10}A_s'] = lnAs

        # Direct DER emulator call (bypasses Cython)
        cosmo_vec_der = build_cosmo_vec(self._pvd, self._emu_param_orders['der'])
        der = self._predict_der(cosmo_vec_der)
        # der array: [100*theta_s, sigma8, YHe, z_reio, Neff, tau_rec, z_rec, rs_rec, ra_rec, ...]
        self.sigma8 = float(der[1])
        self.cosmo_params["sigma_8"] = self.sigma8
        self.N_eff = float(der[4])
        self.z_CMB = float(der[6])  # z_rec

        if self.amplitude_parameter == "A_s":
            self.As = self.cosmo_params["A_s"]

        # D_CMB from DER emulator: der[8] = ra_rec (comoving dist to recomb in Mpc)
        # D_CMB = da_rec = ra_rec / (1 + z_rec) = angular diameter distance at recombination
        self.D_CMB = float(der[8]) / (1. + self.z_CMB)

        # T_CMB and neutrinos (constant, from classy — only needed at init)
        if not hasattr(self, 'T_CMB_0'):
            self.T_CMB_0 = self.classy.T_cmb()
        if not hasattr(self, 'Omega_nu'):
            self.Omega_nu = self.classy.Omega_nu
        self.cosmo_params["Onu0"] = self.Omega_nu

        # Precompute Omega components for Delta conversion (matching get_all_relevant_params)
        h = self.cosmo_params["h"]
        Ob = self._pvd['omega_b'] / h**2
        Ocdm = self._pvd['omega_cdm'] / h**2
        m_ncdm = self._pvd.get('m_ncdm', 0.06)
        deg_ncdm = 1  # lcdm default
        Oncdm = deg_ncdm * m_ncdm / (93.14 * h**2)
        self._Om0 = Ocdm + Ob + Oncdm
        self._Om0_nonu = self._Om0 - Oncdm
        # Radiation: Omega_gamma from Stefan-Boltzmann
        sigma_B = 5.670374419e-8  # W/m²/K⁴
        _c = 2.99792458e8
        _G = 6.67428e-11
        _Mpc_m = 3.085677581282e22
        Og = (4. * sigma_B / _c * self.T_CMB_0**4) / (
            3. * _c**2 * 1e10 * h**2 / _Mpc_m**2 / 8. / np.pi / _G)
        N_ur = 2.0328  # lcdm default (N_eff=3.046 with deg_ncdm=1)
        Our = N_ur * 7./8. * (4./11.)**(4./3.) * Og
        self._Or0 = Our + Og
        self._Ol0 = 1. - Og - Ob - Ocdm - Oncdm - Our

    def _Omega_m_z_nonu(self, z):
        """Omega_m(z) without neutrinos — for Delta conversion.
        Matches classy.pyx get_delta_mean_from_delta_crit_at_z (lines 3266-3275)."""
        z1 = 1. + z
        return self._Om0_nonu * z1**3 / (
            self._Om0 * z1**3 + self._Ol0 + self._Or0 * z1**4)

    def update_cosmology(self,cosmo_params_new,cosmology_tool = "astropy"):

        self.cosmo_params = cosmo_params_new

        if cosmology_tool == "classy_sz_jax":

            # Fast path: just update params_values_dict, NO compute_class_szfast()
            self._build_params_values_dict()

            # Re-extract derived quantities from JAX emulator (fast: ~ms)
            self._extract_derived_from_jax()

            # Update wrappers with new params
            self.power_spectrum = classy_sz_jax_cosmo(self.classy, self.cosmo_params, self._pvd)
            self.background_cosmology = classy_sz_jax_cosmo(self.classy, self.cosmo_params, self._pvd)
            self.background_cosmology.H0.value = self.cosmo_params["h"]*100.

        elif cosmology_tool == "cobaya":

            cobaya_cosmology = cobaya_cosmo(self.cnc_params)

            self.power_spectrum = cobaya_cosmology
            self.background_cosmology = cobaya_cosmology
            self.background_cosmology.H0.value = cobaya_cosmology.H(0).value
            h = self.background_cosmology.H0.value/100.

            self.cosmo_params["Om0"] = cobaya_cosmology.Om(0.)
            self.cosmo_params["Ob0"] = cobaya_cosmology.Ob(0.)
            self.cosmo_params["sigma_8"] = cobaya_cosmology.sigma8(0.)
            self.cosmo_params["Onu0"] = cobaya_cosmology.Omega_nu_massive(0.)
            self.cosmo_params["n_s"] = cobaya_cosmology.ns
            self.cosmo_params["h"] = h

            self.z_CMB = cobaya_cosmology.z_cmb
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

        theta_mc = self.get_theta_mc()

    def get_theta_mc(self):

        Ogamma0 = 2.47282*10.**(-5)/self.cosmo_params["h"]**2
        Orad0 =  4.18343*10.**(-5)/self.cosmo_params["h"]**2
        Om0 = self.cosmo_params["Om0"]
        Ob0 = self.cosmo_params["Ob0"]
        OL0 = 1.-Om0-Orad0

        a_cmb = 1./(1.+self.z_CMB)

        # JAX trapezoid quadrature (replaces scipy.integrate.quad)
        x_quad = jnp.linspace(0., a_cmb, 500)
        integrand = 1./jnp.sqrt((1.+3.*Ob0*x_quad/(4.*Ogamma0))*(OL0*x_quad**4+Om0*x_quad+Orad0))
        r_sound = jnp.trapezoid(integrand, x_quad)/(self.cosmo_params["h"]*100.*jnp.sqrt(3.))*constants().c_light/1e3/self.z_CMB
        theta_mc = r_sound/self.D_CMB

        return theta_mc

    def get_z_cmb(self):

        Ob0h2 = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
        Om0h2 = self.cosmo_params["Om0"]*self.cosmo_params["h"]**2

        g1 = 0.0783*(Ob0h2)**(-0.238)/(1.+39.5*(Ob0h2)**0.763)
        g2 = 0.56/(1.+21.1*(Ob0h2)**1.81)
        z_cmb = 1048.*(1.+0.00124*(Ob0h2)**(-0.738))*(1.+g1*Om0h2**g2)

        return z_cmb

    def get_Omega_nu(self):

        return self.Omega_nu


class classy_sz_jax_cosmo:
    """JAX-compatible wrapper for classy_sz background/power spectrum.

    Uses params_values_dict for all calls, enabling JAX tracing and
    avoiding repeated compute_class_szfast() calls.
    """

    def __init__(self, classy, cosmo_params, params_values_dict):
        self.classy = classy
        self.cosmo_params = cosmo_params
        self.pvd = params_values_dict
        self.const = constants()
        self._h = cosmo_params["h"]

    def get_linear_power_spectrum(self, redshift):
        pk, k = self.classy.get_pkl_at_z(redshift, params_values_dict=self.pvd)
        pk = jnp.asarray(pk)
        k = jnp.asarray(k)

        k_cutoff = self.cosmo_params.get("k_cutoff", 10.)
        if k_cutoff < 10:
            x = jnp.linspace(jnp.log10(0.1), jnp.log10(100.), 10000)
            centre = jnp.log10(0.5)
            width = jnp.log10(10.) - jnp.log10(0.1)
            suppression = -jnp.tanh((x - centre) / width * 4) * 0.15 + 0.85
            ps_cutoff = jnp.interp(jnp.log10(k), x + jnp.log10(0.677), suppression)
            pk = pk * ps_cutoff

        return (k, pk)

    def critical_density(self, z):
        conv_fac = 1. / (1000. * self.const.mpc**3 / self.const.solar)
        h = self._h
        rho = self.classy.get_rho_crit_at_z(z, params_values_dict=self.pvd)
        rho = jnp.asarray(rho)

        class result:
            value = rho * conv_fac * h**2

        return result

    def differential_comoving_volume(self, z):
        h = self._h
        vol = self.classy.get_volume_dVdzdOmega_at_z(z, params_values_dict=self.pvd)
        vol = jnp.asarray(vol)

        class result:
            value = vol * h**(-3)

        return result

    def angular_diameter_distance(self, z):
        da = self.classy.get_angular_distance_at_z(z, params_values_dict=self.pvd)
        da = jnp.asarray(da)

        class result:
            value = da

        return result

    def angular_diameter_distance_z1z2(self, z1, z2):
        da1 = jnp.asarray(self.classy.get_angular_distance_at_z(z1, params_values_dict=self.pvd))
        da2 = jnp.asarray(self.classy.get_angular_distance_at_z(z2, params_values_dict=self.pvd))
        z1 = jnp.asarray(z1)

        class result:
            value = -(1. / (1. + z2)) * (da1 * (1. + z1) - da2 * (1. + z2))

        return result

    def H(self, z):
        # get_hubble_at_z returns H(z) in 1/Mpc; convert to km/s/Mpc
        conv_fac = 299792.458
        hz = self.classy.get_hubble_at_z(z, params_values_dict=self.pvd)
        hz = jnp.asarray(hz)

        class result:
            value = hz * conv_fac

        return result

    class H0:
        value = 0


class cobaya_cosmo:

    def __init__(self,cnc_params):

        self.cnc_params = cnc_params
        self.const = constants()
        self.k_arr = np.geomspace(1e-4,50.,500) # same as in cosmopower, in Mpc-1
        self.provider = self.cnc_params["cobaya_provider"]
        self.cnc_params = cnc_params
        self.z_vec =  np.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.z_vec = np.concatenate([[0.],self.z_vec])
        self.ns = self.provider.get_param("ns")
        self.z_cmb = 1100.

    def get_linear_power_spectrum(self,redshift):

       # ps = self.provider.get_Pk_interpolator(nonlinear=False)(redshift,self.k_arr)[0,:]

        (k, z, ps) = self.provider.get_Pk_grid(nonlinear=False)
        index = np.abs(z-redshift).argmin()
        ps = np.interp(self.k_arr,k,ps[index,:])

        return (self.k_arr,ps)

    def critical_density(self,z):

        class result:

            conv_fac = self.const.solar/(1000.*self.const.mpc**3)
            Hz = self.H(z).value
            G = 4.301e-9  # Mpc M_sun^-1 (km/s)^2
            rho_crit = 3*Hz**2/(8*np.pi*G)

            value = rho_crit*conv_fac

        return result

    def differential_comoving_volume(self,z):

        class result:

            Hz = self.H(z).value  # km/s/Mpc
            DM = self.provider.get_comoving_radial_distance(z)  # Mpc
            c = 299792.458  # km/s
            value = c / Hz * DM**2

        return result

    def angular_diameter_distance(self,z):

        class result:

            value = self.provider.get_angular_diameter_distance(z)

        return result

    def angular_diameter_distance_z1z2(self,z1,z2):

        z_pairs = [(z, z2) for z in z1]

        class result:

            value = self.provider.get_angular_diameter_distance_2(z_pairs)

        return result

    def H(self,z):

        class result:

            value =  self.provider.get_Hubble(z)
        return result

    class H0:
        # def __init__(self):
            # conv_fac = 299792.458
            # class result:
        value = 0
        # return {'hubble':np.vectorize(self.classy.Hubble)(z)}
        # return result

    def Oc(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_cdm())
        return self.provider.get_Omega_cdm(z)

    def Ob(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_b())
        return self.provider.get_Omega_b(z)

    def Om(self,z):

        return self.Oc(z)+self.Ob(z)

    def Omega_nu_massive(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_nu_massive())
        return self.provider.get_Omega_nu_massive(z)

    def sigma8(self,z):

        #return np.interp(z,self.z_vec,self.provider.sigma8_z())
        return self.provider.get_sigma8_z(z)
