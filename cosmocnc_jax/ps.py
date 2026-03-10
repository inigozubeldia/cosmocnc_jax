import numpy as np
import jax.numpy as jnp
import pylab as pl
import warnings
warnings.filterwarnings("ignore")
import warnings
from contextlib import contextmanager
import logging

# Suppress absl warnings
@contextmanager
def suppress_warnings():
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        warnings.resetwarnings()

# Suppress TensorFlow warnings
import os

# Additional suppression
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# import cosmopower
import os
import subprocess
def _get_restore_nn():
    from .restore_nn import Restore_NN, Restore_PCAplusNN
    return Restore_NN, Restore_PCAplusNN
from .config import *
# scipy.optimize removed -- replaced with simple Newton iteration
import os
#cosmo_model = "lcdm", "mnu", "neff", "wcdm"

class cosmopower:

    def __init__(self,cosmo_model="lcdm",path=None):

        self.cosmo_params = None


        path_to_emulators = path + cosmo_model + "/"
        str_cmd_subprocess = ["ls",path_to_emulators]

        emulator_dict = {}
        emulator_dict[cosmo_model] = {}

        emulator_dict[cosmo_model]['TT'] = 'TT_v1'
        emulator_dict[cosmo_model]['TE'] = 'TE_v1'
        emulator_dict[cosmo_model]['EE'] = 'EE_v1'
        emulator_dict[cosmo_model]['PP'] = 'PP_v1'
        emulator_dict[cosmo_model]['PKNL'] = 'PKNL_v1'
        emulator_dict[cosmo_model]['PKL'] = 'PKL_v1'
        emulator_dict[cosmo_model]['DER'] = 'DER_v1'
        emulator_dict[cosmo_model]['DAZ'] = 'DAZ_v1'
        emulator_dict[cosmo_model]['HZ'] = 'HZ_v1'
        emulator_dict[cosmo_model]['S8Z'] = 'S8Z_v1'

        #LOAD THE EMULATORS

        self.cp_tt_nn = {}
        self.cp_te_nn = {}
        self.cp_ee_nn = {}
        self.cp_pp_nn = {}
        self.cp_pknl_nn = {}
        self.cp_pkl_nn = {}
        #self.cp_der_nn = {}
        #self.cp_da_nn = {}
        #self.cp_h_nn = {}
        #self.cp_s8_nn = {}

        self.mp = cosmo_model

        path_to_emulators = path + self.mp +'/'

        Restore_NN, Restore_PCAplusNN = _get_restore_nn()

        self.cp_tt_nn[self.mp] = Restore_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['TT'])

        self.cp_te_nn[self.mp] = Restore_PCAplusNN(restore=True,
                                        restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['TE'])

        self.cp_ee_nn[self.mp] = Restore_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['EE'])

        self.cp_pp_nn[self.mp] = Restore_NN(restore=True,
                                 restore_filename=path_to_emulators + 'PP/' + emulator_dict[self.mp]['PP'])

        self.cp_pknl_nn[self.mp] = Restore_NN(restore=True,
                                   restore_filename=path_to_emulators + 'PK/' + emulator_dict[self.mp]['PKNL'])

        self.cp_pkl_nn[self.mp] = Restore_NN(restore=True,
                                  restore_filename=path_to_emulators + 'PK/' + emulator_dict[self.mp]['PKL'])

        self.cp_der_nn = Restore_NN(restore=True,restore_filename=path_to_emulators + 'derived-parameters/DER_v1',)

        # Precompute constant k-grid arrays (same for all redshifts)
        ndspl = 10
        self._k_arr = np.geomspace(1e-4, 50., 5000)[::ndspl]
        self._ls = np.arange(2, 5000+2)[::ndspl]
        self._dls_inv = 1. / (self._ls * (self._ls + 1.) / 2. / np.pi)
        self._log10_k_arr = np.log10(self._k_arr)

    def get_sigma_8(self):

        return self.cp_der_nn.ten_to_predictions_np(self.params_cp)[0][1]

    def set_cosmology(self,H0=67.37,Ob0=0.02233/0.6737**2,Oc0=0.1198/0.6737**2,ln10A_s=3.043,tau_reio=0.0540,n_s=0.9652): # LambdaCDM parameters last column of Table 1 of https://arxiv.org/pdf/1807.06209.pdf:

        h = H0/100.

        self.params_settings = {
                           'H0': H0,
                           'omega_b':Ob0*h**2,
                           'omega_cdm': Oc0*h**2,
                           'ln10^{10}A_s':ln10A_s,
                           'tau_reio': tau_reio,
                           'n_s': n_s,
                           }

        self.params_cp = {}

        for key,value in self.params_settings.items():

            self.params_cp[key] = [value]

    def find_As(self,sigma_8):

        params_cp = self.params_cp

        def to_root(ln10_10_As):
            params_cp["ln10^{10}A_s"] = [ln10_10_As]
            return self.cp_der_nn.ten_to_predictions_np(params_cp)[0][1]-sigma_8

        # Simple Newton iteration (replaces scipy.optimize.root)
        x = 3.04
        for _ in range(20):
            fx = to_root(x)
            if abs(fx) < 1e-12:
                break
            dx = 1e-5
            dfx = (to_root(x + dx) - fx) / dx
            if abs(dfx) < 1e-30:
                break
            x = x - fx / dfx

        A_s = np.exp(x)/1e10

        return A_s

    def get_linear_power_spectrum(self,redshift):

        params_cp_pk = self.params_cp.copy()
        params_cp_pk['z_pk_save_nonclass'] = [redshift]
        prediction = self.cp_pkl_nn[self.mp].predictions_np(params_cp_pk)

        pkl = 10.**np.asarray(prediction[0]) * self._dls_inv

        k_cutoff = self.cosmo_params["k_cutoff"]
        ps_cutoff = self.cosmo_params["ps_cutoff"]

        if k_cutoff < 10:

            x = np.linspace(np.log10(0.1),np.log10(10.))
            centre = np.log10(0.5)
            width = (np.log10(10.)-np.log10(0.1))
            suppression = -np.tanh((x-centre)/width*4)*0.15+0.85

            ps_cutoff = np.interp(self._log10_k_arr,x+np.log10(0.677),suppression)

        pkl = pkl*ps_cutoff

        return (jnp.asarray(self._k_arr),jnp.asarray(pkl))
