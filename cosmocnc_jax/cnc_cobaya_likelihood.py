from cobaya.likelihood import Likelihood
from typing import Optional, Sequence
import numpy as np
import os

class theta_mc_prior(Likelihood):
    # variables from yaml
    theta_mc_mean: float
    theta_mc_std: float

    def initialize(self):

        self.minus_half_invvar = - 0.5 / self.theta_mc_std ** 2

    def get_requirements(self):

        return {'theta_mc': None}

    def logp(self, **params_values):

        theta_mc_theory = self.provider.get_param("theta_mc")
        return self.minus_half_invvar * (theta_mc_theory - self.theta_mc_mean) ** 2

class cnc_likelihood(Likelihood):

    def initialize(self):

        super().initialize()

    def get_requirements(self):

        return {"cluster_log_lik": {}}

    def _get_theory(self, **params_values):

        theory = self.provider.get_cluster_log_lik()

        return theory

    def logp(self, **params_values):

        _derived = params_values.pop("_derived", None)
        theory = self._get_theory(**params_values)
        # cnc.get_log_lik() (classy_sz_jax path) returns a 0-d JAX array; Cobaya's
        # current_logp does value[0] on anything with __len__, which a 0-d JAX
        # array has (and indexing it raises). Return a plain Python float.
        loglkl = float(theory)

        if not np.isfinite(loglkl):
            # A NaN here (e.g. an extreme-parameter region where the predicted
            # cluster count overflows) must be rejected like any other
            # low-probability point, not propagated as NaN: NaN comparisons in
            # cobaya's Metropolis step are always False, so once a walker lands
            # there it gets permanently stuck proposing-and-never-accepting
            # until cobaya's own "stuck for N attempts" abort kills the whole
            # MPI job (seen in production: one rank hit an overflowing N_tot,
            # log_lik -> nan, and accepted almost nothing before the abort).
            return -np.inf

        return loglkl


class joint_gaussian_prior(Likelihood):
    """Joint multivariate-Gaussian prior on a set of sampled parameters.

    Reads `prior_file` (an .npz holding `mean` (n,) and `cov` (n,n)) and adds
    the log-prior  -0.5 (theta - mean)^T cov^-1 (theta - mean)  over the
    parameters listed in `params_order`, IN THAT ORDER -- which MUST match the
    mean/cov ordering stored in the npz. Generic (data-driven), so one class
    serves multiple priors; instantiate it more than once in the YAML via the
    `class:` key. Mirrors the so_cmb_prior_* / desi_prior_* classes below.

    Typical uses: a multi-parameter weak-lensing mass-calibration prior, or a
    CMB-lensing bias prior, each stored as an npz by the calling analysis.
    """

    prior_file: Optional[str] = None
    params_order: Optional[Sequence] = None

    def initialize(self):

        if self.prior_file is None or self.params_order is None:
            raise ValueError(
                "joint_gaussian_prior requires both 'prior_file' and "
                "'params_order' to be set in the YAML.")

        data = np.load(self.prior_file, allow_pickle=True)
        self.mean = np.asarray(data["mean"], dtype=float)
        cov = np.asarray(data["cov"], dtype=float)
        self._params = list(self.params_order)

        n = len(self._params)
        if self.mean.shape != (n,) or cov.shape != (n, n):
            raise ValueError(
                f"joint_gaussian_prior: params_order has {n} entries but "
                f"'{self.prior_file}' stores mean {self.mean.shape}, "
                f"cov {cov.shape}.")

        # [2026-07-08 audit hardening] Guard against a silently transposed/reordered
        # prior: for the KNOWN parameter sets, the npz's own `names` field must be in
        # the exact order that maps positionally onto params_order (shape alone
        # cannot catch a reorder). Keyed by params_order so it holds whatever the npz
        # is called. Unknown sets: pairing is logged for inspection.
        _known_orders = {
            ("b_wl_m", "s_wl_m", "b_wl_0", "s_wl_0", "b_wl_1", "s_wl_1",
             "b_wl_2", "s_wl_2", "b_wl_3", "s_wl_3"):
                ["alpha", "alpha_sigma", "b0", "s0", "b1", "s1", "b2", "s2", "b3", "s3"],
            ("a_beta_generic", "alpha_beta_generic", "beta_beta_generic"):
                ["a_beta", "alpha_beta", "beta_beta"],
        }
        _base = os.path.basename(str(self.prior_file))
        _key = tuple(self._params)
        if "names" in data:
            _names = [str(x) for x in np.atleast_1d(data["names"])][:n]
            if _key in _known_orders and _names != _known_orders[_key]:
                raise ValueError(
                    f"joint_gaussian_prior: npz 'names' order in '{_base}' is "
                    f"{_names}, expected {_known_orders[_key]} — the positional "
                    "map onto params_order would be WRONG (2026-07-08 hardening).")
            self.log.info("joint_gaussian_prior positional map: %s",
                          list(zip(_names, self._params)))

        self.inv_cov = np.linalg.inv(cov)
        self.log.info(
            f"joint_gaussian_prior on {self._params} from {self.prior_file}")

    def get_requirements(self):

        return {p: None for p in self._params}

    def logp(self, **params_values):

        theta = np.array([self.provider.get_param(p) for p in self._params])
        res = theta - self.mean

        return -0.5 * float(res @ self.inv_cov @ res)


class so_cmb_prior_wcdm(Likelihood): #warning: h as input instead of H0

    def initialize(self):

        fisher = np.array([[ 9.00114848e+02,  3.44938030e+04, -1.01268886e+06, 3.29630918e+05,
          -8.21634769e+12, -1.99241898e+04, -5.18764284e+03],
         [ 3.44938030e+04,  8.69979088e+06, -6.33192100e+07,  1.58665297e+07,
          -2.01779166e+15, -3.59432063e+06, -1.98798554e+05],
         [-1.01268886e+06, -6.33192100e+07,  1.88009030e+09, -3.46188758e+08,
           1.49005171e+16,  4.40025363e+07,  5.83644203e+06],
         [ 3.29630918e+05,  1.58665297e+07, -3.46188758e+08, 1.31768098e+08,
          -3.59709579e+15, -7.04475964e+06, -1.89976587e+06],
         [-8.21634769e+12, -2.01779166e+15,  1.49005171e+16, -3.59709579e+15,
           4.73428704e+23,  8.41055456e+14,  4.73533765e+13],
         [-1.99241898e+04, -3.59432063e+06,  4.40025363e+07, -7.04475964e+06,
           8.41055456e+14,  2.08085978e+06,  1.14829325e+05],
         [-5.18764284e+03, -1.98798554e+05,  5.83644203e+06, -1.89976587e+06,
           4.73533765e+13,  1.14829325e+05,  2.98980050e+04]])

        self.minus_half_invcov = - 0.5*fisher
        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        w0_true = -1.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        self.param_vec_true = np.array([H0_true,tau_reio_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true,w0_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"w0":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        w0 = self.provider.get_param("w0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,Ob0h2,Oc0h2,A_s,n_s,w0])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik



class so_cmb_prior_nulcdm(Likelihood):

    def initialize(self):

        cov_matrix = np.array([[ 1.06502e+00, -5.16621e-03, -8.96376e-04,  2.05383e-05, -6.99822e-04,
                      -2.62669e-11,  1.42877e-03],
                     [-5.16621e-03,  9.36274e-05,  6.38655e-06, -3.43076e-08, -1.17204e-06,
                       3.98447e-13,  2.53291e-06],
                     [-8.96376e-04,  6.38655e-06,  8.26514e-07, -1.36349e-08,  4.32241e-07,
                       3.01123e-14, -9.09089e-07],
                     [ 2.05383e-05, -3.43076e-08, -1.36349e-08,  2.94467e-09, -1.63808e-08,
                      -1.82160e-16, -2.31964e-08],
                     [-6.99822e-04, -1.17204e-06,  4.32241e-07, -1.63808e-08,  8.21531e-07,
                      -4.88633e-16, -1.70722e-06],
                     [-2.62669e-11,  3.98447e-13,  3.01123e-14, -1.82160e-16, -4.88633e-16,
                       1.72965e-21, -2.89565e-15],
                     [ 1.42877e-03,  2.53291e-06, -9.09089e-07, -2.31964e-08, -1.70722e-06,
                      -2.89565e-15,  6.65044e-06]])

        self.minus_half_invcov = - 0.5*np.linalg.inv(cov_matrix)

        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        self.param_vec_true = np.array([H0_true,tau_reio_true,Onu0h2_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"Onu0h2":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        Onu0h2 = self.provider.get_param("Onu0h2")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,Onu0h2,Ob0h2,Oc0h2,A_s,n_s])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik

class so_cmb_prior_nulcdm_mnu(Likelihood):

    def initialize(self):

        cov_matrix = np.array([[ 1.06502e+00, -5.16621e-03, -8.96376e-04,  2.05383e-05, -6.99822e-04,
                      -2.62669e-11,  1.42877e-03],
                     [-5.16621e-03,  9.36274e-05,  6.38655e-06, -3.43076e-08, -1.17204e-06,
                       3.98447e-13,  2.53291e-06],
                     [-8.96376e-04,  6.38655e-06,  8.26514e-07, -1.36349e-08,  4.32241e-07,
                       3.01123e-14, -9.09089e-07],
                     [ 2.05383e-05, -3.43076e-08, -1.36349e-08,  2.94467e-09, -1.63808e-08,
                      -1.82160e-16, -2.31964e-08],
                     [-6.99822e-04, -1.17204e-06,  4.32241e-07, -1.63808e-08,  8.21531e-07,
                      -4.88633e-16, -1.70722e-06],
                     [-2.62669e-11,  3.98447e-13,  3.01123e-14, -1.82160e-16, -4.88633e-16,
                       1.72965e-21, -2.89565e-15],
                     [ 1.42877e-03,  2.53291e-06, -9.09089e-07, -2.31964e-08, -1.70722e-06,
                      -2.89565e-15,  6.65044e-06]])

        Onu02_to_mnu = 93.14

        cov_matrix[2,:] = cov_matrix[2,:]*Onu02_to_mnu
        cov_matrix[:,2] = cov_matrix[:,2]*Onu02_to_mnu

        self.minus_half_invcov = - 0.5*np.linalg.inv(cov_matrix)

        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        mnu_true = Onu0h2_true*Onu02_to_mnu

        self.param_vec_true = np.array([H0_true,tau_reio_true,mnu_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"m_nu":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        m_nu = self.provider.get_param("m_nu")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,m_nu,Ob0h2,Oc0h2,A_s,n_s])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik


class desi_prior_nulcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01, -9.855313873854645863e+03, -4.484472083574374892e+02, 2.365607125643557751e+01],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05, -1.819894418709868478e+04],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04, -2.814617165869603355e+02],
        [2.365607125643557751e+01, -1.819894418709868478e+04 ,-2.814617165869603355e+02 ,5.120817278151554319e+01]])

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        mnu_true = 0.06
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true,mnu_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None,"m_nu":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        mnu = self.provider.get_param("m_nu")

        param_vec = np.array([H0,Ob0h2,Oc0h2,mnu])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik


class desi_prior_wcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01,  -9.855313873854645863e+03, -4.484472083574374892e+02, 3.103177771032207488e+02],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05,-2.312677425005600380e+05],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04,-6.818149725908029723e+03],
        [3.103177771032207488e+02, -2.312677425005600380e+05, -6.818149725908029723e+03,7.563758947320854531e+03]])

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        w0_true = -1.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true,w0_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None,"w0":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        w0 = self.provider.get_param("w0")

        param_vec = np.array([H0,Ob0h2,Oc0h2,w0])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik

class desi_prior_lcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01, -9.855313873854645863e+03, -4.484472083574374892e+02, 2.365607125643557751e+01],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05, -1.819894418709868478e+04],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04, -2.814617165869603355e+02],
        [2.365607125643557751e+01, -1.819894418709868478e+04 ,-2.814617165869603355e+02 ,5.120817278151554319e+01]])

        indices = [0,1,2]

        fisher = fisher[indices,:]
        fisher = fisher[:,indices]

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")

        param_vec = np.array([H0,Ob0h2,Oc0h2])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik
