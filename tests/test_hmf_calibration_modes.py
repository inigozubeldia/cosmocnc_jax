"""Unit tests of the optional multiplicative perturbation of the Tinker08 mass function by modes
(cnc_params["hmf_calibration_modes_file"]): the file loader, the factor ln f = Sum_k a_k mode_k(log10(1/sigma), z), the
agreement of the cosmocnc_jax and cosmocnc implementations, and the cobaya theory classes' check of the amplitude set.
No likelihood is built. Run: python tests/test_hmf_calibration_modes.py (exit code 0 = all passed).
"""
import os
import sys
import tempfile
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cosmocnc_jax import hmf as hmf_jax
import cosmocnc.hmf as hmf_np

FAIL = []


def check(cond, label):
    print(("  ok    " if cond else "  FAIL  ") + label, flush=True)
    if not cond:
        FAIL.append(label)


def write(path, **arrays):
    np.savez(path, **arrays)
    return path


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    x = np.linspace(-0.3, 0.5, 17); z = np.linspace(0.02, 1.0, 9); modes = rng.normal(0., 0.02, (17, 9, 6))
    good = write(os.path.join(tmp, "good.npz"), x_grid=x, z_grid=z, modes=modes)

    print("[1] loader")
    for name, mod in (("cosmocnc_jax", hmf_jax), ("cosmocnc", hmf_np)):
        xx, zz, mm = mod.load_hmf_calibration_modes(good)
        check(np.array_equal(xx, x) and np.array_equal(zz, z) and np.array_equal(mm, modes), f"{name}: valid file read back exactly")
        bad = {
            "missing modes": dict(x_grid=x, z_grid=z),
            "x not increasing": dict(x_grid=x[::-1], z_grid=z, modes=modes),
            "z repeated value": dict(x_grid=x, z_grid=np.r_[z[:3], z[2], z[4:]], modes=modes),
            "modes wrong shape": dict(x_grid=x, z_grid=z, modes=modes[:, :5]),
            "2-D modes": dict(x_grid=x, z_grid=z, modes=modes[:, :, 0]),
            "too many modes": dict(x_grid=x, z_grid=z, modes=rng.normal(size=(17, 9, mod.HMF_MODES_MAX + 1))),
            "non-finite": dict(x_grid=x, z_grid=z, modes=np.where(modes > 0.03, np.nan, modes)),
            "grid of length 1": dict(x_grid=x[:1], z_grid=z, modes=modes[:1]),
        }
        for lab, arr in bad.items():
            p = write(os.path.join(tmp, f"bad_{lab.replace(' ', '_')}.npz"), **arr)
            check(raises(mod.load_hmf_calibration_modes, p), f"{name}: refuses a file with {lab}")

    print("[2] the factor (cosmocnc_jax), on a model-like (n_z, n_M) grid")
    zv = np.linspace(0.0, 1.2, 13)                                     # includes z outside the table on both sides
    sig = np.exp(rng.uniform(np.log(0.2), np.log(3.0), (13, 50))); sig.sort(axis=1); sig = sig[:, ::-1]
    f = lambda amps, m=modes: np.asarray(hmf_jax.hmf_calibration_ln_factor(jnp.asarray(sig), jnp.asarray(zv), jnp.asarray(x), jnp.asarray(z), jnp.asarray(m), jnp.asarray(amps, dtype=float)))
    check(np.array_equal(f(np.zeros(6)), np.zeros((13, 50))), "zero amplitudes give exactly 0")
    const = np.full((17, 9, 1), 0.07)
    v = f([2.5], const); check(np.allclose(v, 0.175, rtol=0, atol=1e-15), "constant mode c = 0.07, amplitude 2.5: 0.175 everywhere (incl. outside the table)")
    a1, a2 = rng.normal(size=6), rng.normal(size=6)
    check(np.allclose(f(a1 + 2. * a2), f(a1) + 2. * f(a2), rtol=0, atol=1e-14), "linear in the amplitudes")
    # nodes: sigma chosen so that x falls exactly on the x nodes, at a z node
    sig_n = 10.**(-x)[None, :].repeat(2, 0); zn = z[[2, 5]]
    vn = np.asarray(hmf_jax.hmf_calibration_ln_factor(jnp.asarray(sig_n), jnp.asarray(zn), jnp.asarray(x), jnp.asarray(z), jnp.asarray(modes), jnp.asarray(a1)))
    ref = np.tensordot(modes, a1, axes=([2], [0]))[:, [2, 5]].T
    check(np.allclose(vn, ref, rtol=0, atol=1e-14), "reproduces the table at its nodes")
    # outside the table: clamped to the edge values
    sig_o = np.array([[10.**(-(x[0] - 0.2)), 10.**(-(x[-1] + 0.2))]]); zo = np.array([z[-1] + 0.5])
    vo = np.asarray(hmf_jax.hmf_calibration_ln_factor(jnp.asarray(sig_o), jnp.asarray(zo), jnp.asarray(x), jnp.asarray(z), jnp.asarray(modes), jnp.asarray(a1)))
    F = np.tensordot(modes, a1, axes=([2], [0]))
    check(np.allclose(vo[0], [F[0, -1], F[-1, -1]], rtol=0, atol=1e-14), "outside the table: the edge values (x below / above, z above)")
    # midpoint linear interpolation in x at a z node
    xm = 0.5 * (x[3] + x[4]); vm = np.asarray(hmf_jax.hmf_calibration_ln_factor(jnp.asarray([[10.**(-xm)]]), jnp.asarray([z[4]]), jnp.asarray(x), jnp.asarray(z), jnp.asarray(modes), jnp.asarray(a1)))
    check(np.isclose(vm[0, 0], 0.5 * (F[3, 4] + F[4, 4]), rtol=0, atol=1e-14), "linear in x between nodes")

    print("[3] cosmocnc_jax against cosmocnc (random tables, sigma, redshifts)")
    worst = 0.
    for trial in range(20):
        mo = rng.normal(0., 0.05, (17, 9, 1 + trial % 10)); am = rng.normal(size=mo.shape[2])
        zz = rng.uniform(-0.1, 1.3, 7); ss = np.exp(rng.uniform(np.log(0.1), np.log(5.), (7, 40)))
        vj = np.asarray(hmf_jax.hmf_calibration_ln_factor(jnp.asarray(ss), jnp.asarray(zz), jnp.asarray(x), jnp.asarray(z), jnp.asarray(mo), jnp.asarray(am)))
        vn_ = np.array([hmf_np.hmf_calibration_ln_factor(ss[i], zz[i], x, z, mo, am) for i in range(len(zz))])
        worst = max(worst, float(np.max(np.abs(vj - vn_))))
    check(worst < 1e-12, f"largest |JAX - NumPy| over 20 random cases: {worst:.2e} (< 1e-12)")

    print("[4] the cobaya theory classes' check of the amplitude set")
    import importlib
    for name in ("cosmocnc_jax.cnc_cobaya_theory", "cosmocnc.cnc_cobaya_theory"):
        try:
            th = importlib.import_module(name)
        except Exception as e:                                         # cobaya missing: report, not a pass
            check(False, f"{name}: import failed ({e})"); continue
        c = th.check_hmf_mode_param_set
        ok = lambda n, p: not raises(c, n, p)
        check(ok(0, ["A_s", "H0"]), f"{name}: no file, no amplitudes: accepted")
        check(ok(3, ["hmf_mode_amp_1", "hmf_mode_amp_2", "hmf_mode_amp_3", "H0"]), f"{name}: 3 modes, 3 amplitudes: accepted")
        check(not ok(0, ["hmf_mode_amp_1"]), f"{name}: amplitude without a file: refused")
        check(not ok(3, ["hmf_mode_amp_1", "hmf_mode_amp_2"]), f"{name}: 3 modes, amplitude 3 missing: refused")
        check(not ok(2, ["hmf_mode_amp_1", "hmf_mode_amp_2", "hmf_mode_amp_3"]), f"{name}: 2 modes, amplitude 3 extra: refused")
        check(all(f"hmf_mode_amp_{k}" in th._SR_INPUT_PARAMS for k in range(1, 21)), f"{name}: amplitudes 1-20 advertised to cobaya")
    print(f"\n{'ALL PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
