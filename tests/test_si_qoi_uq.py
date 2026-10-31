"""bench/si_qoi_uq: observable rows (spec §2: linearity, finite differences, the fitted virial convention) and
the implicit-differentiation rows of the delta method (a0 and the elastic constants against finite
differences in theta)."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase.build import bulk  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from conftest import FIXTURE_DIR  # noqa: E402

from bench.si_qoi_uq.linear import LinearACE, observable_rows  # noqa: E402

MODEL = FIXTURE_DIR / "si_fitted.npz"


@pytest.fixture(scope="module")
def lin():
    return LinearACE(MODEL)


@pytest.fixture(scope="module")
def atoms():
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    at.rattle(0.05, seed=1)
    at.set_cell(at.cell * np.array([1.01, 1.0, 0.99]), scale_atoms=True)
    return at


def _efv(lin, theta, atoms):
    from ace_jax.calc.point import ACECalculator
    calc = ACECalculator(str(MODEL), skin=0.0)
    calc.model = lin.model_at(theta)
    at = atoms.copy()
    at.calc = calc
    return (at.get_potential_energy(), at.get_forces().reshape(-1),
            -at.get_stress(voigt=True) * at.get_volume())                 # virial W = -V sigma (fitted convention)


@pytest.fixture(scope="module")
def rows(lin, atoms):
    return observable_rows(atoms, lin)


def test_linearity(lin, atoms, rows):
    """E, F, W of the model at a theta far from theta_hat equal the rows contracted with theta."""
    bE, BF, BV = rows
    th = lin.theta_hat * (1 + 0.3 * np.random.default_rng(0).standard_normal(lin.p))
    E, F, W = _efv(lin, th, atoms)
    assert abs(E - (bE @ th + lin.e_const(atoms))) < 1e-9 * max(1.0, abs(E))
    np.testing.assert_allclose(BF @ th, F, rtol=0, atol=1e-10 * max(1.0, np.abs(F).max()))
    np.testing.assert_allclose(BV @ th, W, rtol=0, atol=1e-10 * max(1.0, np.abs(W).max()))


def test_finite_difference(lin, atoms, rows):
    """Central differences of the evaluated model along a random theta direction match the rows."""
    bE, BF, BV = rows
    d = lin.theta_hat * 1e-3 * np.random.default_rng(1).standard_normal(lin.p)
    p, m = _efv(lin, lin.theta_hat + d, atoms), _efv(lin, lin.theta_hat - d, atoms)
    for got, row in zip(((p[0] - m[0]) / 2, (p[1] - m[1]) / 2, (p[2] - m[2]) / 2), (bE, BF, BV)):
        np.testing.assert_allclose(got, row @ d, rtol=1e-6, atol=1e-10)


def test_virial_rows_match_fit_rows(lin, atoms, rows):
    """B_V (and b_E, B_F) equal the fit's own design rows (fit.rows.linear_rows) for the same structure:
    same columns, same virial sign and Voigt order, no per-volume factor."""
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    from ace_jax.fit.rows import linear_rows
    at = atoms.copy()
    at.info["energy"] = 0.0
    at.arrays["forces"] = np.zeros((len(at), 3))
    at.info["virial"] = np.zeros((3, 3))
    cfgs = load_configs([at], virial_key="virial")
    ds = build_dataset(cfgs, {"rcut": lin.rc, "elements": lin.z}, lin.E0_pre, 1)
    cfg = GPConfig(r0=2.4, rcut=lin.rc, n_B=lin.n_B, n_pair=lin.n_pair, NZ=lin.NZ, C=int(ds.y_E.shape[1]))
    r, _, _ = linear_rows(lin.model, cfg, jax.tree.map(lambda a: a[0], ds))
    bE, BF, BV = rows
    n = len(at)
    np.testing.assert_allclose(np.asarray(r.E)[0], bE, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r.F)[:n].reshape(-1, lin.p), BF, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r.V)[0], BV, rtol=1e-9, atol=1e-10)


def test_delta_rows_finite_difference(lin):
    """The implicit-differentiation rows of a0 and of both elastic routes match central differences."""
    from bench.si_qoi_uq.delta import Diamond
    th = lin.theta_hat
    d = Diamond(lin, 5.46)
    a0 = d.a0(th)
    _, G = d.elastic_stress(th, rows=True)
    _, Ge = d.elastic_energy(th, rows=True)
    da0 = d.da0(th, a0)
    r = 1e-6 * th * np.random.default_rng(2).standard_normal(lin.p)
    np.testing.assert_allclose((d.a0(th + r) - d.a0(th - r)) / 2, da0 @ r, rtol=1e-3)
    np.testing.assert_allclose((d.elastic_stress(th + r) - d.elastic_stress(th - r)) / 2, G @ r, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose((d.elastic_energy(th + r) - d.elastic_energy(th - r)) / 2, Ge @ r, rtol=1e-3, atol=1e-7)
