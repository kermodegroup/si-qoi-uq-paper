"""Delta-method rows G = dQ/dtheta for the smooth QoIs (spec §4a; RECON decision Q1).

Every harness QoI sits at the model's own relaxed diamond lattice constant a0(theta), so none is linear in
theta.  Two kinds of implicit derivative cover the smooth subset:

* Energies at a minimum (phase energies, defect and surface formation energies).  E(x*(theta), theta) is
  stationary in the relaxed coordinates x, so (envelope theorem) dE*/dtheta = dE/dtheta at x* = Phi(x*),
  plus the response to coordinates the protocol fixes from a0: a cell built as a0 x (reference shape)
  contributes dE/da0 . da0/dtheta, with dE/da0 = -tr(W) / a0 (W = the virial, linear.py convention).
  The relaxed geometries are the harness's own output for the mean model.
* Elastic constants (relaxed-ion) and the rigid decohesion curve, written as JAX functions of theta with
  every relaxation a `jax.lax.custom_root` (Newton on the gradient; the tangent solve is the Hessian solve
  dx*/dtheta = -K^-1 d(grad_x E)/dtheta), so jax.jacrev differentiates the root, never the iterations.

Discrete steps of the protocols (the argmin over the E(V) grid, the max over the decohesion grid) are held at
their index (fixed-index rule); §4c flags QoIs where the committee disagrees.
"""
import ase.units
import jax
import jax.numpy as jnp
import numpy as np
from ase.build import bulk

from .linear import VOIGT, strained, voigt_strain

GPA = ase.units.GPa
NEWTON = 12


def _newton_root(g, x0, theta):
    """custom_root of g(x, theta) = 0 (a gradient) by Newton with the dense Jacobian (a Hessian)."""
    def solve(f, x):
        return jax.lax.fori_loop(0, NEWTON, lambda _, x: x - jnp.linalg.solve(jax.jacfwd(f)(x), f(x)), x)
    tangent = lambda f, y: jnp.linalg.solve(jax.jacfwd(f)(jnp.zeros_like(y)), y)
    return jax.lax.custom_root(lambda x: g(x, theta), x0, solve, tangent)


def diamond8(a):
    return bulk("Si", "diamond", a=a, cubic=True)


class Diamond:
    """The 8-atom conventional diamond cell on a fixed topology: a0(theta), elastic constants (both routes)."""

    def __init__(self, lin, a_ref, skin=1.5):
        at = diamond8(a_ref)
        self.lin, self.a_ref, self.N = lin, a_ref, len(at)
        self.E = lin.energy_fn(at, skin)
        self.frac = jnp.asarray(at.get_scaled_positions())

    def geom(self, a, eps=None, u=None):
        cell = a * jnp.eye(3)
        if eps is not None:
            cell = cell @ (jnp.eye(3) + eps).T
        pos = self.frac @ cell
        if u is not None:                                   # atom 0 fixed: removes the 3 translations
            pos = pos + jnp.concatenate([jnp.zeros((1, 3)), u.reshape(-1, 3)])
        return pos, cell

    def a0(self, theta):
        dE = jax.grad(lambda a, th: self.E(*self.geom(a), th))
        return _newton_root(lambda a, th: jnp.atleast_1d(dE(a[0], th)), jnp.array([self.a_ref]), theta)[0]

    def relaxed(self, a, eps, theta):
        """Internal coordinates relaxed at cell a (I + eps): returns (u*, pos, cell)."""
        gu = jax.grad(lambda u, th: self.E(*self.geom(a, eps, u), th))
        u = _newton_root(gu, jnp.zeros(3 * (self.N - 1)), theta)
        return (u,) + self.geom(a, eps, u)

    def stress(self, pos, cell, theta):
        """ASE stress (1/V) dE/d(eps), Voigt xx yy zz yz xz xy, eV/A^3."""
        V = jnp.abs(jnp.linalg.det(cell))
        dE = jax.grad(lambda e: self.E(*strained(pos, cell, e), theta))(jnp.zeros((3, 3)))
        dE = 0.5 * (dE + dE.T)
        return jnp.stack([dE[a, b] for a, b in VOIGT]) / V

    @staticmethod
    def strain_steps(n=5, delta=1e-2):
        """matscipy generate_strained_configs: delta * (step + 1 - (n + 1) / 2), step = 0..n-1."""
        return delta * (np.arange(n) + 1 - (n + 1) / 2.0)

    @staticmethod
    def voigt_to_eps(v):
        """Voigt engineering strain -> symmetric 3x3 (e4..e6 halved), matscipy's convention."""
        e = jnp.zeros((3, 3))
        e = e.at[0, 0].set(v[0]).at[1, 1].set(v[1]).at[2, 2].set(v[2])
        e = e.at[1, 2].set(v[3] / 2).at[2, 1].set(v[3] / 2).at[0, 2].set(v[4] / 2).at[2, 0].set(v[4] / 2)
        return e.at[0, 1].set(v[5] / 2).at[1, 0].set(v[5] / 2)

    def elastic_stress(self, theta):
        """(C11, C12, C44) in GPa by the harness's route: matscipy fit_elastic_constants(symmetry="cubic"),
        one strain pattern e1 + e4 at the 5 steps, internal positions relaxed, linear fits of stress on
        strain: C11 = d s_xx/d e1, C12 = mean(d s_yy, d s_zz)/d e1, C44 = d s_yz / d e4."""
        a0 = self.a0(theta)
        s = self.strain_steps()
        S = []
        for x in s:
            eps = self.voigt_to_eps(jnp.array([x, 0, 0, x, 0, 0]))
            _, pos, cell = self.relaxed(a0, eps, theta)
            S.append(self.stress(pos, cell, theta))
        S = jnp.stack(S)
        sc = s - s.mean()
        slope = lambda y: jnp.sum(sc * (y - y.mean())) / np.sum(sc * sc)
        c11, c21, c31, c44 = slope(S[:, 0]), slope(S[:, 1]), slope(S[:, 2]), slope(S[:, 3])
        return jnp.stack([c11, 0.5 * (c21 + c31), c44]) / GPA

    def elastic_energy(self, theta):
        """(C11, C12, C44) in GPa from energy curvature at the same strain steps (internal positions
        relaxed): pattern e1 gives C11, e4 gives C44, e1 + e2 gives 2 (C11 + C12) (E/V = c/2 s^2).  The
        harness's single e1 + e4 pattern mixes C11 and C44 in the energy, so the patterns differ."""
        a0 = self.a0(theta)
        s = self.strain_steps()
        V0 = a0 ** 3
        X = np.stack([np.ones_like(s), s, 0.5 * s ** 2], 1)
        P = np.linalg.pinv(X)                               # least-squares quadratic: c = P y
        curv = []
        for pat in ([1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0]):
            Es = []
            for x in s:
                _, pos, cell = self.relaxed(a0, self.voigt_to_eps(x * jnp.array(pat, float)), theta)
                Es.append(self.E(pos, cell, theta))
            curv.append((jnp.asarray(P[2]) @ jnp.stack(Es)) / V0)
        c11, c44, c1122 = curv
        return jnp.stack([c11, 0.5 * c1122 - c11, c44]) / GPA

    def bulk_energy_rows(self, a, theta_hat):
        """Phi of the 8-atom cell at a (the bulk reference of every formation energy)."""
        return np.asarray(self.lin.phi_fn(self.lin.graph(diamond8(a), 1.5))(*self.geom(a)))

    def da0(self, theta_hat):
        return np.asarray(jax.grad(self.a0)(jnp.asarray(theta_hat)))


def dE_da(lin, atoms, theta_hat, skin=1.0):
    """(Phi (p,), dPhi/d(ln a) (p,)) at atoms: d/d(ln a) is the homogeneous scaling of positions and cell, so
    dE/da = theta . dPhi/dln a / a = -tr(W) / a."""
    phi = lin.phi_fn(lin.graph(atoms, skin))
    pos, cell = jnp.asarray(atoms.positions), jnp.asarray(atoms.cell.array)
    f = lambda s: phi(pos * (1 + s), cell * (1 + s))
    P, dP = jax.jvp(f, (jnp.zeros(()),), (jnp.ones(()),))
    return np.asarray(P), np.asarray(dP)
