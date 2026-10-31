"""Delta-method rows G = dQ/dtheta for the smooth QoIs (spec §4a; RECON decision Q1).

Every harness QoI sits at the model's own relaxed diamond lattice constant a0(theta), so none is linear in
theta.  E(x, theta) = Phi(x) . theta + const (linear.py), and each relaxation x*(theta) solves
grad_x E = 0, so by implicit differentiation

    dx*/dtheta = -K^-1 d(grad_x E)/dtheta = -K^-1 (dPhi/dx)^T,    K = Hessian of E in x at theta_hat,

and dQ/dtheta = dQ/dtheta|_x + dQ/dx . dx*/dtheta (the spec's dQ/dtheta = dQ/dtheta - dQ/dx K^-1 dF/dtheta).
Relaxations are run forward by plain Newton with jitted gradient/Hessian functions that take theta as an
argument (one compile per structure); the derivative is formed from these pieces, never by
differentiating through iterations.

* a0(theta): the 8-atom diamond cell, E(a) stationary: da0/dtheta = -(dPhi/da) / E_aa.
* Elastic constants (relaxed-ion; Diamond): internal coordinates u (atom 0 fixed) relaxed at each strain;
  stress route d sigma/d theta = S + (d sigma/du) du*/dtheta + (d sigma/da) da0/dtheta, where S are the
  stress rows at fixed geometry; energy route by the envelope theorem in u, plus the a0 term.
* Energies at a minimum (phase energies, defect and surface formation energies) from the harness's own
  relaxed geometries for the mean model (harness.py records them): the envelope theorem gives Phi(x*) for
  every relaxed coordinate, plus dE/da0 . da0/dtheta for the cell the protocol builds from a0
  (dE/da0 = theta . dPhi/d(ln a) / a: homogeneous scaling of cell and positions; the positions' share of it
  vanishes at the minimum, so this is the cell response alone).
* Decohesion (rigid openings of the relaxed 5x1x1 bulk): linear in theta at fixed a0, plus the a0 term; the
  harness's cubic spline derivative is linear in the energies (fixed knots, s = 0).

Discrete steps (the E(V) grid argmin, the decohesion max) are held at their index (fixed-index rule); the
phase rows require that argmin to be the V0 point (else the volume grid itself moves with theta: no row).
"""
import ase.io
import ase.units
import jax
import jax.numpy as jnp
import numpy as np
from ase.build import bulk

from .linear import VOIGT, strained, voigt_strain

GPA = ase.units.GPa
EV_A2_TO_J_M2 = 16.021766
E6 = np.stack([voigt_strain(a, b) for a, b in VOIGT])


def diamond8(a):
    return bulk("Si", "diamond", a=a, cubic=True)


def newton(g, H, x, tol=1e-11, it=40):
    for _ in range(it):
        gx = np.asarray(g(x))
        if np.max(np.abs(gx)) < tol:
            break
        x = x - np.linalg.solve(np.asarray(H(x)), gx)
    return x


class Diamond:
    """The 8-atom conventional diamond cell on a fixed topology."""

    def __init__(self, lin, a_ref, skin=1.5):
        at = diamond8(a_ref)
        self.lin, self.N = lin, len(at)
        self.phi = lin.phi_fn(lin.graph(at, skin))
        self.c = lin.e_const(at)
        self.frac = jnp.asarray(at.get_scaled_positions())
        self.a_ref = a_ref
        nu = 3 * (self.N - 1)
        self.nu = nu
        P = lambda a, eps, u: self.phi(*self.geom(a, eps, u))
        E = lambda a, eps, u, th: P(a, eps, u) @ th + self.c

        def Srows(a, eps, u):                       # sigma = Srows @ theta (ASE stress, eV/A^3)
            pos, cell = self.geom(a, eps, u)
            V = jnp.abs(jnp.linalg.det(cell))
            f = lambda e: self.phi(*strained(pos, cell, e))
            return jax.vmap(lambda e: jax.jvp(f, (jnp.zeros((3, 3)),), (e,))[1])(jnp.asarray(E6)) / V
        sig = lambda a, eps, u, th: Srows(a, eps, u) @ th
        self.E = jax.jit(E)
        self.P = jax.jit(P)
        self.gu = jax.jit(jax.grad(E, 2))
        self.Huu = jax.jit(jax.hessian(E, 2))
        self.gua = jax.jit(jax.jacfwd(jax.grad(E, 2), 0))                 # d(grad_u E)/da
        self.ga = jax.jit(jax.grad(E, 0))
        self.gaa = jax.jit(jax.grad(jax.grad(E, 0), 0))
        self.Pu = jax.jit(jax.jacfwd(P, 2))                               # (p, nu)
        self.Pa = jax.jit(jax.jacfwd(P, 0))                               # (p,)
        self.S = jax.jit(Srows)
        self.sig = jax.jit(sig)
        self.sig_u = jax.jit(jax.jacfwd(sig, 2))
        self.sig_a = jax.jit(jax.jacfwd(sig, 0))

    def geom(self, a, eps=None, u=None):
        cell = a * jnp.eye(3)
        if eps is not None:
            cell = cell @ (jnp.eye(3) + eps).T
        pos = self.frac @ cell
        if u is not None:                           # atom 0 fixed: removes the 3 translations
            pos = pos + jnp.concatenate([jnp.zeros((1, 3)), u.reshape(-1, 3)])
        return pos, cell

    # ------------------------------------------------------------ forward
    def a0(self, th):
        """The relaxed lattice constant (cubic, u = 0 by symmetry), Newton on dE/da."""
        z, u0 = jnp.zeros((3, 3)), jnp.zeros(self.nu)
        a = self.a_ref
        for _ in range(40):
            g = float(self.ga(a, z, u0, th))
            if abs(g) < 1e-11:
                break
            a -= g / float(self.gaa(a, z, u0, th))
        return a

    def relax(self, a, eps, th):
        eps = jnp.asarray(eps)
        return newton(lambda u: self.gu(a, eps, u, th), lambda u: self.Huu(a, eps, u, th), np.zeros(self.nu))

    @staticmethod
    def strain_steps(n=5, delta=1e-2):
        """matscipy generate_strained_configs: delta * (step + 1 - (n + 1) / 2), step = 0..n-1."""
        return delta * (np.arange(n) + 1 - (n + 1) / 2.0)

    @staticmethod
    def voigt_to_eps(v):
        """Voigt engineering strain -> symmetric 3x3 (e4..e6 halved), matscipy's convention."""
        v = np.asarray(v, float)
        return np.array([[v[0], v[5] / 2, v[4] / 2], [v[5] / 2, v[1], v[3] / 2], [v[4] / 2, v[3] / 2, v[2]]])

    # ------------------------------------------------------------ elastic constants
    def _slope_weights(self):
        s = self.strain_steps()
        sc = s - s.mean()
        return sc / np.sum(sc * sc)                  # slope = w . y (least-squares line)

    def elastic_stress(self, th, rows=False):
        """(C11, C12, C44) GPa by the harness's route (matscipy fit_elastic_constants, symmetry cubic): one
        pattern e1 + e4 at the 5 steps, internal positions relaxed, least-squares slopes: C11 = d s_xx/d e1,
        C12 = mean(d s_yy, d s_zz)/d e1, C44 = d s_yz/d e4.  rows=True also returns G (3, p) at th."""
        a0 = self.a0(th)
        w = self._slope_weights()
        da0 = self.da0(th, a0) if rows else None
        sig, dsig = [], []
        for x in self.strain_steps():
            eps = jnp.asarray(self.voigt_to_eps([x, 0, 0, x, 0, 0]))
            u = jnp.asarray(self.relax(a0, eps, th))
            sig.append(np.asarray(self.sig(a0, eps, u, th)))
            if rows:
                du = self._du(a0, eps, u, th, da0)
                dsig.append(np.asarray(self.S(a0, eps, u)) + np.asarray(self.sig_u(a0, eps, u, th)) @ du
                            + np.outer(np.asarray(self.sig_a(a0, eps, u, th)), da0))
        sig = np.stack(sig)
        M = np.array([[1, 0, 0, 0, 0, 0], [0, .5, .5, 0, 0, 0], [0, 0, 0, 1, 0, 0]])   # C11, C12, C44 from slopes
        C = M @ (w @ sig) / GPA
        if not rows:
            return C
        G = np.einsum("k,kjp->jp", w, np.stack(dsig))
        return C, M @ G / GPA

    def elastic_energy(self, th, rows=False):
        """(C11, C12, C44) GPa from energy curvature at the same strain steps (internal positions relaxed):
        e1 gives C11, e4 gives C44, e1 + e2 gives 2 (C11 + C12) (E/V0 = c s^2 / 2, quadratic least squares).
        The harness's single e1 + e4 pattern mixes C11 and C44 in the energy, so the patterns differ."""
        a0 = self.a0(th)
        s = self.strain_steps()
        X = np.stack([np.ones_like(s), s, 0.5 * s ** 2], 1)
        wc = np.linalg.pinv(X)[2]
        V0 = a0 ** 3
        da0 = self.da0(th, a0) if rows else None
        curv, dcurv = [], []
        for pat in ([1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0]):
            Es, dE = [], []
            for x in s:
                eps = jnp.asarray(self.voigt_to_eps(x * np.asarray(pat, float)))
                u = jnp.asarray(self.relax(a0, eps, th))
                Es.append(float(self.E(a0, eps, u, th)))
                if rows:                                              # envelope in u, plus the a0 response
                    dE.append(np.asarray(self.P(a0, eps, u)) + float(self.ga(a0, eps, u, th)) * da0)
            c = wc @ np.array(Es)
            curv.append(c / V0)
            if rows:
                dcurv.append(wc @ np.stack(dE) / V0 - 3 * c / (V0 * a0) * da0)
        M = np.array([[1, 0, 0], [-1, 0, .5], [0, 1, 0]])                # (c11, c44, c1122) -> C11, C12, C44
        C = M @ np.array(curv) / GPA
        return C if not rows else (C, M @ np.stack(dcurv) / GPA)

    # ------------------------------------------------------------ implicit pieces
    def da0(self, th, a0=None):
        a0 = self.a0(th) if a0 is None else a0
        z, u0 = jnp.zeros((3, 3)), jnp.zeros(self.nu)
        return -np.asarray(self.Pa(a0, z, u0)) / float(self.gaa(a0, z, u0, th))

    def _du(self, a, eps, u, th, da0):
        """du*/dtheta (nu, p) = -K^-1 [(dPhi/du)^T + d(grad_u E)/da (x) da0/dtheta]."""
        K = np.asarray(self.Huu(a, eps, u, th))
        rhs = np.asarray(self.Pu(a, eps, u)).T + np.outer(np.asarray(self.gua(a, eps, u, th)), da0)
        return -np.linalg.solve(K, rhs)

    def bulk_rows(self, a):
        """Phi of the 8-atom diamond cell at a (u = 0)."""
        return np.asarray(self.P(a, jnp.zeros((3, 3)), jnp.zeros(self.nu)))


def scaling_rows(lin, atoms, skin=1.0):
    """(Phi (p,), dPhi/d(ln a) (p,)) at atoms: d/d(ln a) scales positions and cell together."""
    phi = lin.phi_fn(lin.graph(atoms, skin))
    pos, cell = jnp.asarray(atoms.positions), jnp.asarray(atoms.cell.array)
    f = lambda s: phi(pos * (1 + s), cell * (1 + s))
    P, dP = jax.jit(lambda: jax.jvp(f, (jnp.zeros(()),), (jnp.ones(()),)))()
    return np.asarray(P), np.asarray(dP)


def _frames(run_dir):
    return ase.io.read(run_dir / "siq_relaxed.xyz", ":")


def energy_qoi_rows(lin, run_dir_of, a0, da0, th):
    """{QoI key: (Q, G (p,), note)} for the energy-at-minimum QoIs from the mean model's relaxed frames.
    run_dir_of(test) -> the harness run dir; Q is recomputed from the rows (Phi . theta + const) so it can be
    checked against the harness's own value."""
    import json
    out = {}
    E = lambda P, at: float(P @ th) + lin.e_const(at)

    # phases: E(V) grid argmin (fixed index); the row exists when it is the V0 point of the grid
    def phase(test):
        fr = _frames(run_dir_of(test))
        props = json.load(open(str(run_dir_of(test)) + "-properties.json"))
        EV = np.array(sorted(props[[k for k in props if k.endswith("E_vs_V")][0]]))
        i = int(np.argmin(EV[:, 1]))
        grid = fr[1:]                                     # call 0: the free relaxation; then the 12 E(V) points
        vols = np.array([f.get_volume() / len(f) for f in grid])
        j = int(np.argmin(np.abs(vols - EV[i, 0])))
        v0 = fr[0].get_volume() / len(fr[0])
        at = grid[j]
        P, _ = scaling_rows(lin, at)
        return E(P, at) / len(at), P / len(at), abs(vols[j] - v0) < 1e-6 * v0, EV[i, 1]
    e_d, P_d, ok_d, ref_d = phase("bulk_diamond")
    for ph in ("beta-Sn", "sh", "hex_diamond", "bc8", "st12", "hcp", "fcc", "bcc"):
        e, P, ok, ref = phase(f"bulk_{ph}")
        note = "" if (ok and ok_d) else "argmin not at the V0 point: grid moves with theta (no delta row)"
        out[f"dE_{ph} (meV/at)"] = (1e3 * (e - e_d), 1e3 * (P - P_d) if not note else None, note,
                                   1e3 * (ref - ref_d))

    def defect(fr_bulk, fr_def, n_bulk):
        at_u, at = fr_bulk, fr_def
        Pu, _ = scaling_rows(lin, at_u)
        P, dP = scaling_rows(lin, at)
        rep = n_bulk / len(at_u)
        ratio = len(at) / n_bulk
        Q = E(P, at) - ratio * rep * E(Pu, at_u)
        G = P + (dP @ th) / a0 * da0 - ratio * rep * Pu
        return Q, G
    fr = _frames(run_dir_of("vacancy-energy"))
    Q, G = defect(fr[0], fr[1], 216)
    out["vacancy_energy"] = (Q, G, "", None)
    fr = _frames(run_dir_of("interstitial-energy"))           # calls: unit cell, tetrahedral, hexagonal, dumbbell
    for k, f in (("tetrahedral", fr[1]), ("hexagonal", fr[2]), ("dumbbell", fr[3])):
        Q, G = defect(fr[0], f, 216)
        out[f"{k}_interstitial_energy"] = (Q, G, "hexagonal: one z coordinate constrained" if k == "hexagonal" else "",
                                           None)
    fr = _frames(run_dir_of("surface-energy-100-relaxed"))    # calls: 4-atom bulk cell, relaxed slab
    at_u, slab = fr[0], fr[1]
    Pu, _ = scaling_rows(lin, at_u)
    P, dP = scaling_rows(lin, slab)
    A = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    rep = len(slab) / len(at_u)
    q = 0.5 * (E(P, slab) - rep * E(Pu, at_u)) / A
    G = 0.5 * (P - rep * Pu) / A + (0.5 * (dP @ th) / a0 / A - 2 * q / a0) * da0
    out["surface_100 (J/m2)"] = (q * EV_A2_TO_J_M2, G * EV_A2_TO_J_M2, "", None)
    return out


def decohesion_rows(lin, run_dir, a0_diamond, da0, th, n_steps=35, max_opening=3.5):
    """(Q, G) of decohesion_max_stress: the harness's rigid openings of its relaxed 5x1x1 bulk, energies per
    area cell[1,1] cell[2,2], cubic spline (s = 0) derivative, max |.| over the opening grid (fixed index).
    The relaxed bulk scales with a0 (positions and cell); the opening is in Angstrom."""
    from scipy import interpolate
    bulk_ = _frames(run_dir)[0] * (5, 1, 1)
    o = np.arange(n_steps + 1) / n_steps * max_opening
    phi = lin.phi_fn(lin.graph(bulk_, 1.5 + max_opening))
    pos0, cell0 = jnp.asarray(bulk_.positions), jnp.asarray(bulk_.cell.array)
    a = bulk_.cell.lengths()[1]

    def Pj(op, s):
        cell = cell0 * (1 + s)
        return phi(pos0 * (1 + s), cell.at[0, 0].add(op))
    Pf = jax.jit(jax.vmap(lambda op: jax.jvp(lambda s: Pj(op, s), (jnp.zeros(()),), (jnp.ones(()),))))
    P, dP = (np.asarray(x) for x in Pf(jnp.asarray(o)))
    area = float(cell0[1, 1] * cell0[2, 2])
    e = (P - P[0]) @ th / area
    de = (P - P[0]) / area + (((dP - dP[0]) @ th) / area - 2 * e)[:, None] / a * da0[None, :]
    D = np.stack([interpolate.splev(o, interpolate.splrep(o, np.eye(len(o))[k], s=0), der=1)
                  for k in range(len(o))], 1)                     # sigma = D e
    sig = D @ e
    i = int(np.argmax(np.abs(sig)))
    return float(abs(sig[i])), np.sign(sig[i]) * (D[i] @ de), i
