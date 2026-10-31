"""The fitted linear ACE as an explicit function of its coefficients theta, and the parameter-space
covariance factors the propagation samples from (spec §2).

theta is the fit's column layout (fit/rows.py `_place`, pipeline/export.py `linear_arrays_from_mean`):
species-major B blocks, then pair blocks, then (``--e0 lsq``) one E0 column per species, the atom count on
the energy row.  So E(x, theta) = Phi(x) . theta + sum_i E0_pre[z_i] with Phi the summed site descriptors,
and E0_pre = the model file's E0 minus the fitted shift theta_hat[E0 cols].

Covariances, all as factors L with Sigma = L L^T (rows of the parameter space, columns any width):
  * CR3 (the served `--uq ard` jackknife shape): L = D^-1 R      (ARDPosterior: V = (R^T D^-1 phi)^2)
  * posterior (B1, `--ard-variance kappa`):   L = D^-1 chol^-T  (var_rows: |chol^-1 D^-1 phi|^2)
Virial convention: W = -dE/d(eps) (symmetric strain eps applied to positions and cell), Voigt order xx, yy,
zz, yz, xz, xy, in eV -- the fitted ``virial`` rows (rows.py `_voigt`); ASE's stress is -W / V."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ace_jax.eval.io import load
from ace_jax.eval.model import fold_readout
from ace_jax.eval.nlist import _neighbour_list
from ace_jax.fit.ard import ARDPosterior

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_strain(a, b):
    """The symmetric unit strain whose energy derivative is W_ab (off-diagonal: 1/2 on each side)."""
    e = np.zeros((3, 3))
    e[a, b] += 0.5
    e[b, a] += 0.5
    return e


class LinearACE:
    """model.npz + posterior.npz of one `aj fit --uq ard` run."""

    def __init__(self, model_npz, posterior_npz=None):
        self.model, self.meta, _ = load(model_npz, fold=False)       # unfolded: site_descriptors reads WB
        self.z = [int(v) for v in self.meta["elements"]]
        self.rc = float(self.meta["rcut"])
        self.NZ = len(self.z)
        self.n_B, self.n_pair = int(self.model.WB.shape[0]), int(self.model.Wpair.shape[0])
        self.len_readout = (self.n_B + self.n_pair) * self.NZ
        self.post = None if posterior_npz is None else ARDPosterior.load(posterior_npz)
        if self.post is not None:
            self.theta_hat = np.asarray(self.post.mean, np.float64)
            self.p = len(self.theta_hat)
            if self.p not in (self.len_readout, self.len_readout + self.NZ):
                raise ValueError(f"posterior width {self.p} vs readout {self.len_readout} (+{self.NZ} E0)")
        else:
            self.theta_hat = self._theta_from_model()
            self.p = len(self.theta_hat)
        self.e0_cols = self.p > self.len_readout
        E0_model = np.asarray(self.model.E0, np.float64)
        self.E0_pre = E0_model - (self.theta_hat[self.len_readout:] if self.e0_cols else 0.0)
        if self.post is not None:                                     # the model file must be theta_hat's
            th = self._theta_from_model()
            if not np.allclose(th[:self.len_readout], self.theta_hat[:self.len_readout], rtol=1e-6, atol=1e-9):
                raise ValueError(f"{model_npz} readout is not the posterior mean of {posterior_npz}")

    def _theta_from_model(self):
        m = self.model
        return np.concatenate([np.asarray(m.WB, np.float64).T.reshape(-1), np.asarray(m.Wpair, np.float64).T.reshape(-1)])

    # ---------------------------------------------------------------- theta -> evaluable model
    def model_at(self, theta):
        """An ACEModel (folded, as `load` gives it) with coefficients theta."""
        th = np.asarray(theta, np.float64)
        nB, nP, NZ = self.n_B, self.n_pair, self.NZ
        E0 = self.E0_pre + (th[self.len_readout:self.len_readout + NZ] if self.e0_cols else 0.0)
        m = dataclasses.replace(self.model, WB=jnp.asarray(th[:NZ * nB].reshape(NZ, nB).T),
                                Wpair=jnp.asarray(th[NZ * nB:NZ * (nB + nP)].reshape(NZ, nP).T),
                                E0=jnp.asarray(E0), folded=False)
        return fold_readout(m)

    # ---------------------------------------------------------------- covariance factors
    def factor(self, kind):
        """L (p, r) with Sigma = L L^T: "cr3" (jackknife shape) or "posterior" (needs the kappa fit's chol)."""
        post = self.post
        dinv = np.asarray(post.dinv, np.float64)
        if kind == "cr3":
            S = post.R if post.R is not None else post.Q
            if S is None:
                raise ValueError("posterior has no jackknife shape (R or Q): fit with --ard-variance sandwich")
            return dinv[:, None] * np.asarray(S, np.float64)
        if kind == "posterior":
            if post.chol is None:
                raise ValueError("posterior saved without chol: the B1 shape needs an --ard-variance kappa fit")
            import scipy.linalg
            Linv_T = scipy.linalg.solve_triangular(np.asarray(post.chol, np.float64), np.eye(len(dinv)), lower=True).T
            return dinv[:, None] * Linv_T
        raise ValueError(kind)

    # ---------------------------------------------------------------- geometry -> Phi
    def graph(self, atoms, skin=1.0):
        """Fixed topology (i, j, S, species index) for cutoff + skin: E stays differentiable while atoms move
        less than skin / 2 (edges beyond rc are masked)."""
        i, j, _, S = _neighbour_list(atoms.positions, atoms.cell.array, atoms.pbc, self.rc + skin)
        idx = {zz: k for k, zz in enumerate(self.z)}
        nz = np.array([idx[int(n)] for n in atoms.numbers], np.int32)
        return dict(i=jnp.asarray(i, jnp.int32), j=jnp.asarray(j, jnp.int32), S=jnp.asarray(S, jnp.float64),
                    node_z=jnp.asarray(nz), n=len(atoms))

    def phi_fn(self, g):
        """Phi(pos (N, 3), cell (3, 3)) -> (p,): E = Phi . theta + sum E0_pre."""
        m, rc, NZ = self.model, self.rc, self.NZ
        i, j, S, node_z, n = g["i"], g["j"], g["S"], g["node_z"], g["n"]
        zi, zj = node_z[i], node_z[j]
        counts = jnp.zeros(NZ).at[node_z].add(1.0)
        safe = jnp.array([rc, 0.0, 0.0])

        def phi(pos, cell):
            rij = pos[j] - pos[i] + S @ cell
            mask = jnp.sum(rij * rij, axis=1) < rc * rc
            rij = jnp.where(mask[:, None], rij, safe)                 # masked edges at rc: finite, zero envelope
            X = m.site_descriptors(rij, zi, zj, i, n, node_z, mask)
            out = X.sum(0)
            return jnp.concatenate([out, counts]) if self.e0_cols else out
        return phi

    def e_const(self, atoms):
        idx = {zz: k for k, zz in enumerate(self.z)}
        return float(sum(self.E0_pre[idx[int(n)]] for n in atoms.numbers))

    def energy_fn(self, atoms, skin=1.0):
        """E(pos, cell, theta) on atoms' fixed topology."""
        phi, c = self.phi_fn(self.graph(atoms, skin)), self.e_const(atoms)
        return lambda pos, cell, theta: phi(pos, cell) @ theta + c


def strained(pos, cell, eps):
    T = jnp.eye(3) + eps
    return pos @ T.T, cell @ T.T


def observable_rows(atoms, lin, skin=1.0, chunk=48):
    """d(observable)/d(theta) for one structure (spec §2): b_E (p,), B_F (3N, p) atom-major xyz,
    B_V (6, p) virial W = -dE/d(eps) in eV, Voigt xx yy zz yz xz xy.  E is linear in theta, so these
    are exact: b_E = Phi, B_F = -dPhi/dx (forward mode over positions, `chunk` directions per pass),
    B_V = -dPhi/d(eps)."""
    phi = lin.phi_fn(lin.graph(atoms, skin))
    pos, cell = jnp.asarray(atoms.positions), jnp.asarray(atoms.cell.array)
    bE = np.asarray(phi(pos, cell))
    n3 = 3 * len(atoms)
    jvp_pos = jax.jit(jax.vmap(lambda t: jax.jvp(lambda x: phi(x, cell), (pos,), (t,))[1]))
    BF = np.empty((n3, lin.p))
    eye = np.eye(n3)
    for a in range(0, n3, chunk):
        T = eye[a:a + chunk].reshape(-1, len(atoms), 3)
        BF[a:a + len(T)] = -np.asarray(jvp_pos(jnp.asarray(T)))
    E6 = jnp.asarray(np.stack([voigt_strain(a, b) for a, b in VOIGT]))
    dphi_de = jax.vmap(lambda e: jax.jvp(lambda s: phi(*strained(pos, cell, s)), (jnp.zeros((3, 3)),), (e,))[1])
    BV = -np.asarray(dphi_de(E6))
    return bE, BF, BV
