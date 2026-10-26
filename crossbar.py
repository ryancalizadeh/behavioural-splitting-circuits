"""
The N x N crossbar arrays of the experiments.

Word line r carries nodes w[r, c] joined by R_WIRE resistors and is driven at w[r, 0] by a source E_r ~ U(-1, 1.5) V
through R_SOURCE; bit line c carries nodes b[r, c] joined by R_WIRE and ends at b[N-1, c] in R_LOAD to ground. Every
cross point holds a cell:

    1R              a memristor i = v / R_max + (1 / R_min - 1 / R_max) tanh(v) from w[r, c] to b[r, c];
    1D1R-ideal      an ideal diode from w[r, c] to an internal node m[r, c], then the memristor from m[r, c] to b[r, c];
    1D1R-Shockley   the same with a Shockley diode i = I_S (exp(v / V_T) - 1).

R_min ~ log-U(1, 10) kohm and R_max ~ log-U(50, 200) kohm, drawn per cell. Nodes (ground excluded) are numbered word,
bit, then internal, each row-major; branches are numbered word wires, bit wires, memristors, sources, loads, then diodes.
Branch k runs from tail[k] to head[k]: v = A^T u and KCL is A i = 0, with A the reduced incidence matrix.
"""
import numpy as np
import scipy.sparse as sp

CELLS = ("1R", "1D1R-ideal", "1D1R-Shockley")
R_WIRE, R_SOURCE, R_LOAD = 5.0, 50.0, 1e3
I_S, V_T = 1e-9, 0.02585  # Shockley diode (emission coefficient 1)
R_ON = 1e-3  # SPICE cannot represent a short, so its ideal diode is uramp(v) / R_ON


def memristor(v, R_min, R_max):
    return v / R_max + (1 / R_min - 1 / R_max) * np.tanh(v)


class Crossbar:
    def __init__(self, n, seed, cell):
        rng = np.random.default_rng(seed)
        self.n, self.seed, self.cell = n, seed, cell
        self.diode = {"1R": None, "1D1R-ideal": "ideal", "1D1R-Shockley": "shockley"}[cell]
        self.E = rng.uniform(-1.0, 1.5, n)
        self.R_min = np.exp(rng.uniform(np.log(1e3), np.log(1e4), n * n))
        self.R_max = np.exp(rng.uniform(np.log(5e4), np.log(2e5), n * n))

        r, c = np.indices((n, n))
        w, b, m, gnd = r * n + c, n * n + r * n + c, 2 * n * n + r * n + c, np.full(n, -1)
        blocks = [(w[:, :-1], w[:, 1:]), (b[:-1], b[1:]), (m if self.diode else w, b), (w[:, 0], gnd), (b[-1], gnd)]
        blocks += [(w, m)] if self.diode else []
        self.tail, self.head = (np.concatenate([blk[k].ravel() for blk in blocks]) for k in (0, 1))
        self.num_branches, self.num_nodes = self.tail.size, (3 if self.diode else 2) * n * n
        self.mem = slice(2 * n * (n - 1), 2 * n * (n - 1) + n * n)
        sources = slice(self.mem.stop, self.mem.stop + n)
        self.dio = slice(sources.stop + n, None) if self.diode else None
        # resistance and EMF of the linear branches (R is a placeholder on memristors and diodes)
        self.R = np.full(self.num_branches, R_WIRE)
        self.R[sources], self.R[sources.stop:sources.stop + n] = R_SOURCE, R_LOAD
        self.emf = np.zeros(self.num_branches)
        self.emf[sources] = self.E

    def incidence(self):
        k = np.arange(self.num_branches)
        rows, cols = np.concatenate([self.tail, self.head]), np.concatenate([k, k])
        vals = np.repeat([1.0, -1.0], self.num_branches)
        keep = rows >= 0
        return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(self.num_nodes, self.num_branches))

    def current(self, v, spice=False):
        """Branch currents at branch voltages v. An ideal diode's current is set-valued, so both branches of its cell
        get the cell's current memristor(max(v_d + v_m, 0)), exact at a solution; spice=True instead gives the laws
        ngspice solves, with the ideal diode as uramp(v) / R_ON."""
        i = (v - self.emf) / self.R
        i[self.mem] = memristor(v[self.mem], self.R_min, self.R_max)
        if self.diode == "ideal" and spice:
            i[self.dio] = np.maximum(v[self.dio], 0.0) / R_ON
        elif self.diode == "ideal":
            i[self.mem] = i[self.dio] = memristor(np.maximum(v[self.mem] + v[self.dio], 0.0), self.R_min, self.R_max)
        elif self.diode == "shockley":
            i[self.dio] = I_S * np.expm1(v[self.dio] / V_T)
        return i


def kcl(cb, A, u, reltol, spice=False, abstol=1e-12):
    """Branch voltages of node potentials u, and whether their currents satisfy KCL at every node to SPICE's test:
    |sum of currents| <= reltol * sum of |currents| + abstol. Both solvers' answers must pass it."""
    v = A.T @ u
    i = cb.current(v, spice)
    residual, scale = np.abs(A @ i), abs(A) @ np.abs(i)
    return v, bool(np.all(np.isfinite(residual)) and np.all(residual <= reltol * scale + abstol))
