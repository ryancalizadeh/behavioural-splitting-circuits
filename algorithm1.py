"""
Algorithm 1 of the paper on a Crossbar, with a per-step profile.

Port resistances are fixed: gamma = R on linear branches, R_min (the memristor's zero-bias resistance) on memristors and
sqrt(R_min R_max) on diodes. A Gamma^-1 A^T is factorized once by CHOLMOD (through cvxopt) in its own AMD ordering, with
one BLAS thread. The iteration stops when the node potentials of v pass the KCL test of crossbar.kcl with tolerance TOL
and the KVL residual max |v - A^T u| is below TOL max |v|.
"""
import ctypes
import time
from pathlib import Path

import cvxopt
import numpy as np
import scipy.sparse as sp
from cvxopt import cholmod
from scipy.special import wrightomega

from crossbar import I_S, V_T, kcl

ALPHA = {"1R": 1.0, "1D1R-ideal": 0.8, "1D1R-Shockley": 0.8}
TOL, MAX_ITERATIONS = 1e-9, 20_000
STEPS = ("resolvent", "solve", "matvec", "check", "update")

# CHOLMOD calls the OpenBLAS bundled with cvxopt; on these matrices one thread is fastest.
for lib in Path(cvxopt.__file__).parent.glob(".libs/*openblas*"):
    ctypes.CDLL(str(lib)).openblas_set_num_threads(1)


def memristor_resolvent(a, gamma, R_min, R_max, v0):
    """The root v of c v + d tanh(v) = a, c = 1 + gamma / R_max, d = gamma (1 / R_min - 1 / R_max), by Newton's method
    from the warm start v0. The left side is concave for v >= 0, so steps clipped to the root's bracket [lo, hi] rise
    monotonically to it."""
    c, d = 1 + gamma / R_max, gamma * (1 / R_min - 1 / R_max)
    sign, a = np.sign(a), np.abs(a)
    lo, hi = np.maximum(a / (c + d), (a - d) / c), a / c
    x = np.clip(np.abs(v0), lo, hi)
    for _ in range(100):
        t = np.tanh(x)
        x_new = np.clip(x - (c * x + d * t - a) / (c + d * (1 - t * t)), lo, hi)
        done = np.all(np.abs(x_new - x) <= 1e-12 * x_new)
        x = x_new
        if done:
            return sign * x
    raise RuntimeError("memristor resolvent did not converge")


def shockley_resolvent(a, gamma):
    """The root x of x + gamma I_S (exp(x / V_T) - 1) = a: the Wright omega function in closed form (the linearization
    for |a| < V_T, where the closed form cancels), then four Newton steps to rounding."""
    s = gamma * I_S
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.where(np.abs(a) < V_T, a / (1 + s / V_T), a + s - V_T * wrightomega((a + s) / V_T + np.log(s / V_T)))
    for _ in range(4):
        x = x - (x + s * np.expm1(x / V_T) - a) / (1 + s / V_T * np.exp(x / V_T))
    return x


def cholesky(K):
    """A function solving K x = rhs by CHOLMOD's Cholesky factorization of the SPD matrix K."""
    low, n = sp.tril(K, format="coo"), K.shape[0]
    Kc = cvxopt.spmatrix(cvxopt.matrix(low.data), cvxopt.matrix(low.row.astype(np.int64)),
                         cvxopt.matrix(low.col.astype(np.int64)), (n, n))
    F = cholmod.symbolic(Kc, uplo="L")
    cholmod.numeric(Kc, F)
    B = cvxopt.matrix(0.0, (n, 1))
    b = np.asarray(B)[:, 0]  # a view onto B, which cholmod.solve overwrites with the solution

    def solve(rhs):
        b[:] = rhs
        cholmod.solve(F, B)
        return b.copy()

    return solve


def solve(cb):
    """Run Algorithm 1 on cb from a = 0. Returns convergence, iterations and seconds spent in setup (assembly and
    factorization), the whole loop, and each of its STEPS: the branch resolvents (lines 2-3), CHOLMOD's triangular
    solves (line 4), the products with A and A^T (lines 4-5), the convergence test, and the relaxation (line 5)."""
    alpha, t0 = ALPHA[cb.cell], time.perf_counter()
    A = cb.incidence()
    AT = A.T.tocsr()
    gamma = cb.R.copy()
    gamma[cb.mem] = cb.R_min
    if cb.diode:
        gamma[cb.dio] = np.sqrt(cb.R_min * cb.R_max)
    weight = 1 / gamma
    solve_K = cholesky(A @ sp.diags(weight) @ AT)
    out = dict(setup_s=time.perf_counter() - t0, **{f"{step}_s": 0.0 for step in STEPS})

    a, w, v_mem = np.zeros(cb.num_branches), np.zeros(cb.num_nodes), np.zeros(cb.n ** 2)
    t_loop = time.perf_counter()
    for k in range(MAX_ITERATIONS):
        t1 = time.perf_counter()
        v = (a + cb.emf) / 2  # resolvent of the linear branches, which are matched (gamma = R)
        v[cb.mem] = v_mem = memristor_resolvent(a[cb.mem], gamma[cb.mem], cb.R_min, cb.R_max, v_mem)
        if cb.diode == "ideal":
            v[cb.dio] = np.minimum(a[cb.dio], 0.0)
        elif cb.diode == "shockley":
            v[cb.dio] = shockley_resolvent(a[cb.dio], gamma[cb.dio])
        b = 2 * v - a
        t2 = time.perf_counter()
        rhs = A @ (b * weight)
        t3 = time.perf_counter()
        phi = solve_K(rhs)
        t4 = time.perf_counter()
        Sb = 2 * (AT @ phi) - b
        t5 = time.perf_counter()
        v_u, kcl_ok = kcl(cb, A, (w + phi) / 2, TOL)  # (w + phi) / 2 are the node potentials of v
        converged = kcl_ok and np.abs(v - v_u).max() <= TOL * np.abs(v).max()
        t6 = time.perf_counter()
        if not converged:
            a = (1 - alpha) * a + alpha * Sb
            w = (1 - alpha) * w + alpha * phi  # the node potentials of a
        for step, dt in zip(STEPS, (t2 - t1, t4 - t3, t3 - t2 + t5 - t4, t6 - t5, time.perf_counter() - t6)):
            out[f"{step}_s"] += dt
        if converged:
            break
    return dict(out, converged=converged, iterations=k + 1, loop_s=time.perf_counter() - t_loop)
