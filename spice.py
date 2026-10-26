"""
DC operating point (.op) of a Crossbar from ngspice through PySpice's shared-library interface, with KLU, RELTOL 1e-3
and ITL1 100. Memristors and diodes are behavioural current sources: the ideal diode as uramp(v) / R_ON, the Shockley
diode as I_S (exp(v / V_T) - 1). ngspice falls back to gmin stepping, source stepping and a transient ramp when Newton
fails, so a run counts as converged only if its operating point also passes crossbar.kcl at RELTOL.
"""
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError, NgSpiceShared

from crossbar import I_S, R_LOAD, R_ON, R_SOURCE, R_WIRE, V_T, kcl

RELTOL = 1e-3
# how ngspice reached the operating point, from its notes on stderr (plain Newton prints none)
METHODS = [("operating point could not be simulated", "failed"), ("transient op finished successfully", "transient op"),
           ("source stepping completed", "source stepping"), ("gmin stepping completed", "gmin stepping")]
logging.getLogger("PySpice").setLevel(logging.CRITICAL)  # PySpice logs ngspice's routine notes as errors
_ngspice = None


def ngspice():
    """The process's ngspice instance (the shared library holds one simulator per process)."""
    global _ngspice
    if _ngspice is None:
        dll_dir = Path(NgSpiceShared.LIBRARY_PATH).parent  # cffi does not search the DLL's folder for its dependencies
        if hasattr(os, "add_dll_directory") and dll_dir.is_absolute() and dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
        _ngspice = NgSpiceShared.new_instance()
    return _ngspice


def netlist(cb):
    n, mem = cb.n, "m" if cb.diode else "w"
    g_lin, g_tanh = (1 / cb.R_max).reshape(n, n), (1 / cb.R_min - 1 / cb.R_max).reshape(n, n)
    cells = [(r, c) for r in range(n) for c in range(n)]
    lines = [f".title crossbar {n}x{n} seed={cb.seed}"]
    lines += [f"V{r} s{r} 0 {float(cb.E[r])!r}" for r in range(n)]
    lines += [f"RS{r} s{r} w{r}_0 {R_SOURCE!r}" for r in range(n)]
    lines += [f"RW{r}_{c} w{r}_{c} w{r}_{c + 1} {R_WIRE!r}" for r in range(n) for c in range(n - 1)]
    lines += [f"RB{r}_{c} b{r}_{c} b{r + 1}_{c} {R_WIRE!r}" for r in range(n - 1) for c in range(n)]
    lines += [f"B{r}_{c} {mem}{r}_{c} b{r}_{c} I={float(g_lin[r, c])!r}*{x}+{float(g_tanh[r, c])!r}*tanh({x})"
              for r, c in cells for x in [f"V({mem}{r}_{c},b{r}_{c})"]]
    if cb.diode == "ideal":
        lines += [f"BD{r}_{c} w{r}_{c} m{r}_{c} I={1 / R_ON!r}*uramp(V(w{r}_{c},m{r}_{c}))" for r, c in cells]
    elif cb.diode == "shockley":
        lines += [f"BD{r}_{c} w{r}_{c} m{r}_{c} I={I_S!r}*(exp(V(w{r}_{c},m{r}_{c})/{V_T!r})-1)" for r, c in cells]
    lines += [f"RL{c} b{n - 1}_{c} 0 {R_LOAD!r}" for c in range(n)]
    return "\n".join(lines + [f".options reltol={RELTOL!r} itl1=100 klu", ".op", ".end"])


def node_potentials(ng, plot, cb):
    """Node potentials of an operating-point plot in Crossbar's node order, read from a binary rawfile (far faster
    than one API call per vector)."""
    path = Path(tempfile.gettempdir()) / f"crossbar_op_{os.getpid()}.raw"
    for command in (f"setplot {plot}", "set filetype=binary", f"write '{path.as_posix()}'"):
        ng.exec_command(command)
    data = path.read_bytes()
    path.unlink()
    k = data.index(b"Binary:")
    header = data[:k].decode().splitlines()
    count = int(next(line for line in header if line.startswith("No. Variables:")).split(":")[1])
    start = header.index("Variables:") + 1
    names = [line.split()[1].removeprefix("v(").removesuffix(")") for line in header[start:start + count]]
    values = dict(zip(names, np.frombuffer(data, np.float64, count, data.index(b"\n", k) + 1)))
    n = cb.n
    nodes = [f"{x}{r}_{c}" for x in ("wbm" if cb.diode else "wb") for r in range(n) for c in range(n)]
    return np.array([values[name] for name in nodes])


def solve(cb):
    """Solve cb with ngspice. Returns convergence, ngspice's total Newton iterations (fallbacks included) and the
    method that reached the operating point."""
    ng, before = ngspice(), set(ngspice().plot_names)
    try:
        ng.load_circuit(netlist(cb))
        try:
            ng.exec_command("run")
        except NgSpiceCommandError:
            pass  # raised for any note on stderr, such as gmin stepping; the outcome is judged below
        notes = ng.stderr.lower()
        method = next((method for marker, method in METHODS if marker in notes), "newton")
        try:
            stats = ng.exec_command("rusage everything")
        except NgSpiceCommandError:  # after a failed analysis rusage prints its statistics and an error
            stats = ng.stdout
        iterations = int(float(re.search(r"Total iterations\s*=\s*(\S+)", stats).group(1)))
        plots = [name for name in ng.plot_names if name.startswith("op") and name not in before]
        if method == "failed" or not plots:
            return dict(converged=False, iterations=iterations, method=method)
        u = node_potentials(ng, plots[0], cb)
        _, ok = kcl(cb, cb.incidence(), u, RELTOL, spice=True)
        return dict(converged=ok, iterations=iterations, method=method)
    finally:
        for command in ("destroy all", "remcirc"):
            try:
                ng.exec_command(command)
            except NgSpiceCommandError:
                pass
