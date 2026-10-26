"""
Runs SPICE and Algorithm 1 on the 1R, ideal 1D1R and Shockley 1D1R crossbars, writes the data as CSV and plots it.

    python reproduce.py [--sizes 16 32 ...] [--seeds 10] [--cap 1600] [--out results]

Each run (cell, solver, N, seed) executes in a fresh process: an untimed warm-up solve of a 2 x 2 array, then the timed
solve of the N x N array of that seed (the circuit is generated before the clock starts). Memory is the process's peak
physical memory. A run still solving --cap seconds into its timed solve is killed and recorded as a timeout; a timeout or
crash ends that solver's sweep of that cell (its remaining seeds and larger sizes). SPICE is not run above N = 512,
where ngspice's KLU fails. Runs are appended to OUT/runs.csv as they finish, and rerunning resumes from it.
"""
import argparse
import csv
import math
import multiprocessing
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import algorithm1
import plots
import spice
from crossbar import CELLS, Crossbar

SOLVERS = {"SPICE": spice.solve, "Algorithm 1": algorithm1.solve}
SPICE_MAX_N = 512
FIELDS = ["cell", "solver", "N", "seed", "status", "runtime_s", "memory_gb", "iterations", "method", "setup_s",
          "loop_s"] + [f"{step}_s" for step in algorithm1.STEPS] + ["machine"]
NUMERIC = [f for f in FIELDS if f.endswith("_s") or f in ("memory_gb", "iterations")]


def peak_memory_gb():
    """Peak physical memory of this process: the peak working set on Windows, the maximum resident set elsewhere."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD), ("peak", ctypes.c_size_t),
                        ("rest", ctypes.c_size_t * 7)]

        kernel32, counters = ctypes.WinDLL("kernel32"), Counters(cb=ctypes.sizeof(Counters))
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.peak / 1e9
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


def child(conn, solver, cell, n, seed):
    SOLVERS[solver](Crossbar(2, 0, cell))  # warm-up: libraries load and initialize outside the timed solve
    cb = Crossbar(n, seed, cell)
    conn.send("started")
    t = time.perf_counter()
    try:
        out = SOLVERS[solver](cb)
        out["status"] = "converged" if out.pop("converged") else "not converged"
    except Exception as e:
        out = dict(status=f"error: {e!r}")
    conn.send(dict(out, runtime_s=time.perf_counter() - t, memory_gb=peak_memory_gb()))


def run(solver, cell, n, seed, cap):
    receiver, sender = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(target=child, args=(sender, solver, cell, n, seed), daemon=True)
    proc.start()
    sender.close()  # the child holds the only write end, so its death shows up as EOFError
    try:
        receiver.recv()  # "started"
        out = receiver.recv() if receiver.poll(cap) else dict(status="timeout")
    except EOFError:
        out = dict(status="crashed")
    if out["status"] == "timeout":
        proc.kill()
    proc.join()
    return out


def sweep(sizes, seeds, cap, path):
    done = {}
    if path.exists():
        with open(path, newline="") as f:
            done = {(r["cell"], r["solver"], int(r["N"]), int(r["seed"])): r for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    ended = {}  # (cell, solver) -> the N at which a timeout or crash ended its sweep
    for cell in CELLS:
        for n in sizes:
            for seed in range(seeds):
                for solver in SOLVERS:
                    key = (cell, solver, n, seed)
                    if (solver == "SPICE" and n > SPICE_MAX_N) or ended.get((cell, solver), math.inf) <= n:
                        continue
                    if key in done:
                        rec = done[key]
                    else:
                        rec = dict(run(solver, cell, n, seed, cap), cell=cell, solver=solver, N=n, seed=seed,
                                   machine=platform.node())
                        with open(path, "a", newline="") as f:
                            csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(rec)
                        runtime = f"{rec['runtime_s']:.3f} s" if "runtime_s" in rec else ""
                        print(f"{cell:<14} {solver:<12} {n:>4} seed {seed}: {rec['status']} {runtime}", flush=True)
                    if rec["status"] in ("timeout", "crashed"):
                        ended[(cell, solver)] = n


def median(xs):
    return statistics.median(xs) if xs else math.nan


def censored_median(xs, timeouts):
    """The median with timed-out runs counted as above the cap: nan once half or more of the runs timed out."""
    m = median(xs + [math.inf] * timeouts)
    return math.nan if math.isinf(m) else m


def summarize(path, out):
    """summary.csv: per cell, solver and N, the runs attempted, converged and timed out, the median runtime with
    timeouts counted as above the cap (nan once half or more timed out), and the runtime's and peak memory's median, min
    and max over converged runs. profiling.csv: Algorithm 1's medians over converged runs of the iterations, the time
    per iteration, each step's share of the loop, the setup time, and the loop's share of setup plus loop."""
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            groups[(r["cell"], r["solver"], int(r["N"]))].append(r)
    summary, profiling = [], []
    for (cell, solver, n), rs in sorted(groups.items(), key=lambda kv: (CELLS.index(kv[0][0]), kv[0][1], kv[0][2])):
        ok = [{k: float(r[k]) for k in NUMERIC if r[k]} for r in rs if r["status"] == "converged"]
        timeouts = sum(r["status"] == "timeout" for r in rs)
        col = lambda key: [r[key] for r in ok]
        summary.append(dict(cell=cell, solver=solver, N=n, runs=len(rs), converged=len(ok), timeouts=timeouts,
                            runtime_s=censored_median(col("runtime_s"), timeouts),
                            runtime_min_s=min(col("runtime_s"), default=math.nan),
                            runtime_max_s=max(col("runtime_s"), default=math.nan),
                            memory_gb=median(col("memory_gb")), memory_min_gb=min(col("memory_gb"), default=math.nan),
                            memory_max_gb=max(col("memory_gb"), default=math.nan), iterations=median(col("iterations"))))
        if solver == "Algorithm 1" and ok:
            profiling.append(dict(
                cell=cell, N=n, runs=len(ok), iterations=median(col("iterations")),
                ms_per_iteration=median([1e3 * r["loop_s"] / r["iterations"] for r in ok]),
                **{f"{step}_pct": median([100 * r[f"{step}_s"] / r["loop_s"] for r in ok]) for step in algorithm1.STEPS},
                setup_s=median(col("setup_s")),
                loop_pct=median([100 * r["loop_s"] / (r["loop_s"] + r["setup_s"]) for r in ok])))
    for name, rows in (("summary.csv", summary), ("profiling.csv", profiling)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    return summary, profiling


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--seeds", type=int, default=10, help="random circuits per size (seeds 0..SEEDS-1)")
    ap.add_argument("--cap", type=float, default=1600.0, help="seconds of solve time before a run is killed")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sweep(sorted(args.sizes), args.seeds, args.cap, args.out / "runs.csv")
    summary, profiling = summarize(args.out / "runs.csv", args.out)
    plots.runtime_memory(summary, args.out / "runtime_memory.png")
    plots.profiling(profiling, args.out / "profiling.png")
    print(f"results in {args.out}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
