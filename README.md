# Behavioural splitting for circuits: crossbar experiments

Reproduces the paper's comparison of SPICE (ngspice) and Algorithm 1 on N x N memristive crossbars, N = 16 to 1024,
with three cells: a memristor alone (1R), and a memristor in series with an ideal or a Shockley diode (1D1R). The
outputs are runtime, peak memory, iterations, and a per-step timing profile of Algorithm 1.

## Setup

```
python -m venv venv
venv/Scripts/pip install -r requirements.txt      # venv/bin/pip on Linux and macOS
```

PySpice also needs ngspice's shared library:

- **Linux:** install `libngspice0` (Debian/Ubuntu) or your distribution's equivalent.
- **Windows:**
  1. Extract `ngspice-47_dll_64.7z` from <https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/> into
     `venv/Lib/site-packages/PySpice/Spice/NgSpice/`, giving `Spice64_dll/dll-vs/ngspice.dll`.
  2. Copy `samplerate.dll` and `sndfile.dll` from the `bin` folder of `ngspice-47_64.7z` next to `ngspice.dll`.

  Windows' `tar -xf` extracts `.7z` archives.

Tested with Python 3.14, numpy 2.5, scipy 1.18, matplotlib 3.11, cvxopt 1.3.3, PySpice 1.5 and ngspice-47.

## Run

```
python reproduce.py
```

This runs 10 random circuits per size, cell and solver, each in its own process. Runs are stopped after 1600 s of solve
time. On the paper's machine (6-core Intel Core i5, 31.8 GB RAM) it takes about 10.5 hours, mostly Algorithm 1 on the
1024 x 1024 1D1R arrays. For a quick check, run `python reproduce.py --sizes 16 32 64 --seeds 2 --out quick`.
`results/runs.csv` is written as runs finish, and rerunning resumes from it. Delete `results/` to start over.

Outputs, in `results/`:

| file | contents |
|---|---|
| `runs.csv` | one row per run: status, runtime, peak memory, iterations, SPICE's method, Algorithm 1's step times |
| `summary.csv` | per cell, solver and N: runs, timeouts, median/min/max runtime and peak memory, median iterations |
| `profiling.csv` | Algorithm 1 per cell and N: iterations, ms per iteration, each step's share of an iteration (%), setup time, iterations' share of runtime (%) |
| `runtime_memory.png` | runtime and peak memory against N (the paper's figure) |
| `profiling.png` | Algorithm 1's runtime split by step, iterations and time per iteration against N |

Medians are over converged runs. A timed-out run counts as above the cap in the runtime median, so the median is
missing once half or more of the runs time out.

## Code

| file | contents |
|---|---|
| `crossbar.py` | the circuits: random crossbar, incidence matrix, branch laws, SPICE's KCL test |
| `algorithm1.py` | Algorithm 1: fixed port resistances, CHOLMOD factorization, branch resolvents, step timers |
| `spice.py` | the ngspice netlist and its operating point (KLU, RELTOL 1e-3, ITL1 100) |
| `reproduce.py` | isolated runs, runtime cap, CSV files |
| `plots.py` | the two figures |

Settings, as in the paper:

- **Circuits:** E ~ U(-1, 1.5) V, R_min ~ log-U(1, 10) kΩ and R_max ~ log-U(50, 200) kΩ per cell, 5 Ω wires, 50 Ω
  sources, 1 kΩ loads. Shockley diodes have I_s = 1 nA, n = 1, V_T = 25.85 mV. SPICE writes the ideal diode as
  uramp(V)/1 mΩ.
- **Algorithm 1:** α = 1 on 1R and 0.8 on 1D1R. γ = R on linear branches, R_min on memristors, √(R_min R_max) on diodes.
  It stops when KCL holds to 1e-9 relative (+1e-12 A) and KVL to 1e-9 relative.
- **SPICE:** a run counts as converged only if its operating point passes the same KCL test at RELTOL. SPICE is not run
  at N = 1024, where ngspice's KLU fails.
