# hopfieldpc-memory-core

Standalone modern Hopfield memory core for the HopfieldPC project.

<p align="justify">
This repository implements and validates the core Hopfield memory mechanism before integration into FabricPC. It focuses on beta-controlled associative retrieval, tensor-shape validation, diagnostics, synthetic benchmarks, and unit tests.

## Setup

Clone the repository:

```bash
git clone https://github.com/iCog-Labs-Dev/hopfieldpc-memory-core.git
cd hopfieldpc-memory-core
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import hopfieldpc; print('hopfieldpc import OK')"
```

## What is in this repository

| Area | Files | Purpose |
|---|---|---|
| Core retrieval | `src/hopfieldpc/functional.py`, `memory.py`, `api.py` | One-step and multi-step modern Hopfield retrieval, the `HopfieldMemory` module and its shape contract |
| Energy and diagnostics | `energy.py`, `diagnostics.py` | Hopfield energy (`hopfield_full_energy` is the one retrieval descends), entropy, top-k mass, retrieval distance |
| PC coupling | `coupling.py`, `docs/energy_consistent_coupling.md` | Joint energy `E_PC + lam * E_H`, the pull term `z - retrieved(z)`, the step-size rule, a curvature estimator, dot and l2 scores |
| Readout | `readout.py` | Heteroassociative readout (keys matched, values returned) and class prototypes |
| Experiments | `benchmarks/`, `configs/`, `docs/mini_pc_protocol.md` | Synthetic retrieval, beta sweep, MNIST prototype retrieval, coupling toy study, readout baselines, mini PC comparison |

## Run

```bash
pytest -q
python -m benchmarks.pc_hopfield_toy
python -m benchmarks.readout_controls
python -m benchmarks.mini_pc_mnist
bash scripts/run_all.sh        # tests and every experiment, raw logs under reports/logs
```

`QUICK=1 bash scripts/run_all.sh` is a smoke run only. Its numbers should not be reported.

## Status

The standalone memory core, diagnostics and benchmarks are implemented and tested. The coupling with
PC inference, the heteroassociative readout and a standalone PC comparison are implemented, with the
protocol and success criteria for the comparison written down before it was run
(`docs/mini_pc_protocol.md`). FabricPC integration is not done. `hopfieldpc.settle` takes any
differentiable PC energy as a function of the hidden state, which is the intended integration point.
