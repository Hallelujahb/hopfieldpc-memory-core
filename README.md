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
| Core retrieval | `src/hopfieldpc/functional.py`, `memory.py`, `api.py` | One-step and multi-step modern Hopfield retrieval, `HopfieldMemory` module, shape contract |
| Energy and diagnostics | `energy.py`, `diagnostics.py` | Hopfield energy; entropy, top-k mass, retrieval distance |
| PC coupling | `coupling.py`, `docs/energy_consistent_coupling.md` | Joint energy `E_PC + lam * E_H`, pull term `z - retrieved(z)`, step-size rule, curvature estimator |
| Readout | `readout.py` | Heteroassociative readout (keys matched, values returned), class prototypes |
| Mini PC network | `minipc.py`, `docs/mini_pc_experiment.md` | Standalone PC classifier used to test the coupling and readout before FabricPC |
| Benchmarks | `benchmarks/` | Synthetic retrieval, beta sweep, MNIST prototype retrieval, MNIST controls, mini-PC comparison, coupling toy study |

## Run

```bash
pytest -q
python -m benchmarks.pc_hopfield_toy
python -m benchmarks.mnist_readout_controls
python -m benchmarks.mini_pc_mnist
bash scripts/run_all.sh        # everything, outputs saved under reports/
```

## Status

Standalone memory core, diagnostics and benchmarks are implemented and tested.
Energy-consistent coupling, the heteroassociative readout and a standalone PC comparison are
implemented. FabricPC integration is not done yet: `hopfieldpc.settle` takes any differentiable
PC energy as a function of the hidden state, which is the intended integration point.
