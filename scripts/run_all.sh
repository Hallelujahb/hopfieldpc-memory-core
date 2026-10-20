#!/usr/bin/env bash
# Runs the tests and every benchmark and saves the raw output under reports/.
set -uo pipefail
mkdir -p reports
{ date; python --version; python -c "import torch; print('torch', torch.__version__)"; git rev-parse --short HEAD 2>/dev/null; } | tee reports/environment.txt
pytest -q 2>&1 | tee reports/pytest.txt
python -m benchmarks.pc_hopfield_toy 2>&1 | tee reports/pc_hopfield_toy.txt
python -m benchmarks.mnist_prototype_retrieval 2>&1 | tee reports/mnist_prototype_retrieval.txt
python -m benchmarks.mnist_readout_controls 2>&1 | tee reports/mnist_readout_controls.txt
python -m benchmarks.mini_pc_mnist --json-out reports/mini_pc_mnist.json 2>&1 | tee reports/mini_pc_mnist.txt
echo; echo "Saved: $(ls reports | tr '\n' ' ')"
