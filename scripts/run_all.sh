#!/usr/bin/env bash
# Runs the tests and every experiment script, keeping the raw logs under reports/logs.
#   QUICK=1 bash scripts/run_all.sh   smoke run (1 seed, 2 epochs, small test set), do not report these numbers
#   bash scripts/run_all.sh           full protocol from configs/ (slow)
set -uo pipefail
mkdir -p reports/logs
FLAG=""
if [ -n "${QUICK:-}" ]; then
  FLAG="--quick"
  echo "quick mode: smoke test only, do not report these numbers"
fi
{ date; python --version; python -c "import torch; print('torch', torch.__version__)"; git rev-parse --short HEAD 2>/dev/null; } | tee reports/logs/environment.txt
pytest -q 2>&1 | tee reports/logs/pytest.txt
python -m benchmarks.pc_hopfield_toy 2>&1 | tee reports/logs/pc_hopfield_toy.txt
python -m benchmarks.readout_controls $FLAG 2>&1 | tee reports/logs/readout_controls.txt
python -m benchmarks.mini_pc_mnist $FLAG 2>&1 | tee reports/logs/mini_pc_mnist.txt
