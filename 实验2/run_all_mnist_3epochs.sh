#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPTS=(
  "mnist_basic.py"
  "mnist_custom_linear.py"
  "mnist_custom_linear+op.py"
  "mnist_custom_linear_cpp.py"
  "mnist_custom_linear+op_cpp.py"
)

for script in "${SCRIPTS[@]}"; do
  echo "==> Running ${script} (epochs=5)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/${script}" --epochs 14
done

echo "All runs finished."
