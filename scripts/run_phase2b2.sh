#!/usr/bin/env bash
set -euo pipefail

# Phase 2B-2: SACBenchmark/CIFAR10, seed=123456789, budgets 20 and 50.
# Run this script from the QuRD repository root.

output_dir="generated/results/phase2b2_sacbench_cifar10_s123456789"
log_dir="${output_dir}/logs"

mkdir -p "${output_dir}" "${log_dir}" generated/huggingface
export HF_HOME="$PWD/generated/huggingface"

run_one() {
  local method="$1"
  local method_slug="$2"
  local budget="$3"
  local output_file="${output_dir}/phase2b2_sacbench_cifar10_${method_slug}_b${budget}_s123456789.csv"
  local log_file="${log_dir}/${method_slug}_b${budget}.log"

  echo "Running ${method}, budget=${budget}"
  /usr/bin/time -p uv run python main.py bench \
    --device cpu \
    --batch-size 16 \
    --seed 123456789 \
    scores SACBenchmark "${method}" \
    --budget "${budget}" \
    --output "${output_file}" 2>&1 | tee "${log_file}"
}

# Complete and inspect all budget=20 runs before the more expensive budget=50 runs.
run_one AKH akh 20
run_one Random random 20
run_one IPGuard ipguard 20

run_one AKH akh 50
run_one Random random 50
run_one IPGuard ipguard 50

uv run python scripts/validate_phase2b2.py

