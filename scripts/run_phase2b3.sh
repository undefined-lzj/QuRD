#!/usr/bin/env bash
set -euo pipefail

# Phase 2B-3 random-seed validation.
# Run from the QuRD repository root. No seed is selected from observed results.

results_root="generated/results/phase2b3_sacbench_cifar10"
mkdir -p "${results_root}" generated/huggingface
export HF_HOME="$PWD/generated/huggingface"

run_one() {
  local seed="$1"
  local method="$2"
  local method_slug="$3"
  local budget="$4"
  local seed_dir="${results_root}/seed_${seed}"
  local cache_dir="generated/phase2b3_cache/seed_${seed}"
  local output_file="${seed_dir}/phase2b3_sacbench_cifar10_${method_slug}_b${budget}_s${seed}.csv"
  local log_file="${seed_dir}/logs/${method_slug}_b${budget}.log"

  mkdir -p "${seed_dir}/logs" "${cache_dir}"
  echo "Running seed=${seed}, method=${method}, budget=${budget}"
  /usr/bin/time -p uv run python main.py bench \
    --generated-dir "${cache_dir}" \
    --device cpu \
    --batch-size 16 \
    --seed "${seed}" \
    scores SACBenchmark "${method}" \
    --budget "${budget}" \
    --output "${output_file}" 2>&1 | tee "${log_file}"
}

run_seed() {
  local seed="$1"

  # Validate each budget before moving to the next, with IPGuard last.
  run_one "${seed}" AKH akh 10
  run_one "${seed}" Random random 10
  run_one "${seed}" IPGuard ipguard 10

  run_one "${seed}" AKH akh 20
  run_one "${seed}" Random random 20
  run_one "${seed}" IPGuard ipguard 20

  run_one "${seed}" AKH akh 50
  run_one "${seed}" Random random 50
  run_one "${seed}" IPGuard ipguard 50

  uv run python scripts/validate_phase2b3.py --seed "${seed}"
}

case "${1:-all}" in
  42)
    run_seed 42
    ;;
  2026)
    # Do not start the second seed unless the first seed's nine CSVs pass.
    uv run python scripts/validate_phase2b3.py --seed 42
    run_seed 2026
    uv run python scripts/validate_phase2b3.py --seed 42 --seed 2026
    ;;
  all)
    run_seed 42
    run_seed 2026
    uv run python scripts/validate_phase2b3.py --seed 42 --seed 2026
    ;;
  *)
    echo "Usage: $0 [42|2026|all]" >&2
    exit 2
    ;;
esac

