#!/usr/bin/env bash
set -euo pipefail

# First Fixed-Hybrid round only: SACBenchmark/CIFAR10, budget=20,
# seed=123456789, AKH budgets 0/5/10/15/20.

output="generated/results/phase3_fixed_hybrid_cifar10_b20_s123456789.csv"
summary="generated/results/phase3_fixed_hybrid_cifar10_b20_s123456789_summary.csv"
cache_dir="generated/phase3_fixed_hybrid_cache"
log="generated/results/phase3_fixed_hybrid_cifar10_b20_s123456789.log"

mkdir -p generated/results "${cache_dir}" generated/huggingface
export HF_HOME="$PWD/generated/huggingface"

uv run python scripts/preflight_fixed_hybrid.py

/usr/bin/time -p uv run python main.py bench \
  --generated-dir "${cache_dir}" \
  --device cpu \
  --batch-size 16 \
  --seed 123456789 \
  fixed-hybrid SACBenchmark \
  --budget 20 \
  --baseline-cache-dir generated \
  --output "${output}" 2>&1 | tee "${log}"

uv run python scripts/summarize_fixed_hybrid.py

echo "Scores: ${output}"
echo "Summary: ${summary}"

