#!/usr/bin/env bash
set -euo pipefail

# Offline Phase 4A export. This reads Phase 3 query/label caches and does not
# generate queries, load models, train classifiers, or change thresholds.

uv run python scripts/export_phase4a_query_trace.py \
  --cache-root generated/phase3_fixed_hybrid_cache \
  --phase3-scores generated/results/phase3/phase3_fixed_hybrid_cifar10_b20_s123456789.csv \
  --output generated/results/phase4a/phase4a_query_trace_cifar10_b20_s123456789.csv

