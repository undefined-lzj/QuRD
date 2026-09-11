#!/usr/bin/env bash
set -euo pipefail

# Offline analysis only: reads Phase 4A mismatch records and does not generate
# queries, run models, train a classifier, or implement BA-Hybrid.

uv run python scripts/analyze_phase4b_prefix_evidence.py \
  --input generated/results/phase4a/phase4a_query_trace_cifar10_b20_s123456789.csv \
  --output-dir generated/results/phase4b
