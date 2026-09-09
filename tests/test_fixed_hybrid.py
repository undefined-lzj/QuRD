import csv
import tempfile
import unittest
from pathlib import Path

import torch

from qurd.fixed_hybrid import (
    FIXED_HYBRID_COLUMNS,
    FixedHybridScoresCsv,
    fixed_hybrid_cache_dir,
    make_fixed_hybrid_queries,
    score_fixed_hybrid_labels,
)
from scripts.summarize_fixed_hybrid import auc_with_ties, operating_point


class FixedHybridTest(unittest.TestCase):
    def test_queries_use_fixed_component_prefixes(self):
        akh = torch.arange(20).reshape(20, 1)
        ipguard = torch.arange(100, 120).reshape(20, 1)

        queries = make_fixed_hybrid_queries(akh, ipguard, 5, 15)

        self.assertEqual(len(queries), 20)
        self.assertEqual(queries[:5].flatten().tolist(), list(range(5)))
        self.assertEqual(queries[5:].flatten().tolist(), list(range(100, 115)))

    def test_component_and_hybrid_scores(self):
        source = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 1, 0, 1])

        hybrid, akh, ipguard = score_fixed_hybrid_labels(source, target, 2, 2)

        self.assertEqual(hybrid, 0.5)
        self.assertEqual(akh, 0.5)
        self.assertEqual(ipguard, 0.5)
        hybrid, akh, ipguard = score_fixed_hybrid_labels(source, target, 0, 4)
        self.assertEqual(hybrid, 0.5)
        self.assertIsNone(akh)
        self.assertEqual(ipguard, 0.5)

    def test_csv_keeps_multiple_ratios_for_the_same_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed_hybrid.csv"
            writer = FixedHybridScoresCsv(path)
            base = {
                "method": "Fixed-Hybrid",
                "budget": 20,
                "seed": 123456789,
                "source_model": "source",
                "target_model": "target",
                "score": 0.25,
                "dataset": "CIFAR10",
                "pair_label": 1,
                "attack_type": "finetune",
                "hybrid_score": 0.25,
            }
            writer.upsert(
                {
                    **base,
                    "akh_budget": 0,
                    "ipguard_budget": 20,
                    "akh_ratio": 0,
                    "akh_score": None,
                    "ipguard_score": 0.25,
                }
            )
            writer.upsert(
                {
                    **base,
                    "akh_budget": 20,
                    "ipguard_budget": 0,
                    "akh_ratio": 1,
                    "akh_score": 0.25,
                    "ipguard_score": None,
                }
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(handle.seek(0), 0)
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0]), FIXED_HYBRID_COLUMNS)
            self.assertEqual(rows[0]["akh_score"], "")
            self.assertEqual(rows[1]["ipguard_score"], "")

    def test_cache_path_contains_all_experiment_identity_fields(self):
        path = fixed_hybrid_cache_dir(
            Path("cache"),
            "SACBenchmark",
            "CIFAR10",
            123456789,
            20,
            5,
            15,
            "source",
        )
        rendered = str(path)
        self.assertIn("seed=123456789", rendered)
        self.assertIn("method=Fixed-Hybrid", rendered)
        self.assertIn("budget=20", rendered)
        self.assertIn("akh_ratio=0.25", rendered)
        self.assertIn("akh_budget=5,ipguard_budget=15", rendered)

    def test_summary_metrics_treat_lower_scores_as_more_positive(self):
        rows = [
            {"pair_label": "1", "score": "0.0"},
            {"pair_label": "1", "score": "0.5"},
            {"pair_label": "0", "score": "0.5"},
            {"pair_label": "0", "score": "1.0"},
        ]
        self.assertEqual(auc_with_ties(rows), 0.875)
        self.assertEqual(operating_point(rows), (0.5, 0.0))


if __name__ == "__main__":
    unittest.main()
