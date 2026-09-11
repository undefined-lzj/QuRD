import unittest

import torch

from scripts.export_phase4a_query_trace import (
    BUDGET,
    TRACE_COLUMNS,
    rows_for_pair,
    stable_query_id,
)


class Phase4AQueryTraceTest(unittest.TestCase):
    def test_csv_columns_match_phase4a_contract(self):
        self.assertEqual(
            TRACE_COLUMNS,
            [
                "benchmark",
                "dataset",
                "seed",
                "source_model",
                "target_model",
                "pair_label",
                "attack_type",
                "query_method",
                "query_index",
                "query_id",
                "source_label",
                "target_label",
                "mismatch",
            ],
        )

    def test_query_id_is_stable_and_method_specific(self):
        query = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        self.assertEqual(stable_query_id("AKH", query), stable_query_id("AKH", query.clone()))
        self.assertNotEqual(stable_query_id("AKH", query), stable_query_id("IPGuard", query))

    def test_rows_have_complete_indices_and_correct_mismatch(self):
        source = torch.tensor([index % 3 for index in range(BUDGET)])
        target = source.clone()
        target[1] = (target[1] + 1) % 3
        query_ids = [f"akh-{index}" for index in range(1, BUDGET + 1)]

        rows = rows_for_pair(
            target_model="target",
            pair_label=1,
            attack_type="finetune",
            query_method="AKH",
            query_ids=query_ids,
            source_labels=source,
            target_labels=target,
        )

        self.assertEqual(len(rows), BUDGET)
        self.assertEqual({row["query_index"] for row in rows}, set(range(1, 21)))
        self.assertEqual(sum(row["mismatch"] for row in rows), 1)
        self.assertEqual(rows[1]["mismatch"], 1)
        self.assertEqual(rows[0]["mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
