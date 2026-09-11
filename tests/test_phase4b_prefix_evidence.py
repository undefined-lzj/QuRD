import unittest

from scripts.analyze_phase4b_prefix_evidence import (
    PairMetadata,
    PairTrace,
    TracePoint,
    auc_lower_is_positive,
    average_ranks,
    build_prefix_rows,
    operating_point_lower_score,
    spearman_with_ties,
)


class Phase4BPrefixEvidenceTest(unittest.TestCase):
    @staticmethod
    def points(method, mismatches):
        return [
            TracePoint(
                query_index=index,
                query_id=f"{method}-{index}",
                source_label=0,
                target_label=mismatch,
                mismatch=mismatch,
            )
            for index, mismatch in enumerate(mismatches, start=1)
        ]

    def test_prefix_and_hybrid_scores_use_ordered_mismatches(self):
        akh_mismatches = [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5
        ipguard_mismatches = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        pair = PairTrace(
            metadata=PairMetadata("source", "target", 1, "finetune", 123456789),
            methods={
                "AKH": self.points("akh", akh_mismatches),
                "IPGuard": self.points("ipguard", ipguard_mismatches),
            },
        )

        row = build_prefix_rows({("source", "target"): pair})[0]

        self.assertEqual(row["akh_score_5"], 1.0)
        self.assertEqual(row["akh_score_10"], 0.5)
        self.assertAlmostEqual(row["akh_score_15"], 2 / 3)
        self.assertEqual(row["akh_score_20"], 0.5)
        self.assertEqual(row["ipguard_score_5"], 0.0)
        self.assertEqual(row["ipguard_score_10"], 0.5)
        self.assertAlmostEqual(row["ipguard_score_15"], 1 / 3)
        self.assertEqual(row["ipguard_score_20"], 0.5)
        self.assertEqual(row["hybrid_15_5"], 0.5)
        self.assertEqual(row["hybrid_10_10"], 0.5)
        self.assertEqual(row["hybrid_5_15"], 0.5)

    def test_auc_uses_lower_scores_as_positive(self):
        labels = [1, 1, 0, 0]
        scores = [0.0, 0.5, 0.5, 1.0]
        self.assertEqual(auc_lower_is_positive(labels, scores), 0.875)

    def test_operating_point_returns_score_threshold_and_actual_fpr(self):
        labels = [1, 1, 0, 0]
        scores = [0.0, 0.5, 0.5, 1.0]
        tpr, fpr, threshold = operating_point_lower_score(labels, scores)
        self.assertEqual((tpr, fpr, threshold), (0.5, 0.0, 0.0))

    def test_average_ranks_handle_ties(self):
        self.assertEqual(average_ranks([1.0, 2.0, 2.0, 4.0]), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_is_tie_aware(self):
        self.assertAlmostEqual(
            spearman_with_ties([0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]),
            0.8333333333333334,
        )


if __name__ == "__main__":
    unittest.main()
