import unittest

from qurd.benchmark.sac import SACBenchmark


class SacBenchmarkTests(unittest.TestCase):
    def test_pairs_cover_positive_and_negative_models(self):
        benchmark = SACBenchmark(None, None, "cpu")
        pairs = list(benchmark.pairs("CIFAR10"))

        self.assertEqual(len(pairs), 131)
        self.assertEqual(len({source for source, _ in pairs}), 1)

        metadata = [benchmark.pair_metadata(source, target) for source, target in pairs]
        self.assertEqual(sum(item["is_positive"] for item in metadata), 101)
        self.assertEqual(sum(not item["is_positive"] for item in metadata), 30)

    def test_pair_metadata_identifies_attack_type(self):
        benchmark = SACBenchmark(None, None, "cpu")
        source = "train(vgg_model,CIFAR10,base)"

        self.assertEqual(
            benchmark.pair_metadata(source, source),
            {"is_positive": True, "attack_type": "same"},
        )
        self.assertEqual(
            benchmark.pair_metadata(source, source + "->fineprune(0)"),
            {
                "is_positive": True,
                "attack_type": "fineprune",
            },
        )
        self.assertEqual(
            benchmark.pair_metadata(source, "train(resnet18,CIFAR10,5)"),
            {
                "is_positive": False,
                "attack_type": "unrelated",
            },
        )


if __name__ == "__main__":
    unittest.main()
