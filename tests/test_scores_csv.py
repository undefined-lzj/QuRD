import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qurd.experiments import SCORE_COLUMNS, ScoresCsv


class ScoresCsvTests(unittest.TestCase):
    def test_writes_required_columns_and_deduplicates_model_pair(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            scores = ScoresCsv(path)
            record = {
                "dataset": "mini-imagenet",
                "method": "AKH",
                "budget": 10,
                "seed": 123456789,
                "source_model": "ResNet",
                "target_model": "ViT",
                "is_positive": False,
                "attack_type": "unrelated",
                "score": 0.2,
            }

            scores.upsert(record)
            scores.upsert({**record, "score": 0.3})
            ScoresCsv(path).upsert({**record, "score": 0.4})

            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, SCORE_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "AKH")
            self.assertEqual(rows[0]["budget"], "10")
            self.assertEqual(rows[0]["seed"], "123456789")
            self.assertEqual(rows[0]["source_model"], "ResNet")
            self.assertEqual(rows[0]["target_model"], "ViT")
            self.assertEqual(rows[0]["is_positive"], "False")
            self.assertEqual(rows[0]["attack_type"], "unrelated")
            self.assertEqual(rows[0]["score"], "0.4")

    def test_reads_legacy_label_but_drops_it_when_rewriting(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text(
                "method,budget,seed,source_model,target_model,label,attack_type,score,dataset\n"
                "AKH,10,1,source,target,positive,finetune,0.2,CIFAR10\n",
                encoding="utf-8",
            )

            scores = ScoresCsv(path)
            scores.upsert(
                {
                    "method": "AKH",
                    "budget": 10,
                    "seed": 1,
                    "source_model": "source",
                    "target_model": "target",
                    "is_positive": True,
                    "attack_type": "finetune",
                    "score": 0.3,
                    "dataset": "CIFAR10",
                }
            )

            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, SCORE_COLUMNS)
            self.assertNotIn("label", reader.fieldnames)
            self.assertEqual(rows[0]["is_positive"], "True")


if __name__ == "__main__":
    unittest.main()
