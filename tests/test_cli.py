import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps


def cli(*args):
    return subprocess.run([sys.executable, "-m", "digits", *map(str, args)],
                          text=True, capture_output=True)


class CommandTests(unittest.TestCase):
    def test_batch_json_preserves_image_order_and_score_ranking(self):
        example = Path("examples/digit-6.png")
        with TemporaryDirectory() as directory:
            dark = Path(directory) / "dark.png"
            with Image.open(example) as sample:
                ImageOps.invert(sample).save(dark)
            run = cli("predict", dark, example, "--auto-ink", "--top-k", 10, "--json", "-")
        self.assertEqual(run.returncode, 0, run.stderr)
        rows = json.loads(run.stdout)["predictions"]
        self.assertEqual([row["image"] for row in rows], [str(dark), str(example)])
        for row in rows:
            self.assertEqual(row["digit"], 6)
            scores = [s["probability"] for s in row["scores"]]
            self.assertEqual(len(scores), 10)
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertAlmostEqual(sum(scores), 1.0, places=5)

    def test_reports_account_for_every_test_image(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.json"
            run = cli("evaluate", "--batch-size", 7, "--json", output)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report["samples"], 269)
        self.assertEqual(report["correct"], 268)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), report["samples"])
        self.assertEqual(len(report["errors"]), report["samples"] - report["correct"])
        self.assertEqual(sum(row["support"] for row in report["per_class"]), report["samples"])

    def test_invalid_arguments_and_unreadable_files_have_clean_errors(self):
        cases = [
            ("train", "--learning-rate", "nan"), ("train", "--seed", "-1"),
            ("predict", "unused.png", "--top-k", "11"), ("predict", "missing.png"),
            ("predict", "examples/digit-6.png", "--json", "examples/digit-6.png"),
            ("evaluate", "--json", "artifacts/digits.pt"),
        ]
        for args in cases:
            with self.subTest(args=args):
                run = cli(*args)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr.strip())
                self.assertNotIn("Traceback", run.stderr)
