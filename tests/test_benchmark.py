import hashlib
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from digits.reporting import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_timings_equivalence_and_checkpoint_integrity(self):
        torch.set_num_threads(1)
        path = Path("artifacts/digits.pt")
        before = hashlib.sha256(path.read_bytes()).digest()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.json"
            args = Namespace(checkpoint=path, warmup=1, repeats=3, json=output)
            with redirect_stdout(StringIO()):
                benchmark(args)
            metrics = json.loads(output.read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)
        self.assertEqual(metrics["timed_iterations"], 3)
        self.assertEqual(metrics["warmup_iterations"], 1)
        self.assertLess(metrics["max_probability_difference"], 1e-5)
        for row in metrics["single_image_forward_ms"].values():
            self.assertGreater(row["p50"], 0)
            self.assertGreaterEqual(row["p95"], row["p50"])
        rates = metrics["training_images_per_second_by_batch_size"]
        self.assertEqual(set(rates), {"1", "64"})
        self.assertTrue(all(rate > 0 for rate in rates.values()))
