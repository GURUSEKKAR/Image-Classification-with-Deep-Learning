import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from digits.io import atomic_output
from digits.model import inference_model, load_checkpoint
from digits.reporting import split_tensors


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.checkpoint = load_checkpoint("artifacts/digits.pt")

    def test_fusion_preserves_all_test_predictions(self):
        images, _ = split_tensors(self.checkpoint, "test")
        inputs = (images - self.checkpoint["mean"]) / self.checkpoint["std"]
        with torch.inference_mode():
            first = self.model(inputs).softmax(1)
            second = inference_model(self.model)(inputs).softmax(1)
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(first.argmax(1), second.argmax(1)))

    def test_normalization_matches_training_pixels(self):
        images, _ = split_tensors(self.checkpoint, "train")
        self.assertAlmostEqual(float(images.numpy().mean()), self.checkpoint["mean"])
        self.assertAlmostEqual(float(images.numpy().std()), self.checkpoint["std"])

    def test_invalid_metadata_and_weights_fail_cleanly(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            for change in ({"std": 0}, {"mean": float("nan")}, {"format_version": 8}, {"weights": {}}):
                torch.save({**self.checkpoint, **change}, path)
                with self.subTest(field=next(iter(change))), self.assertRaises(ValueError):
                    load_checkpoint(path)
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "decode"):
                load_checkpoint(path)

    def test_evaluation_requires_the_original_dataset(self):
        with self.assertRaisesRegex(ValueError, "contents differ"):
            split_tensors({**self.checkpoint, "dataset_sha256": "different"}, "test")

    def test_atomic_write_failure_preserves_existing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result"
            path.write_text("original")
            with self.assertRaises(OSError):
                with atomic_output(path) as pending:
                    pending.write_text("partial")
                    raise OSError("interrupted")
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(path.parent.iterdir()), [path])
