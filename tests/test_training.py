import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from digits import training
from digits.model import load_checkpoint


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.pt"
        self.config = Namespace(seed=12, epochs=3, patience=2, batch_size=64,
                                learning_rate=0.002, output=self.path)

    def fit(self):
        with redirect_stdout(StringIO()):
            training.train(self.config)
        return load_checkpoint(self.path)[1]

    def test_best_epoch_and_reproducibility(self):
        first, second = self.fit(), self.fit()
        history = first["history"]
        self.assertEqual(first["best_epoch"], min(history, key=lambda row: row["validation_loss"])["epoch"])
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        for key in first["weights"]:
            self.assertTrue(torch.equal(first["weights"][key], second["weights"][key]), key)

    def test_interruption_leaves_a_loadable_best_checkpoint(self):
        original = training.train_step
        def interrupt_after_first_checkpoint(*args):
            if self.path.exists():
                raise KeyboardInterrupt
            return original(*args)
        with patch.object(training, "train_step", side_effect=interrupt_after_first_checkpoint):
            with self.assertRaises(KeyboardInterrupt):
                self.fit()
        checkpoint = load_checkpoint(self.path)[1]
        self.assertEqual(checkpoint["best_epoch"], 1)
        self.assertEqual(len(checkpoint["history"]), 1)
