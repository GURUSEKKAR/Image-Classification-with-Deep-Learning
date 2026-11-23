import unittest

import numpy as np
import torch

from digits.data import augment, dataset, split_indices, validate_splits


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.labels, cls.digest = dataset()

    def test_input_format_and_range(self):
        self.assertEqual(self.images.shape, (1797, 1, 8, 8))
        self.assertEqual(self.images.dtype, np.float32)
        self.assertEqual((self.images.min(), self.images.max()), (0, 1))
        self.assertEqual(len(self.digest), 64)

    def test_stratification_is_reproducible_and_exhaustive(self):
        partitions = split_indices(self.labels, 42)
        validate_splits(partitions, len(self.images))
        self.assertEqual(partitions, split_indices(self.labels, 42))
        self.assertNotEqual(partitions, split_indices(self.labels, 43))
        for name, indices in partitions.items():
            self.assertEqual(set(self.labels[indices]), set(range(10)))
            if name != "train":
                np.testing.assert_array_equal(
                    np.bincount(self.labels[indices]), np.rint(0.15 * np.bincount(self.labels))
                )

    def test_overlapping_or_incomplete_splits_fail(self):
        parts = {"train": [0], "validation": [1], "test": [2]}
        for invalid in ([1], [-1], [], [True], [5]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                validate_splits({**parts, "test": invalid}, 3)

    def test_augmentation_is_seeded_and_preserves_input(self):
        inputs = torch.from_numpy(self.images[:8].copy())
        torch.manual_seed(42)
        first = augment(inputs)
        torch.manual_seed(42)
        self.assertTrue(torch.equal(first, augment(inputs)))
        self.assertFalse(torch.equal(first, inputs))
        torch.testing.assert_close(inputs, torch.from_numpy(self.images[:8]))
        self.assertTrue(((first >= 0) & (first <= 1)).all())
