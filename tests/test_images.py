import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image, ImageOps

from digits.data import image_tensor


class ImageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "digit.png"
        with Image.open("examples/digit-6.png") as source:
            self.digit = source.convert("L")

    def test_black_ink_and_auto_polarity_match_original(self):
        self.digit.save(self.path)
        expected = image_tensor(self.path)
        ImageOps.invert(self.digit).save(self.path)
        torch.testing.assert_close(image_tensor(self.path, True), expected)
        torch.testing.assert_close(image_tensor(self.path, auto_ink=True), expected)

    def test_transparent_dark_strokes(self):
        dark = Image.new("RGBA", self.digit.size, "black")
        dark.putalpha(self.digit)
        dark.save(self.path)
        torch.testing.assert_close(image_tensor(self.path, auto_ink=True), image_tensor(self.path, True))

    def test_trimming_removes_padding(self):
        self.digit.save(self.path)
        expected = image_tensor(self.path, trim=True)
        ImageOps.expand(self.digit, border=(40, 12, 4, 80)).save(self.path)
        torch.testing.assert_close(image_tensor(self.path, trim=True), expected)

    def test_uniform_and_transparent_images_are_rejected(self):
        for color in ("white", "black", (0, 0, 0, 0)):
            Image.new("RGBA", (16, 16), color).save(self.path)
            with self.subTest(color=color), self.assertRaisesRegex(ValueError, "contrast"):
                image_tensor(self.path, auto_ink=True)
