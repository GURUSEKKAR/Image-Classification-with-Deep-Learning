import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class ExportTests(unittest.TestCase):
    def test_export_matches_shipped_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/model.json"
            subprocess.run([sys.executable, "-m", "digits.export_web", "--output", str(output)],
                           check=True)
            actual = json.loads(output.read_text())
            expected = json.loads(Path("web/model.json").read_text())
            layers, reference = actual.pop("layers"), expected.pop("layers")
            self.assertEqual(actual, expected)
            self.assertEqual(len(layers), len(reference))
            for layer, baseline in zip(layers, reference):
                for field in ("w", "b"):
                    np.testing.assert_allclose(layer[field], baseline[field], atol=1e-6, rtol=1e-5)

    def test_output_collision_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"original")
            result = subprocess.run([sys.executable, "-m", "digits.export_web", "--checkpoint",
                                     str(path), "--output", str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ", result.stderr)
            self.assertEqual(path.read_bytes(), b"original")
