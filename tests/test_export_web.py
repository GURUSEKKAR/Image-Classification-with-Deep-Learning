from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ExportTests(unittest.TestCase):
    def test_export_matches_shipped_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/model.json"
            subprocess.run([sys.executable, "-m", "digits.export_web", "--output", str(output)],
                           check=True)
            self.assertEqual(output.read_bytes(), Path("web/model.json").read_bytes())

    def test_output_collision_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"original")
            result = subprocess.run([sys.executable, "-m", "digits.export_web", "--checkpoint",
                                     str(path), "--output", str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ", result.stderr)
            self.assertEqual(path.read_bytes(), b"original")
