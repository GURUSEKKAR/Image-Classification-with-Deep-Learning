import json
from pathlib import Path
import shutil
import subprocess
import unittest

import torch

from digits.model import inference_model, load_checkpoint
from digits.reporting import split_tensors

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for browser model parity")
class WebParityTests(unittest.TestCase):
    def test_held_out_probabilities(self):
        model, checkpoint = load_checkpoint(ROOT / "artifacts/digits.pt")
        images, _ = split_tensors(checkpoint, "test")
        script = """
          import fs from 'node:fs';
          import {probabilities, validateModel} from './web/model.mjs';
          const model = JSON.parse(fs.readFileSync('web/model.json'));
          validateModel(model);
          const inputs = JSON.parse(fs.readFileSync(0));
          console.log(JSON.stringify(inputs.map(image => probabilities(image, model))));
        """
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script], cwd=ROOT,
            input=json.dumps(images.flatten(1).tolist()), capture_output=True, text=True, check=True,
        )
        actual = torch.tensor(json.loads(result.stdout))
        with torch.inference_mode():
            expected = inference_model(model)((images-checkpoint["mean"]) / checkpoint["std"]).softmax(1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))
