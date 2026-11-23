"""Measure upload contrast handling with the reference model's validation split."""
import json
from pathlib import Path
import subprocess

from digits.model import load_checkpoint
from digits.reporting import split_tensors


def main():
    root = Path(__file__).resolve().parents[1]
    _, checkpoint = load_checkpoint(root / "artifacts/digits.pt")
    images, labels = split_tensors(checkpoint, "validation")
    rows = [dict(pixels=image.flatten().tolist(), label=int(label))
            for image, label in zip(images, labels)]
    subprocess.run(["node", str(root / "scripts/check-upload-accuracy.mjs")],
                   input=json.dumps(rows), text=True, check=True)


if __name__ == "__main__":
    main()
