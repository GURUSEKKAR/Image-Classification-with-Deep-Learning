"""Rebuild the static browser model and example inputs from the trained checkpoint."""
import argparse
import json
from pathlib import Path

from .data import dataset, validate_splits
from .model import inference_model, load_checkpoint
from .io import atomic_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/digits.pt"))
    parser.add_argument("--output", type=Path, default=Path("web/model.json"))
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    fused = inference_model(model)
    layers = [fused.features[0][0], fused.features[1][0], fused.classifier[1], fused.classifier[4]]
    images, labels, digest = dataset()
    if digest != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset differs from the checkpoint.")
    samples = {}
    validate_splits(checkpoint["split_indices"], len(labels))
    for idx in checkpoint["split_indices"]["test"]:
        samples.setdefault(str(labels[idx]), images[idx].flatten().tolist())
    if len(samples) != 10:
        raise ValueError("The sample split must include all ten digits.")
    payload = {"layers": [{"w": layer.weight.flatten().tolist(), "b": layer.bias.tolist()}
                          for layer in layers],
               "mean": checkpoint["mean"], "std": checkpoint["std"], "samples": samples}
    with atomic_output(args.output) as pending:
        pending.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
