"""Rank digit predictions for image files."""

import torch

from .data import image_tensor
from .io import report
from .model import inference_model, load_checkpoint


@torch.inference_mode()
def predict(args):
    model, checkpoint = load_checkpoint(args.checkpoint)
    model = inference_model(model)
    predictions = []
    for start in range(0, len(args.images), 64):
        paths = args.images[start : start + 64]
        images = torch.cat([
            image_tensor(path, args.dark_ink, auto_ink=args.auto_ink, trim=args.trim)
            for path in paths
        ])
        logits = model((images - checkpoint["mean"]) / checkpoint["std"])
        values, classes = logits.softmax(dim=1).topk(args.top_k)
        for path, digits, scores in zip(paths, classes.tolist(), values.tolist()):
            if args.json:
                predictions.append({
                    "image": str(path), "digit": digits[0],
                    "scores": [{"digit": d, "probability": s} for d, s in zip(digits, scores)],
                })
                continue
            if len(args.images) > 1:
                print(path)
            print(f"Predicted digit {digits[0]}")
            for digit, score in zip(digits, scores):
                print(f"  {digit}   {score:.2%}")
    if args.json:
        report({"predictions": predictions}, args.json)
