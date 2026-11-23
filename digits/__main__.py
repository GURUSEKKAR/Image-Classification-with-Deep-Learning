"""Run with python -m digits."""

import argparse
from pathlib import Path

import torch

from .data import image_tensor
from .model import inference_model, load_checkpoint
from .reporting import benchmark, evaluate, report
from .training import train


CHECKPOINT = Path("artifacts/digits.pt")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


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


def main():
    parser = argparse.ArgumentParser(description="Train and use a handwritten-digit CNN on CPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train a model and save its best validation epoch")
    fit.add_argument("--epochs", type=positive_int, default=100)
    fit.add_argument("--patience", type=positive_int, default=10)
    fit.add_argument("--batch-size", type=positive_int, default=64)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--output", type=Path, default=CHECKPOINT)
    fit.set_defaults(run=train)

    score = commands.add_parser("evaluate", help="report accuracy and class-level metrics")
    score.add_argument("--split", choices=("validation", "test"), default="test")
    score.set_defaults(run=evaluate)
    perf = commands.add_parser("benchmark", help="measure training throughput and prediction latency")
    perf.add_argument("--repeats", type=positive_int, default=200)
    perf.set_defaults(run=benchmark)
    infer = commands.add_parser("predict", help="classify an image file")
    infer.add_argument("images", type=Path, nargs="+", metavar="IMAGE")
    infer.add_argument("--top-k", type=int, choices=range(1, 11), default=3, metavar="1..10",
                       help="number of ranked scores per image (default 3)")
    ink = infer.add_mutually_exclusive_group()
    ink.add_argument("--dark-ink", action="store_true", help="use for dark strokes on light paper")
    ink.add_argument("--auto-ink", action="store_true", help="infer polarity from the background")
    infer.add_argument("--trim", action="store_true", help="crop surrounding margins to the digit")
    infer.set_defaults(run=predict)

    for command in (score, perf, infer):
        command.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    for command in (score, perf, infer):
        command.add_argument("--json", type=Path, help="write the report to this file")
    for command in (fit, score, perf, infer):
        command.add_argument("--threads", type=positive_int, default=1, help="CPU threads (default 1)")
    args = parser.parse_args()
    output = getattr(args, "json", None)
    if output:
        inputs = [args.checkpoint, *getattr(args, "images", [])]
        if any(path.resolve() == output.resolve() for path in inputs):
            parser.error("the JSON destination must be different from each input file")
    if args.command == "train" and not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2^63 - 1")
    torch.set_num_threads(args.threads)
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
