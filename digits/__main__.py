"""Run with python -m digits."""

import argparse
from pathlib import Path

import torch

from .data import image_tensor
from .model import inference_model, load_checkpoint
from .reporting import benchmark, evaluate
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
    image = image_tensor(args.image, args.dark_ink, auto_ink=args.auto_ink)
    logits = inference_model(model)((image - checkpoint["mean"]) / checkpoint["std"])
    values, classes = logits[0].softmax(dim=0).topk(3)
    print(f"Predicted digit {classes[0].item()}")
    for digit, score in zip(classes.tolist(), values.tolist()):
        print(f"  {digit}   {score:.2%}")


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
    infer.add_argument("image", type=Path)
    ink = infer.add_mutually_exclusive_group()
    ink.add_argument("--dark-ink", action="store_true", help="use for dark strokes on light paper")
    ink.add_argument("--auto-ink", action="store_true", help="infer polarity from the background")
    infer.set_defaults(run=predict)

    for command in (score, perf, infer):
        command.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    for command in (score, perf):
        command.add_argument("--json", type=Path, help="write the report to this file")
    for command in (fit, score, perf, infer):
        command.add_argument("--threads", type=positive_int, default=1, help="CPU threads (default 1)")
    args = parser.parse_args()
    if args.command == "train" and not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2^63 - 1")
    torch.set_num_threads(args.threads)
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
