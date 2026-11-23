"""Run with python -m digits."""

import argparse
from importlib import import_module
from math import isfinite
from pathlib import Path

CHECKPOINT = Path("artifacts/digits.pt")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="Train and use a handwritten-digit CNN on CPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train a model and save its best validation epoch")
    fit.add_argument("--epochs", type=positive_int, default=100)
    fit.add_argument("--patience", type=positive_int, default=10)
    fit.add_argument("--batch-size", type=positive_int, default=64)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--learning-rate", type=float, default=0.002)
    fit.add_argument("--output", type=Path, default=CHECKPOINT)
    fit.set_defaults(module="training")

    score = commands.add_parser("evaluate", help="report accuracy and class-level metrics")
    score.add_argument("--split", choices=("validation", "test"), default="test")
    score.add_argument("--batch-size", type=positive_int, default=64)
    score.set_defaults(module="reporting")
    perf = commands.add_parser("benchmark", help="measure training throughput and prediction latency")
    perf.add_argument("--repeats", type=positive_int, default=200)
    perf.add_argument("--warmup", type=positive_int, default=10)
    perf.set_defaults(module="reporting")
    infer = commands.add_parser("predict", help="classify an image file")
    infer.add_argument("images", type=Path, nargs="+", metavar="IMAGE")
    infer.add_argument("--top-k", type=int, choices=range(1, 11), default=3, metavar="1..10",
                       help="number of ranked scores per image (default 3)")
    ink = infer.add_mutually_exclusive_group()
    ink.add_argument("--dark-ink", action="store_true", help="use for dark strokes on light paper")
    ink.add_argument("--auto-ink", action="store_true", help="infer polarity from the background")
    infer.add_argument("--trim", action="store_true", help="crop surrounding margins to the digit")
    infer.set_defaults(module="prediction")

    for command in (score, perf, infer):
        command.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    for command in (score, perf, infer):
        command.add_argument("--json", type=Path, help="write the report to a file or - for stdout")
    for command in (fit, score, perf, infer):
        command.add_argument("--threads", type=positive_int, default=1, help="CPU threads (default 1)")
    args = parser.parse_args()
    output = getattr(args, "json", None)
    if output and str(output) != "-":
        inputs = [args.checkpoint, *getattr(args, "images", [])]
        if any(path.resolve() == output.resolve() for path in inputs):
            parser.error("the JSON destination must be different from each input file")
    if args.command == "train" and not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2^63 - 1")
    if args.command == "train" and (not isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    import torch

    torch.set_num_threads(args.threads)
    run = getattr(import_module(f".{args.module}", __package__), args.command)
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
