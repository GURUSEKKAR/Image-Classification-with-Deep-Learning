"""Evaluation on the saved split and CPU performance measurements."""

import json
import platform
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as F

from .data import dataset
from .model import inference_model, load_checkpoint
from .training import train_step


def split_tensors(checkpoint, split):
    images, labels, fingerprint = dataset()
    if fingerprint != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset contents differ from the training checkpoint.")
    indices = checkpoint["split_indices"][split]
    return torch.from_numpy(images[indices]), torch.from_numpy(labels[indices])


def report(result, path):
    output = json.dumps(result, indent=2, allow_nan=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(output)


@torch.inference_mode()
def evaluate(args):
    model, checkpoint = load_checkpoint(args.checkpoint)
    model = inference_model(model)
    images, labels = split_tensors(checkpoint, args.split)
    logits = model((images - checkpoint["mean"]) / checkpoint["std"])
    predicted = logits.argmax(dim=1).numpy()
    actual = labels.numpy()
    classes = np.arange(10)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0
    )
    correct = int(np.sum(predicted == actual))
    result = {
        "split": args.split, "samples": len(labels), "correct": correct,
        "accuracy": correct / len(labels), "macro_f1": float(f1.mean()),
        "loss": F.cross_entropy(logits, labels).item(), "best_epoch": checkpoint["best_epoch"],
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
        "per_class": [
            dict(digit=int(i), precision=float(precision[i]), recall=float(recall[i]),
                 f1=float(f1[i]), support=int(support[i])) for i in classes
        ],
    }
    print(f"{args.split.capitalize()} accuracy {result['accuracy']:.2%} ({correct}/{len(labels)})  "
          f"macro F1 {result['macro_f1']:.4f}")
    report(result, args.json)


def training_rate(model, images, labels, checkpoint, batch_size, repeats, warmup):
    """Time full optimizer updates on private copies with the same image stream."""
    torch.manual_seed(checkpoint["config"]["seed"])
    working = deepcopy(model).train()
    optimizer = torch.optim.AdamW(working.parameters(), lr=0.002, weight_decay=0.01)
    for step in range(warmup + repeats):
        start = (step * batch_size) % len(images)
        indices = torch.arange(start, start + batch_size) % len(images)
        if step == warmup:
            clock = perf_counter()
        train_step(working, optimizer, images[indices], labels[indices],
                   checkpoint["mean"], checkpoint["std"])
    return repeats * batch_size / (perf_counter() - clock)


def benchmark(args):
    model, checkpoint = load_checkpoint(args.checkpoint)
    images, labels = split_tensors(checkpoint, "train")
    tile = (images[:1] - checkpoint["mean"]) / checkpoint["std"]
    fused = inference_model(model)
    latency = {"original": [], "fused": []}
    warmup = 10
    with torch.inference_mode():
        check = (images[:64] - checkpoint["mean"]) / checkpoint["std"]
        original_scores = model(check).softmax(1)
        fused_scores = fused(check).softmax(1)
        torch.testing.assert_close(fused_scores, original_scores, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fused_scores.argmax(1), original_scores.argmax(1))
        # Alternate order to reduce systematic bias from CPU warming or cooling.
        variants = [("original", model), ("fused", fused)]
        for step in range(warmup + args.repeats):
            for name, variant in variants:
                start = perf_counter()
                variant(tile)
                elapsed_ms = (perf_counter() - start) * 1000
                if step >= warmup:
                    latency[name].append(elapsed_ms)
            variants.reverse()
    batch_sizes = sorted({1, checkpoint["config"]["batch_size"]})
    throughput = {
        str(size): training_rate(model, images, labels, checkpoint, size, args.repeats, warmup)
        for size in batch_sizes
    }
    result = {
        "device": "cpu", "architecture": platform.machine(), "system": platform.system(),
        "torch": str(torch.__version__), "threads": torch.get_num_threads(),
        "warmup_iterations": warmup, "timed_iterations": args.repeats,
        "fusion_check_samples": len(check),
        "max_probability_difference": (fused_scores - original_scores).abs().max().item(),
        "training_images_per_second_by_batch_size": throughput,
        "single_image_forward_ms": {
            name: {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
            for name, values in latency.items()
        },
    }
    report(result, args.json)
