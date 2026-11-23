"""Fit the CNN and keep the weights with the lowest validation loss."""

import platform
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .data import augment, dataset, split_indices
from .io import atomic_output
from .model import CHECKPOINT_VERSION, DigitCNN


def train_step(model, optimizer, images, labels, mean, std):
    inputs = (augment(images) - mean) / std
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return loss.detach()


def save_checkpoint(checkpoint, path: Path):
    with atomic_output(path) as pending:
        torch.save(checkpoint, pending)


def train(args):
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    images, labels, fingerprint = dataset()
    splits = split_indices(labels, args.seed)
    train_x = images[splits["train"]]
    mean, std = float(train_x.mean()), float(train_x.std())
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(labels[splits["train"]])
    val_x = (torch.from_numpy(images[splits["validation"]]) - mean) / std
    val_y = torch.from_numpy(labels[splits["validation"]])
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "mean": mean,
        "std": std,
        "split_indices": splits,
        "dataset_sha256": fingerprint,
        "history": history,
        "config": {
            "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "threads": torch.get_num_threads(),
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("torch", "numpy", "scikit-learn", "Pillow")},
        },
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        start = perf_counter()
        for indices in torch.randperm(len(train_x)).split(args.batch_size):
            loss = train_step(model, optimizer, train_x[indices], train_y[indices], mean, std)
            total_loss += loss.item() * len(indices)
        throughput = len(train_x) / (perf_counter() - start)
        model.eval()
        with torch.inference_mode():
            logits = model(val_x)
            val_loss = F.cross_entropy(logits, val_y).item()
            correct = (logits.argmax(1) == val_y).sum().item()
        accuracy = correct / len(val_y)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(train_x),
                            validation_loss=val_loss, validation_accuracy=accuracy,
                            learning_rate=lr, train_images_per_second=throughput))
        print(f"Epoch {epoch:3d}  train loss {total_loss / len(train_x):.4f}  "
              f"validation loss {val_loss:.4f}  validation accuracy {accuracy:.2%}", flush=True)
        if val_loss < best_loss:
            best_epoch, best_loss = epoch, val_loss
            checkpoint["best_epoch"] = epoch
            checkpoint["weights"] = deepcopy(model.state_dict())
            save_checkpoint(checkpoint, args.output)
        schedule.step(val_loss)
        if epoch - best_epoch >= args.patience:
            break

    save_checkpoint(checkpoint, args.output)
    print(f"Saved epoch {best_epoch} to {args.output} after {epoch} epochs.")
