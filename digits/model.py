"""Classifier and its inference-time transformation."""

from copy import deepcopy
from math import isfinite
from pickle import UnpicklingError

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 24), conv_block(24, 48))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(48 * 2 * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 10)
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def inference_model(model):
    fused = deepcopy(model).eval()
    for block in fused.features:
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = nn.Identity()
    return fused


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, UnpicklingError, IndexError) as error:
        raise ValueError(f"Could not decode model checkpoint {path}.") from error
    required = {"weights", "mean", "std", "split_indices", "dataset_sha256", "best_epoch", "config"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"{path} is missing required digit classifier fields.")
    for field in ("mean", "std"):
        value = checkpoint[field]
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"Checkpoint {field} must be a finite number.")
    if checkpoint["std"] <= 0:
        raise ValueError("Checkpoint standard deviation must be positive.")
    config = checkpoint["config"]
    if not isinstance(config, dict) or any(type(config.get(k)) is not int for k in ("batch_size", "seed")):
        raise ValueError("Checkpoint must specify an integer batch size and seed.")
    if config["batch_size"] < 1 or not 0 <= config["seed"] < 2**63:
        raise ValueError("Checkpoint batch size or seed is outside the supported range.")
    model = DigitCNN()
    try:
        model.load_state_dict(checkpoint["weights"])
    except (TypeError, RuntimeError) as error:
        raise ValueError("Checkpoint weights do not match the digit CNN.") from error
    return model.eval(), checkpoint
