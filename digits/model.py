"""Classifier and its inference-time transformation."""

from copy import deepcopy

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
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DigitCNN()
    model.load_state_dict(checkpoint["weights"])
    return model.eval(), checkpoint
