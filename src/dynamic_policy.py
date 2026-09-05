"""Small CPU-friendly actor-critic used by the dynamic PKMAI rollout protocol."""

from __future__ import annotations

import torch
from torch import nn


class PKMAIPolicy(nn.Module):
    action_count = 7

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.shared = nn.Sequential(nn.Linear(32 * 4 * 4 + 28, 256), nn.ReLU())
        self.policy = nn.Linear(256, self.action_count)
        self.value = nn.Linear(256, 1)

    def forward(self, images: torch.Tensor, nav: torch.Tensor):
        image = images.float() / 255.0
        if image.ndim == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        features = self.shared(torch.cat((self.image_encoder(image), nav.float()), dim=1))
        return self.policy(features), self.value(features).squeeze(1)
