"""RLlib model for PKMAI's image and RAM/navigation Dict observation."""

from __future__ import annotations

import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class PKMAIDictCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
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
        self.policy = nn.Linear(256, num_outputs)
        self.value = nn.Linear(256, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        image = obs["image"].float() / 255.0
        if image.ndim == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        nav = obs["nav"].float().reshape(image.shape[0], -1)
        features = self.shared(torch.cat((self.image_encoder(image), nav), dim=1))
        self._value_out = self.value(features).squeeze(1)
        return self.policy(features), state

    def value_function(self):
        return self._value_out
