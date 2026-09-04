import unittest

import torch
from gymnasium import spaces

from rllib_model import PKMAIDictCNN


class PKMAIDictCNNTests(unittest.TestCase):
    def test_forward_accepts_image_and_navigation_features(self):
        observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (64, 64, 1), dtype=float),
                "nav": spaces.Box(-1.0, 1.0, (28,), dtype=float),
            }
        )
        action_space = spaces.Discrete(7)
        model = PKMAIDictCNN(observation_space, action_space, 7, {}, "test")
        logits, state = model(
            {
                "obs": {
                    "image": torch.zeros((2, 64, 64, 1)),
                    "nav": torch.zeros((2, 28)),
                }
            },
            [],
            None,
        )

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(state, [])
        self.assertEqual(tuple(model.value_function().shape), (2,))
