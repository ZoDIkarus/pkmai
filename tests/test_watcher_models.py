import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from watcher_models import validate_policy


class ModelValidationTests(unittest.TestCase):
    def model(self, value):
        return SimpleNamespace(policy=SimpleNamespace(state_dict=lambda: {'weight': torch.tensor([value])}), predict=Mock())

    def test_nan_and_infinity_rejected_before_prediction(self):
        for value in (float('nan'), float('inf')):
            model = self.model(value)
            with self.assertRaises(ValueError):
                validate_policy(model, {})
            model.predict.assert_not_called()

    def test_valid_weights_still_require_successful_prediction(self):
        model = self.model(1.)
        model.predict.side_effect = ValueError('bad logits')
        with self.assertRaises(ValueError):
            validate_policy(model, {})

    def test_finite_predicting_model_accepted(self):
        model = self.model(1.)
        validate_policy(model, {'nav': 1})
        model.predict.assert_called_once_with({'nav': 1}, deterministic=False)
