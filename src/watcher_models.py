"""Validate replacement policies before switching the visible evaluator."""
import torch


def validate_policy(model, observation):
    bad = [name for name, value in model.policy.state_dict().items()
           if not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f'non-finite policy parameters: {bad[:3]}')
    # Also rejects incompatible observations and non-finite action logits.
    model.predict(observation, deterministic=False)


def load_valid_policy(path, observation):
    from stable_baselines3 import PPO
    model = PPO.load(path, device='cpu')
    validate_policy(model, observation)
    return model
