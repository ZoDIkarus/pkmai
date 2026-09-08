"""Real-model isolation assertions (torch / SB3).

Kept in its own module so the pure-logic :mod:`twoby2.isolation` never imports
torch. Used by the integration tests that build two real small PPOs and prove
they share no parameter tensor, no optimizer, and that stepping one changes
nothing in the other.
"""
from __future__ import annotations


def _policy(model):
    return getattr(model, "policy", model)


def param_ids(model):
    return {id(p) for p in _policy(model).parameters()}


def assert_models_isolated(model_a, model_b):
    """Raise AssertionError if the two models share a parameter tensor or an
    optimizer object."""
    shared = param_ids(model_a) & param_ids(model_b)
    if shared:
        raise AssertionError(f"{len(shared)} parameter tensors are shared")
    oa = getattr(_policy(model_a), "optimizer", None)
    ob = getattr(_policy(model_b), "optimizer", None)
    if oa is not None and oa is ob:
        raise AssertionError("the two models share an optimizer object")
    return True


def param_snapshot(model):
    """Detached clone of every parameter, keyed by name."""
    return {n: p.detach().clone() for n, p in _policy(model).named_parameters()}


def params_unchanged(model, snapshot):
    """True iff every parameter still bit-exactly equals its snapshot."""
    import torch
    for n, p in _policy(model).named_parameters():
        if n not in snapshot or not torch.equal(p.detach(), snapshot[n]):
            return False
    return True


def max_param_delta(model, snapshot):
    """Largest absolute change of any parameter vs the snapshot (0.0 == frozen)."""
    import torch
    m = 0.0
    for n, p in _policy(model).named_parameters():
        if n in snapshot:
            m = max(m, (p.detach() - snapshot[n]).abs().max().item())
    return m
