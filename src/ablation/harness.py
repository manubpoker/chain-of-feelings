"""Ablation harness: context-manager that modifies a_t to test the affect channel.

Five modes:
- NONE: normal operation (control)
- ZERO: a_t = 0 at all timesteps
- CLAMP: a_t clamped to its running mean (removes dynamics)
- NOISE: a_t replaced with Gaussian noise matching trained statistics
- SHUFFLE: a_t from a different input sequence (wrong content, right statistics)
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import torch

from src.affect.injection import AffectInjector


class AblationMode(Enum):
    NONE = "none"
    ZERO = "zero"
    CLAMP = "clamp"
    NOISE = "noise"
    SHUFFLE = "shuffle"


@contextmanager
def ablation_mode(injector: AffectInjector, mode: AblationMode,
                  shuffle_state: torch.Tensor | None = None):
    """Context manager that modifies the affect channel's behaviour.

    Usage:
        with ablation_mode(injector, AblationMode.ZERO):
            outputs = model(inputs)

    For SHUFFLE mode, provide shuffle_state: a pre-computed affect state
    tensor from a different input sequence.
    """
    if mode == AblationMode.NONE:
        yield
        return

    # Save original forward method
    original_forward = injector.channel.forward

    # Collect statistics from history for CLAMP and NOISE modes
    mean_state = None
    std_state = None
    if injector.channel.history:
        history = torch.stack(injector.channel.history, dim=0)
        mean_state = history.mean(dim=0)
        std_state = history.std(dim=0).clamp(min=1e-6)

    def ablated_forward(residual: torch.Tensor) -> torch.Tensor:
        """Modified forward that applies the ablation."""
        # Run original to get shapes right
        result = original_forward(residual)
        batch, seq_len, affect_dim = result.shape

        if mode == AblationMode.ZERO:
            return torch.zeros_like(result)

        elif mode == AblationMode.CLAMP:
            if mean_state is not None:
                return mean_state.unsqueeze(1).expand_as(result)
            return torch.zeros_like(result)

        elif mode == AblationMode.NOISE:
            if mean_state is not None and std_state is not None:
                noise = torch.randn_like(result)
                return mean_state.unsqueeze(1) + std_state.unsqueeze(1) * noise
            return torch.randn_like(result) * 0.1

        elif mode == AblationMode.SHUFFLE:
            if shuffle_state is not None:
                # Use the provided state, expanding to match sequence length
                if shuffle_state.dim() == 2:
                    return shuffle_state.unsqueeze(1).expand_as(result)
                return shuffle_state[:, :seq_len, :]
            # Fallback: permute the batch dimension
            perm = torch.randperm(batch, device=result.device)
            return result[perm]

        return result

    try:
        injector.channel.forward = ablated_forward
        yield
    finally:
        injector.channel.forward = original_forward
