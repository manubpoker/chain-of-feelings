"""AffectChannel: GRU-based persistent affect state.

Maintains a_t ∈ ℝ^affect_dim that evolves across tokens via GRU recurrence.
Input: compressed residual stream snapshot from a designated readout layer.
Output: the current affect state, used by AffectFiLM for per-layer modulation.

The GRU ensures temporal persistence — a_t depends on a_{t-1}, creating a slowly-
varying internal "mood" that carries information across the token sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AffectConfig:
    """Configuration for the affect channel."""
    affect_dim: int = 16          # dimensionality of a_t
    model_dim: int = 2048         # Gemma 3n E2B hidden size
    readout_layer: int = 8        # which layer to read the residual stream from
    compress_dim: int = 64        # intermediate compression before GRU input
    film_rank: int = 4            # rank of FiLM modulation (bottleneck)
    num_layers: int = 26          # number of transformer layers
    init_scale: float = 0.01     # scale for near-identity FiLM init
    use_mlp: bool = False         # use MLP instead of GRU (better gradient flow)


class AffectChannel(nn.Module):
    """Affect state computed from residual stream.

    Two modes:
    - GRU (default): persistent state a_t that evolves across tokens via recurrence.
      Biologically motivated but gradients vanish through the recurrence.
    - MLP (use_mlp=True): per-token affect computed directly from residual stream,
      then smoothed with exponential moving average. Better gradient flow.
    """

    def __init__(self, config: AffectConfig):
        super().__init__()
        self.config = config
        self.use_mlp = config.use_mlp

        # Compress residual stream
        self.compress = nn.Sequential(
            nn.Linear(config.model_dim, config.compress_dim),
            nn.GELU(),
        )

        if self.use_mlp:
            # MLP: direct mapping from compressed residual to affect state
            self.mlp = nn.Sequential(
                nn.Linear(config.compress_dim, config.compress_dim),
                nn.GELU(),
                nn.Linear(config.compress_dim, config.affect_dim),
                nn.Tanh(),  # bound to [-1, 1]
            )
            # Learnable EMA smoothing factor (how much to blend with previous state)
            self.ema_logit = nn.Parameter(torch.tensor(1.0))  # sigmoid → ~0.73
        else:
            # GRU cell: updates a_t from compressed input
            self.gru = nn.GRUCell(
                input_size=config.compress_dim,
                hidden_size=config.affect_dim,
            )

        # Current state (set during forward pass, persists across tokens)
        self._state: torch.Tensor | None = None

        # History of states for regularisation and analysis
        self._history: list[torch.Tensor] = []

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset affect state to zero at the start of a new sequence."""
        device = device or next(self.parameters()).device
        self._state = torch.zeros(batch_size, self.config.affect_dim, device=device)
        self._history = []

    @property
    def state(self) -> torch.Tensor | None:
        """Current affect state a_t."""
        return self._state

    @property
    def history(self) -> list[torch.Tensor]:
        """List of all affect states from the current sequence."""
        return self._history

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Update affect state from residual stream snapshot.

        Args:
            residual: tensor of shape (batch, seq_len, model_dim) from readout layer.
                      We process each position sequentially to maintain GRU recurrence.

        Returns:
            Tensor of shape (batch, seq_len, affect_dim) — affect states for all positions.
        """
        batch, seq_len, _ = residual.shape

        if self._state is None:
            self.reset(batch, residual.device)

        # Compress the residual stream
        compressed = self.compress(residual)  # (batch, seq_len, compress_dim)

        if self.use_mlp:
            # MLP: compute per-token affect, then smooth with EMA
            raw_affect = self.mlp(compressed)  # (batch, seq_len, affect_dim)
            alpha = torch.sigmoid(self.ema_logit)  # smoothing factor
            states = []
            for t in range(seq_len):
                new_state = raw_affect[:, t, :]
                self._state = alpha * self._state + (1 - alpha) * new_state
                states.append(self._state)
                self._history.append(self._state.detach())
        else:
            # GRU: recurrent update
            states = []
            for t in range(seq_len):
                self._state = self.gru(compressed[:, t, :], self._state)
                states.append(self._state)
                self._history.append(self._state.detach())

        return torch.stack(states, dim=1)  # (batch, seq_len, affect_dim)

    def get_regularisation_terms(self) -> dict[str, torch.Tensor]:
        """Compute regularisation losses for the affect channel.

        Returns dict with:
        - bottleneck: KL divergence pushing a_t toward N(0, I)
        - stability: L2 penalty on consecutive state differences
        """
        if len(self._history) < 2:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return {"bottleneck": zero, "stability": zero}

        states = torch.stack(self._history, dim=0)  # (T, batch, affect_dim)

        # Bottleneck: encourage low-norm states (simplified KL against N(0,I))
        # Full KL = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
        # Simplified: just penalise ||a_t||^2 (pushes mean toward 0)
        bottleneck = (states ** 2).mean()

        # Stability: penalise large changes between consecutive states
        diffs = states[1:] - states[:-1]
        stability = (diffs ** 2).mean()

        return {"bottleneck": bottleneck, "stability": stability}
