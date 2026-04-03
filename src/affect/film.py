"""FiLM (Feature-wise Linear Modulation) conditioning from affect state.

Generates per-layer gamma and beta vectors from the affect state a_t.
Modulation: h_l' = gamma_l * h_l + beta_l

Uses a low-rank bottleneck (rank 4) to force the affect channel to encode
only a few independent modulation directions, preventing it from becoming
a general-purpose bypass around the frozen weights.

Reference: Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .module import AffectConfig


class AffectFiLM(nn.Module):
    """Generate per-layer FiLM parameters (gamma, beta) from affect state.

    Architecture:
        a_t (affect_dim) → down (affect_dim → rank * num_layers) → reshape per layer
        per-layer rank vector → up (rank → model_dim * 2) → split into gamma, beta

    The 'up' projection is shared across layers to save parameters.
    gamma is initialised near 1.0, beta near 0.0 so the initial modulation
    is near-identity (the model behaves normally before training).
    """

    def __init__(self, config: AffectConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.model_dim = config.model_dim
        self.rank = config.film_rank

        # Down projection: affect state → per-layer rank vectors
        self.down = nn.Linear(config.affect_dim, self.rank * self.num_layers)

        # Up projection: rank → (gamma, beta) — shared across layers
        self.up = nn.Linear(self.rank, config.model_dim * 2)

        # Initialise near-identity: gamma ≈ 1, beta ≈ 0
        self._init_near_identity(config.init_scale)

    def _init_near_identity(self, scale: float) -> None:
        """Initialise so gamma ≈ 1 and beta ≈ 0 at the start."""
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        # Up projection: gamma part gets small positive bias, beta part gets zero
        nn.init.normal_(self.up.weight, std=scale)
        with torch.no_grad():
            # bias[:model_dim] = gamma bias → 1.0 (will be added to 1.0 later)
            # bias[model_dim:] = beta bias → 0.0
            self.up.bias.zero_()

    def forward(self, affect_state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute FiLM parameters for all layers from the current affect state.

        Args:
            affect_state: (batch, affect_dim) or (batch, seq_len, affect_dim)

        Returns:
            List of (gamma, beta) tuples, one per layer.
            Each gamma/beta has shape matching the input's batch/seq dimensions + (model_dim,).
        """
        # Down projection to per-layer rank vectors
        down_out = self.down(affect_state)  # (..., rank * num_layers)

        # Reshape to (num_layers, ..., rank)
        shape = affect_state.shape[:-1]  # batch dims
        per_layer = down_out.view(*shape, self.num_layers, self.rank)

        film_params = []
        for l in range(self.num_layers):
            # Extract this layer's rank vector
            rank_vec = per_layer[..., l, :]  # (..., rank)

            # Up projection to gamma/beta
            gamma_beta = self.up(rank_vec)  # (..., model_dim * 2)

            gamma, beta = gamma_beta.chunk(2, dim=-1)

            # gamma is a multiplicative modulation centered at 1.0
            gamma = 1.0 + gamma

            film_params.append((gamma, beta))

        return film_params

    def get_param_count(self) -> int:
        """Total trainable parameters in the FiLM module."""
        return sum(p.numel() for p in self.parameters())
