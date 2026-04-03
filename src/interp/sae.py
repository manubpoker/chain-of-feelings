"""Sparse Autoencoder training on collected activations.

Trains small SAEs on cached residual stream activations to identify
features that are modulated by the affect channel.

Comparison methodology:
1. Collect activations with affect ON and affect OFF
2. Train SAEs on each condition
3. Compare: which features only appear in affect-ON? Which change pattern?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Sparse autoencoder configuration."""
    input_dim: int = 2048      # model hidden dim
    hidden_dim: int = 4096     # SAE feature count (2x expansion is standard)
    l1_weight: float = 1e-3    # sparsity penalty
    lr: float = 1e-3
    batch_size: int = 256
    max_steps: int = 5000


class SparseAutoencoder(nn.Module):
    """Simple top-k sparse autoencoder for mechanistic interpretability."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder: input → sparse features
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        # Decoder: sparse features → reconstruction
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=False)

        # Tie decoder weights to encoder (transpose)
        self.decoder.weight = nn.Parameter(self.encoder.weight.t().clone())

        # Pre-encoder bias (subtract mean activation)
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            reconstruction: reconstructed input
            features: sparse feature activations
            loss: reconstruction + sparsity loss
        """
        # Center input
        centered = x - self.pre_bias

        # Encode → ReLU → sparse features
        features = F.relu(self.encoder(centered))

        # Decode
        reconstruction = self.decoder(features) + self.pre_bias

        # Losses
        recon_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = features.abs().mean()
        loss = recon_loss + self.config.l1_weight * sparsity_loss

        return reconstruction, features, loss


def train_sae_on_activations(
    activation_path: Path,
    shape_path: Path,
    config: SAEConfig | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SparseAutoencoder:
    """Train an SAE on cached activation memmap files.

    Args:
        activation_path: path to .memmap file
        shape_path: path to .shape.npy file
        config: SAE hyperparameters
    """
    config = config or SAEConfig()

    # Load shape and create memmap reader
    shape = tuple(np.load(shape_path))
    data = np.memmap(activation_path, dtype="float32", mode="r", shape=shape)
    config.input_dim = shape[-1]

    print(f"  Training SAE: {shape[0]} samples, dim={config.input_dim}, "
          f"features={config.hidden_dim}")

    sae = SparseAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)

    # Set pre-bias to mean of data
    sample = torch.tensor(data[:min(10000, len(data))]).to(device)
    sae.pre_bias.data = sample.mean(dim=0)

    n_samples = shape[0]
    for step in range(config.max_steps):
        # Random batch
        indices = np.random.randint(0, n_samples, size=config.batch_size)
        batch = torch.tensor(data[indices]).to(device)

        _, features, loss = sae(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            sparsity = (features > 0).float().mean().item()
            print(f"    Step {step}: loss={loss.item():.4f}, "
                  f"sparsity={sparsity:.3f} (fraction active)")

    return sae


def compare_features(
    sae_on: SparseAutoencoder,
    sae_off: SparseAutoencoder,
    data_on: np.ndarray,
    data_off: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Compare SAE features between affect-ON and affect-OFF conditions.

    Returns dict with:
    - unique_to_on: feature indices only active in affect-ON
    - unique_to_off: feature indices only active in affect-OFF
    - modulated: features active in both but with different activation patterns
    """
    with torch.no_grad():
        sample_on = torch.tensor(data_on[:1000]).to(device)
        sample_off = torch.tensor(data_off[:1000]).to(device)

        _, feats_on, _ = sae_on(sample_on)
        _, feats_off, _ = sae_off(sample_off)

        # Mean activation per feature
        mean_on = feats_on.mean(dim=0).cpu()
        mean_off = feats_off.mean(dim=0).cpu()

        # Active threshold
        threshold = 0.01
        active_on = mean_on > threshold
        active_off = mean_off > threshold

        unique_on = (active_on & ~active_off).nonzero(as_tuple=True)[0].tolist()
        unique_off = (~active_on & active_off).nonzero(as_tuple=True)[0].tolist()

        # Modulated: active in both but different magnitude
        both_active = active_on & active_off
        if both_active.any():
            ratio = mean_on[both_active] / mean_off[both_active].clamp(min=1e-6)
            modulated_mask = (ratio > 2.0) | (ratio < 0.5)
            modulated_indices = both_active.nonzero(as_tuple=True)[0]
            modulated = modulated_indices[modulated_mask].tolist()
        else:
            modulated = []

    return {
        "unique_to_affect_on": unique_on,
        "unique_to_affect_off": unique_off,
        "modulated_by_affect": modulated,
        "total_on": int(active_on.sum()),
        "total_off": int(active_off.sum()),
    }
