"""Activation collection for interpretability analysis.

Registers hooks on all layers to capture residual stream activations.
Saves to numpy memmap files for efficient out-of-core SAE training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.affect.injection import _get_decoder_layers


class ActivationCollector:
    """Collect residual stream activations from all layers."""

    def __init__(self, model: nn.Module, output_dir: Path):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, list[np.ndarray]] = {}

    def start(self) -> None:
        """Register collection hooks on all decoder layers."""
        self.stop()
        layers = _get_decoder_layers(self.model)

        for layer_idx, layer in enumerate(layers):
            self._activations[layer_idx] = []
            hook = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        print(f"  Collecting activations from {len(layers)} layers")

    def stop(self) -> None:
        """Remove collection hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Detach, move to CPU, convert to numpy
            self._activations[layer_idx].append(
                hidden.detach().float().cpu().numpy()
            )
        return hook_fn

    def save(self, prefix: str = "activations") -> dict[int, Path]:
        """Save collected activations to memmap files.

        Returns dict mapping layer_idx → file path.
        """
        paths = {}
        for layer_idx, acts in self._activations.items():
            if not acts:
                continue
            # Concatenate all collected activations
            all_acts = np.concatenate(acts, axis=0)  # (total_tokens, model_dim)
            if all_acts.ndim == 3:
                # Flatten batch and seq dims
                all_acts = all_acts.reshape(-1, all_acts.shape[-1])

            # Save as memmap
            fpath = self.output_dir / f"{prefix}_layer{layer_idx:02d}.memmap"
            mm = np.memmap(fpath, dtype="float32", mode="w+", shape=all_acts.shape)
            mm[:] = all_acts
            mm.flush()

            # Save shape metadata
            meta_path = self.output_dir / f"{prefix}_layer{layer_idx:02d}.shape"
            np.save(meta_path, np.array(all_acts.shape))

            paths[layer_idx] = fpath
            print(f"  Layer {layer_idx}: {all_acts.shape} → {fpath.name}")

        return paths

    def clear(self) -> None:
        """Clear collected activations from memory."""
        for acts in self._activations.values():
            acts.clear()
