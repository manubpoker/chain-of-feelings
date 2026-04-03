"""PLE injection: attach affect channel to Gemma 3n via forward hooks.

No model code fork — hooks compose with QLoRA and can be toggled at runtime.
Attaches to each Gemma3nDecoderLayer:
  - At the readout layer: feeds residual stream to AffectChannel GRU
  - At ALL layers: applies FiLM modulation (gamma * h + beta)

The hooks fire after each layer's forward pass completes (including PLE).
"""

from __future__ import annotations

from typing import Any
from contextlib import contextmanager

import torch
import torch.nn as nn

from .module import AffectChannel, AffectConfig
from .film import AffectFiLM


class AffectInjector:
    """Manages affect channel injection into a Gemma 3n model via hooks."""

    def __init__(self, config: AffectConfig):
        self.config = config
        self.channel = AffectChannel(config)
        self.film = AffectFiLM(config)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._active = True
        self._film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        # Token position counter for sequential GRU processing
        self._current_layer = 0

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All trainable parameters (affect channel + FiLM)."""
        params = list(self.channel.parameters()) + list(self.film.parameters())
        return params

    def param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def to(self, device: torch.device) -> "AffectInjector":
        """Move affect modules to device."""
        self.channel = self.channel.to(device)
        self.film = self.film.to(device)
        return self

    def inject(self, model: nn.Module) -> None:
        """Register forward hooks on all decoder layers.

        Expects model.model.layers to be the list of transformer layers
        (standard HuggingFace Gemma3n structure).
        """
        self.remove()  # clean up any existing hooks

        layers = _get_decoder_layers(model)
        if len(layers) != self.config.num_layers:
            print(f"  Warning: expected {self.config.num_layers} layers, "
                  f"found {len(layers)}. Adjusting.")
            self.config.num_layers = len(layers)

        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._make_hook(layer_idx),
            )
            self._hooks.append(hook)

        print(f"  Injected affect channel into {len(layers)} layers")
        print(f"  Readout layer: {self.config.readout_layer}")
        print(f"  Affect dim: {self.config.affect_dim}")
        print(f"  FiLM rank: {self.config.film_rank}")
        print(f"  Trainable params: {self.param_count():,}")

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset affect state for a new sequence."""
        self.channel.reset(batch_size, device)
        self._film_params = None
        self._current_layer = 0

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""

        def hook_fn(module: nn.Module, input: Any, output: Any) -> Any:
            if not self._active:
                return output

            # Extract hidden states from output
            # Gemma 3n decoder layer outputs a 4D tensor: (altup, batch, seq, dim)
            # where altup is the AltUp width (typically 4 for E2B).
            # Standard models output (batch, seq, dim) or tuple((batch, seq, dim), ...).
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Handle Gemma 3n's 4D AltUp output: (altup, batch, seq, dim)
            # We operate on the first AltUp slot (the "main" prediction)
            is_4d = hidden_states.dim() == 4 and not isinstance(output, tuple)
            if is_4d:
                # Work with the first AltUp slot for affect readout/modulation
                main_hidden = hidden_states[0]  # (batch, seq, dim)
            else:
                main_hidden = hidden_states

            # Ensure 3D: (batch, seq, dim)
            if main_hidden.dim() == 2:
                main_hidden = main_hidden.unsqueeze(0)

            # At readout layer: update affect state from residual stream
            if layer_idx == self.config.readout_layer:
                affect_states = self.channel(main_hidden.float())
                mean_affect = affect_states.mean(dim=1)  # (batch, affect_dim)
                self._film_params = self.film(mean_affect)

            # Apply FiLM modulation at all layers (after readout has been computed)
            # All operations must be out-of-place to preserve autograd graph
            if self._film_params is not None and layer_idx < len(self._film_params):
                gamma, beta = self._film_params[layer_idx]
                # Cast to match hidden states dtype (model runs in bfloat16)
                gamma = gamma.to(dtype=hidden_states.dtype).unsqueeze(1)  # (batch, 1, model_dim)
                beta = beta.to(dtype=hidden_states.dtype).unsqueeze(1)

                if is_4d:
                    # Modulate all AltUp slots (out-of-place)
                    hidden_states = torch.stack([
                        gamma * hidden_states[i] + beta
                        for i in range(hidden_states.shape[0])
                    ])
                else:
                    hidden_states = gamma * main_hidden + beta

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook_fn


def _get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Extract the list of decoder layers from a HuggingFace model.

    Tries common attribute paths for different model architectures.
    """
    # Gemma 3n: model.model.language_model.layers
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return list(model.model.language_model.layers)

    # Standard Gemma/Llama: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)

    # Generic: model.layers
    if hasattr(model, "layers"):
        return list(model.layers)

    # Fallback: search for the largest nn.ModuleList with "layer" in its path
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            candidates.append((name, module))
    if candidates:
        # Prefer paths containing "layer" in the name
        for name, module in candidates:
            if "layer" in name.lower():
                return list(module)
        return list(candidates[0][1])

    raise ValueError(
        "Could not find decoder layers. Expected model.model.language_model.layers "
        "or similar. Check the model architecture."
    )


# ---------------------------------------------------------------------------
# Convenience: load model + inject affect in one call
# ---------------------------------------------------------------------------

def setup_affective_model(
    model_name: str = "google/gemma-3n-E2B-it",
    config: AffectConfig | None = None,
    load_in_4bit: bool = True,
    device: str = "auto",
) -> tuple[nn.Module, AffectInjector]:
    """Load a Gemma 3n model and inject the affect channel.

    Returns (model, injector) — injector holds trainable affect params.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = config or AffectConfig()

    print(f"Loading {model_name}...")
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "altup", "prediction_coefs", "altup_projections",
                "altup_unembed_projections", "per_layer_projection",
                "lm_head", "embed_tokens",
            ],
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detect layer count and hidden size
    layers = _get_decoder_layers(model)
    config.num_layers = len(layers)
    # Gemma 3n nests text config under text_config
    if hasattr(model.config, "text_config"):
        config.model_dim = model.config.text_config.hidden_size
    else:
        config.model_dim = model.config.hidden_size
    print(f"  Detected {config.num_layers} layers, dim={config.model_dim}")

    # Inject affect channel
    injector = AffectInjector(config)
    injector.to(next(model.parameters()).device)
    injector.inject(model)

    return model, injector, tokenizer
