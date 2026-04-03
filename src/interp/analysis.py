"""Circuit tracing and dimension mapping for the affect channel.

Maps the 16 affect dimensions to failure categories and specific behaviours.
Identifies which attention heads and MLP neurons are downstream of FiLM modulation.
Supports steering experiments: clamp individual dimensions and observe changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.affect.injection import AffectInjector
from src.eval.metrics import FailureCategory


@dataclass
class DimensionProfile:
    """Profile of a single affect dimension's behaviour."""
    dim_idx: int
    mean_activation: float
    std_activation: float
    category_correlations: dict[str, float]  # correlation with each failure category
    top_category: str                         # most correlated category
    steering_effect: dict[str, float] | None = None  # effect of clamping this dim


def map_dimensions_to_categories(
    affect_states: dict[str, list[torch.Tensor]],
) -> list[DimensionProfile]:
    """Map each affect dimension to failure categories based on activation patterns.

    Args:
        affect_states: dict mapping category name → list of affect state tensors
                       collected during evaluation on that category's prompts.

    Returns:
        List of DimensionProfile, one per affect dimension.
    """
    # Stack all states and compute per-dim statistics
    all_states = []
    all_labels = []
    categories = list(affect_states.keys())

    for cat_idx, (cat, states) in enumerate(affect_states.items()):
        for s in states:
            if s.dim() > 1:
                s = s.mean(dim=0)  # average over sequence
            if s.dim() > 1:
                s = s.mean(dim=0)  # average over batch
            all_states.append(s)
            all_labels.append(cat_idx)

    if not all_states:
        return []

    states_tensor = torch.stack(all_states)  # (n_samples, affect_dim)
    labels_tensor = torch.tensor(all_labels)
    affect_dim = states_tensor.shape[1]

    profiles = []
    for d in range(affect_dim):
        dim_values = states_tensor[:, d]

        # Correlation with each category (point-biserial approximation)
        correlations = {}
        for cat_idx, cat in enumerate(categories):
            mask = (labels_tensor == cat_idx).float()
            if mask.sum() < 2 or (1 - mask).sum() < 2:
                correlations[cat] = 0.0
                continue
            # Mean activation when this category is present vs absent
            mean_in = dim_values[mask.bool()].mean().item()
            mean_out = dim_values[~mask.bool()].mean().item()
            # Normalise by overall std
            std = dim_values.std().item()
            if std > 1e-6:
                correlations[cat] = (mean_in - mean_out) / std
            else:
                correlations[cat] = 0.0

        top_cat = max(correlations, key=lambda k: abs(correlations[k]))

        profiles.append(DimensionProfile(
            dim_idx=d,
            mean_activation=dim_values.mean().item(),
            std_activation=dim_values.std().item(),
            category_correlations=correlations,
            top_category=top_cat,
        ))

    return profiles


def steering_experiment(
    model,
    injector: AffectInjector,
    tokenizer,
    prompt: str,
    dim_idx: int,
    values: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
) -> dict[float, str]:
    """Clamp a single affect dimension to different values and observe output changes.

    Returns dict mapping clamped value → model response.
    """
    results = {}

    for val in values:
        injector.reset(batch_size=1, device=next(model.parameters()).device)

        # Hook to clamp the specific dimension after each GRU update
        original_forward = injector.channel.forward

        def clamped_forward(residual, _val=val, _dim=dim_idx):
            result = original_forward(residual)
            # Clamp the target dimension
            result[:, :, _dim] = _val
            return result

        injector.channel.forward = clamped_forward

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results[val] = response

        # Restore original forward
        injector.channel.forward = original_forward

    return results


def format_dimension_report(profiles: list[DimensionProfile]) -> str:
    """Format dimension mapping as a readable report."""
    lines = [
        "Affect Dimension Mapping",
        "=" * 70,
        f"{'Dim':<5} {'Mean':>8} {'Std':>8} {'Top Category':<25} {'Correlation':>12}",
        "-" * 70,
    ]

    for p in profiles:
        top_corr = p.category_correlations.get(p.top_category, 0.0)
        lines.append(
            f"{p.dim_idx:<5} {p.mean_activation:>8.4f} {p.std_activation:>8.4f} "
            f"{p.top_category:<25} {top_corr:>+12.4f}"
        )

    lines.append("=" * 70)

    # Summary: which dims are most category-specific
    lines.append("\nMost category-specific dimensions:")
    ranked = sorted(profiles, key=lambda p: max(
        abs(v) for v in p.category_correlations.values()
    ), reverse=True)
    for p in ranked[:5]:
        top_corr = p.category_correlations[p.top_category]
        lines.append(f"  Dim {p.dim_idx}: {p.top_category} (r={top_corr:+.3f})")

    return "\n".join(lines)
