"""Three-part loss for Chain of Feelings training.

1. Task loss: cross-entropy on correct responses to failure-case prompts
2. Bottleneck + stability: regularise affect channel to be low-dim and slowly varying
3. Somatic marker: contrastive loss on paired scenarios (surface-similar, outcome-divergent)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.affect.injection import AffectInjector


@dataclass
class LossConfig:
    """Weights for the three loss components."""
    lambda_bottleneck: float = 0.001  # reduced — was causing fixed-point collapse
    lambda_stability: float = 0.001  # minimal — let the affect state move freely
    lambda_somatic: float = 0.05
    somatic_margin: float = 0.5    # contrastive margin


@dataclass
class LossResult:
    """Breakdown of loss components for logging."""
    total: torch.Tensor
    task: torch.Tensor
    bottleneck: torch.Tensor
    stability: torch.Tensor
    somatic: torch.Tensor


def compute_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard cross-entropy on next-token prediction.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) — -100 for positions to ignore
    """
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_somatic_loss(
    affect_a: torch.Tensor,
    affect_b: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Contrastive loss on paired scenarios.

    Forces the affect channel to produce different states for surface-similar
    scenarios with different long-term desirability.

    Args:
        affect_a: (batch, affect_dim) — mean affect state for scenario A
        affect_b: (batch, affect_dim) — mean affect state for scenario B
        margin: minimum cosine distance required between pairs

    Returns:
        Scalar loss: max(0, margin - cosine_distance(a, b))
    """
    # Cosine similarity → distance
    cos_sim = F.cosine_similarity(affect_a, affect_b, dim=-1)
    cos_dist = 1.0 - cos_sim  # 0 = identical, 2 = opposite

    # Hinge: penalise when pairs are too similar
    loss = F.relu(margin - cos_dist)
    return loss.mean()


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    injector: AffectInjector,
    config: LossConfig,
    paired_affects: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> LossResult:
    """Compute the full 3-part loss.

    Args:
        logits: model output logits
        labels: target token ids
        injector: the affect injector (for regularisation terms)
        config: loss weights
        paired_affects: optional (affect_a, affect_b) for somatic marker loss

    Returns:
        LossResult with individual components and total
    """
    device = logits.device

    # 1. Task loss
    task = compute_task_loss(logits, labels)

    # 2. Regularisation
    reg = injector.channel.get_regularisation_terms()
    bottleneck = reg["bottleneck"]
    stability = reg["stability"]

    # 3. Somatic marker (if paired scenarios provided)
    if paired_affects is not None:
        somatic = compute_somatic_loss(
            paired_affects[0], paired_affects[1],
            margin=config.somatic_margin,
        )
    else:
        somatic = torch.tensor(0.0, device=device)

    total = (
        task
        + config.lambda_bottleneck * bottleneck
        + config.lambda_stability * stability
        + config.lambda_somatic * somatic
    )

    return LossResult(
        total=total,
        task=task,
        bottleneck=bottleneck,
        stability=stability,
        somatic=somatic,
    )
