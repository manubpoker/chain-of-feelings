"""Training loop for Chain of Feelings.

Single-GPU training with:
- AdamW optimizer with separate LR groups (affect channel vs LoRA)
- TSV logging following evolver/Evolution pattern
- Checkpoint saving
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.affect.injection import AffectInjector
from src.training.loss import LossConfig, LossResult, compute_total_loss


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Learning rates
    affect_lr: float = 1e-3
    lora_lr: float = 1e-4

    # Training
    max_steps: int = 1000
    batch_size: int = 1
    max_seq_len: int = 512
    gradient_accumulation: int = 4

    # Loss weights
    loss: LossConfig = None

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    results_dir: str = "results"

    def __post_init__(self):
        if self.loss is None:
            self.loss = LossConfig()


@dataclass
class StepLog:
    """One row of the training log."""
    step: int
    task_loss: float
    bn_loss: float
    stab_loss: float
    som_loss: float
    total_loss: float
    a_t_norm: float
    a_t_delta: float
    lr_affect: float
    lr_lora: float
    step_time: float


class Trainer:
    """Training loop orchestrator."""

    def __init__(
        self,
        model: nn.Module,
        injector: AffectInjector,
        tokenizer,
        config: TrainConfig,
        lora_params: list[nn.Parameter] | None = None,
    ):
        self.model = model
        self.injector = injector
        self.tokenizer = tokenizer
        self.config = config

        # Build optimizer with separate param groups
        param_groups = [
            {"params": injector.trainable_parameters(), "lr": config.affect_lr},
        ]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": config.lora_lr})

        self.optimizer = AdamW(param_groups)

        # Logging
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "training.tsv"
        self._init_log()

    def _init_log(self):
        """Initialise TSV log file with header."""
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "step", "task_loss", "bn_loss", "stab_loss", "som_loss",
                "total_loss", "a_t_norm", "a_t_delta", "lr_affect", "lr_lora",
                "step_time",
            ])

    def _log_step(self, log: StepLog):
        """Append one row to the TSV log."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                log.step,
                f"{log.task_loss:.6f}",
                f"{log.bn_loss:.6f}",
                f"{log.stab_loss:.6f}",
                f"{log.som_loss:.6f}",
                f"{log.total_loss:.6f}",
                f"{log.a_t_norm:.6f}",
                f"{log.a_t_delta:.6f}",
                f"{log.lr_affect:.2e}",
                f"{log.lr_lora:.2e}",
                f"{log.step_time:.2f}",
            ])

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        paired_affects: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> LossResult:
        """Single training step."""
        self.model.train()
        self.injector.reset(
            batch_size=input_ids.shape[0],
            device=input_ids.device,
        )

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Compute loss
        loss_result = compute_total_loss(
            logits=logits,
            labels=labels,
            injector=self.injector,
            config=self.config.loss,
            paired_affects=paired_affects,
        )

        # Backward + step (with gradient accumulation)
        (loss_result.total / self.config.gradient_accumulation).backward()

        return loss_result

    def get_affect_stats(self) -> tuple[float, float]:
        """Get current affect state statistics (norm, delta)."""
        history = self.injector.channel.history
        if not history:
            return 0.0, 0.0

        states = torch.stack(history, dim=0)
        norm = states.norm(dim=-1).mean().item()
        if len(history) > 1:
            diffs = states[1:] - states[:-1]
            delta = diffs.norm(dim=-1).mean().item()
        else:
            delta = 0.0
        return norm, delta

    def save_checkpoint(self, step: int):
        """Save affect channel and FiLM weights."""
        ckpt_path = self.results_dir / f"affect_step_{step}.pt"
        torch.save({
            "step": step,
            "channel_state_dict": self.injector.channel.state_dict(),
            "film_state_dict": self.injector.film.state_dict(),
            "config": asdict(self.injector.config),
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, path: str | Path):
        """Load affect channel weights from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.injector.channel.load_state_dict(ckpt["channel_state_dict"])
        self.injector.film.load_state_dict(ckpt["film_state_dict"])
        print(f"  Loaded checkpoint from step {ckpt['step']}")
        return ckpt["step"]
