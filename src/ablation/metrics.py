"""Ablation metrics: measure selective vs uniform collapse.

The affect channel earns its place if:
1. affect_on_score > baseline_score on failure categories
2. Improvement is concentrated in specific categories (selective)
3. Pattern-matching / general language tasks are NOT degraded
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.metrics import FailureCategory


@dataclass
class CategoryResult:
    """Scores for a single failure category under one ablation mode."""
    category: str
    affect_on: float
    affect_off: float
    baseline: float  # no-affect base model

    @property
    def improvement_vs_baseline(self) -> float:
        """How much the affect channel helps vs raw model."""
        return self.affect_on - self.baseline

    @property
    def degradation_under_ablation(self) -> float:
        """How much ablation hurts (positive = ablation hurts)."""
        return self.affect_on - self.affect_off


@dataclass
class AblationReport:
    """Full ablation results across all modes and categories."""
    mode: str
    categories: dict[str, CategoryResult] = field(default_factory=dict)
    pattern_affect_on: float = 0.0
    pattern_affect_off: float = 0.0
    pattern_baseline: float = 0.0

    @property
    def pattern_degradation(self) -> float:
        """How much ablation hurts pattern-matching (should be ~0)."""
        return self.pattern_affect_on - self.pattern_affect_off

    def selectivity_ratios(self) -> dict[str, float]:
        """Per-category selectivity: eval_degradation / pattern_degradation.

        High ratio = selective collapse (good — affect does something unique).
        Ratio ≈ 1.0 = uniform collapse (bad — just extra capacity).
        """
        pat_drop = max(self.pattern_degradation, 1e-6)
        return {
            cat: min(result.degradation_under_ablation / pat_drop, 10.0)
            for cat, result in self.categories.items()
        }

    def format_table(self) -> str:
        """Format results as a readable table."""
        lines = [
            f"Ablation Mode: {self.mode}",
            "-" * 80,
            f"{'Category':<25} {'Affect ON':>10} {'Affect OFF':>10} "
            f"{'Baseline':>10} {'Drop':>10} {'Selectivity':>12}",
            "-" * 80,
        ]

        ratios = self.selectivity_ratios()
        for cat, result in self.categories.items():
            ratio = ratios.get(cat, 0.0)
            lines.append(
                f"{cat:<25} {result.affect_on:>10.3f} {result.affect_off:>10.3f} "
                f"{result.baseline:>10.3f} {result.degradation_under_ablation:>+10.3f} "
                f"{ratio:>12.2f}"
            )

        lines.append("-" * 80)
        lines.append(
            f"{'pattern_matching':<25} {self.pattern_affect_on:>10.3f} "
            f"{self.pattern_affect_off:>10.3f} {self.pattern_baseline:>10.3f} "
            f"{self.pattern_degradation:>+10.3f} {'(reference)':>12}"
        )

        return "\n".join(lines)
