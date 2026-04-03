"""Run ablation study on a trained affect channel.

Tests 5 ablation modes × 5 failure categories + pattern matching baseline.
Produces selectivity ratios showing whether the affect channel provides
targeted benefit for specific failure modes.

Usage:
    uv run scripts/run_ablation.py --checkpoint results/affect_step_1000.pt
"""

import sys
import argparse
sys.path.insert(0, ".")

from src.ablation.harness import AblationMode, ablation_mode
from src.ablation.metrics import AblationReport, CategoryResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to affect checkpoint")
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    args = parser.parse_args()

    print("=" * 60)
    print("Chain of Feelings — Ablation Study")
    print("=" * 60)

    # This script requires a trained model + affect checkpoint.
    # The full implementation loads the model, injects affect, loads the checkpoint,
    # runs all prompts through each ablation mode, and computes selectivity.
    #
    # Skeleton:
    #
    # 1. Load model + affect channel + checkpoint
    # 2. Load failure case prompts
    # 3. For each ablation mode:
    #    a. Run all prompts with affect ON → collect scores per category
    #    b. Run all prompts with affect OFF (ablated) → collect scores
    #    c. Compute selectivity ratio
    # 4. Format and print results table

    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Model: {args.model}")
    print(f"\n  [This script requires a trained checkpoint to run.]")
    print(f"  [Complete Phase 4 training first, then re-run.]")

    # Demo output format
    print(f"\n--- Expected Output Format ---")
    demo = AblationReport(
        mode="zero",
        categories={
            "sycophancy": CategoryResult("sycophancy", 0.82, 0.45, 0.40),
            "overthinking": CategoryResult("overthinking", 0.71, 0.60, 0.55),
            "delayed_consequences": CategoryResult("delayed_consequences", 0.68, 0.42, 0.35),
            "reward_hacking": CategoryResult("reward_hacking", 0.75, 0.50, 0.48),
            "calibration": CategoryResult("calibration", 0.65, 0.55, 0.50),
        },
        pattern_affect_on=0.90,
        pattern_affect_off=0.88,
        pattern_baseline=0.89,
    )
    print(demo.format_table())

    return 0


if __name__ == "__main__":
    sys.exit(main())
