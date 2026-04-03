"""Run interpretability pipeline on a trained affect channel.

1. Collect activations with affect ON and OFF
2. Train SAEs on each condition
3. Compare features
4. Map affect dimensions to failure categories
5. Run steering experiments

Usage:
    uv run scripts/run_interp.py --checkpoint results/affect_step_1000.pt
"""

import sys
import argparse
sys.path.insert(0, ".")

from pathlib import Path
from src.interp.analysis import format_dimension_report, DimensionProfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    parser.add_argument("--output-dir", default="results/interp")
    args = parser.parse_args()

    print("=" * 60)
    print("Chain of Feelings — Interpretability Pipeline")
    print("=" * 60)

    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")

    # This script requires a trained model + affect checkpoint.
    # Full pipeline:
    #
    # 1. Load model + affect + checkpoint
    # 2. Collect activations (affect ON and OFF) on eval set
    # 3. Train SAEs on each condition
    # 4. Compare features (unique/modulated)
    # 5. Map affect dims → categories
    # 6. Run steering experiments on key dims

    print(f"\n  [This script requires a trained checkpoint to run.]")
    print(f"  [Complete Phase 4 training first, then re-run.]")

    # Demo output format
    print(f"\n--- Expected Output Format ---")
    demo_profiles = [
        DimensionProfile(i, 0.1 * i, 0.05 * i, {
            "sycophancy": 0.3 * (1 if i == 2 else 0),
            "overthinking": 0.2 * (1 if i == 5 else 0),
            "delayed_consequences": 0.4 * (1 if i == 7 else 0),
            "reward_hacking": 0.25 * (1 if i == 11 else 0),
            "calibration": 0.35 * (1 if i == 14 else 0),
        }, "sycophancy" if i == 2 else "calibration")
        for i in range(16)
    ]
    print(format_dimension_report(demo_profiles))

    return 0


if __name__ == "__main__":
    sys.exit(main())
