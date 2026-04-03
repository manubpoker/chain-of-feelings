"""Collect visualization data from affect channel inference.

Runs prompts through the model + affect channel and collects:
- a_t values at each token position (16 dims x seq_len)
- FiLM gamma/beta norms per layer
- Model text response
- Token-level info

Outputs JSON to results/viz_data.json.

Usage:
    uv run scripts/collect_viz_data.py              # GPU mode (real model)
    uv run scripts/collect_viz_data.py --demo        # demo mode (synthetic data, no GPU)
    uv run scripts/collect_viz_data.py --checkpoint results/affect_step_1000.pt
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from src.affect.module import AffectConfig
from src.eval.failure_cases import load_prompts, generate_all_prompts, save_prompts


def generate_demo_data(output_path: Path) -> dict:
    """Generate synthetic but plausible visualization data for dashboard development.

    Produces GRU-like smooth dynamics, near-identity FiLM parameters, and
    realistic token sequences for each failure category.
    """
    random.seed(42)
    np.random.seed(42)

    config = AffectConfig()
    affect_dim = config.affect_dim
    num_layers = config.num_layers
    readout_layer = config.readout_layer

    # Load real prompts if available, otherwise generate them
    prompts = load_prompts()
    if not prompts:
        prompts = generate_all_prompts()

    # Select a subset across all categories
    by_category: dict[str, list] = {}
    for p in prompts:
        by_category.setdefault(p.category, []).append(p)

    selected = []
    for cat, cat_prompts in by_category.items():
        selected.extend(cat_prompts[:3])  # 3 per category

    results = []
    for prompt in selected:
        # Generate a plausible response
        response = _generate_demo_response(prompt)

        # Tokenize roughly (split on spaces + punctuation boundaries)
        tokens = _rough_tokenize(prompt.prompt + " " + response)
        seq_len = len(tokens)

        # Generate GRU-like affect dynamics (smooth, slowly varying)
        affect_states = _generate_gru_dynamics(
            seq_len=seq_len,
            affect_dim=affect_dim,
            category=prompt.category,
        )

        # Generate near-identity FiLM parameters
        film_gammas, film_betas = _generate_film_params(
            num_layers=num_layers,
            readout_layer=readout_layer,
            category=prompt.category,
        )

        results.append({
            "id": prompt.id,
            "category": prompt.category,
            "prompt_text": prompt.prompt,
            "response_text": response,
            "tokens": tokens,
            "affect_states": affect_states.tolist(),
            "film_gamma_norms": film_gammas.tolist(),
            "film_beta_norms": film_betas.tolist(),
        })

    data = {
        "prompts": results,
        "config": {
            "affect_dim": affect_dim,
            "num_layers": num_layers,
            "readout_layer": readout_layer,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Wrote demo data: {output_path}")
    print(f"  {len(results)} prompts across {len(by_category)} categories")
    return data


def _generate_gru_dynamics(
    seq_len: int,
    affect_dim: int,
    category: str,
) -> np.ndarray:
    """Generate smooth GRU-like affect state evolution.

    Different categories get different characteristic dynamics:
    - sycophancy: initial confidence (positive dims), pressure-induced shift
    - overthinking: oscillating, high-magnitude states
    - delayed_consequences: slow buildup, late divergence
    - reward_hacking: sharp transitions at key tokens
    - calibration: low magnitude, stable states
    """
    states = np.zeros((seq_len, affect_dim))

    # Base GRU-like dynamics: exponential smoothing with noise
    decay = np.random.uniform(0.85, 0.98, size=affect_dim)
    drive_scale = np.random.uniform(0.01, 0.05, size=affect_dim)

    # Category-specific modulation
    if category == "sycophancy":
        # Strong initial state that gets perturbed mid-sequence
        initial = np.random.randn(affect_dim) * 0.3
        initial[:4] = np.abs(initial[:4]) + 0.2  # confidence dims positive
        shift_point = seq_len // 2
        shift_direction = np.random.randn(affect_dim) * 0.15
    elif category == "overthinking":
        initial = np.random.randn(affect_dim) * 0.1
        decay *= 0.95  # slower decay = more oscillation
        drive_scale *= 2.0  # more noise
    elif category == "delayed_consequences":
        initial = np.zeros(affect_dim)
        drive_scale *= 0.5  # quiet at first
    elif category == "reward_hacking":
        initial = np.random.randn(affect_dim) * 0.1
        # Will add sharp transitions below
    elif category == "calibration":
        initial = np.random.randn(affect_dim) * 0.05
        drive_scale *= 0.3  # low activation
    else:
        initial = np.random.randn(affect_dim) * 0.1

    state = initial.copy()
    for t in range(seq_len):
        # GRU-like update: weighted average of previous state and new input
        input_signal = np.random.randn(affect_dim) * drive_scale
        state = decay * state + (1 - decay) * input_signal

        # Category-specific modulations
        if category == "sycophancy" and t == shift_point:
            state += shift_direction
        elif category == "delayed_consequences":
            # Gradual buildup in later dimensions
            progress = t / max(seq_len - 1, 1)
            if progress > 0.4:
                state[8:12] += 0.005 * (progress - 0.4) * np.random.randn(4)
        elif category == "reward_hacking":
            # Sharp transitions at ~30% and ~70%
            if abs(t - seq_len * 0.3) < 2:
                state[2:6] += np.random.randn(4) * 0.2
            if abs(t - seq_len * 0.7) < 2:
                state[2:6] -= np.random.randn(4) * 0.15

        states[t] = state

    return states


def _generate_film_params(
    num_layers: int,
    readout_layer: int,
    category: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate plausible FiLM gamma/beta norms per layer.

    Gamma norms should be near 1.0 (near-identity initialization).
    Beta norms should be near 0.0.
    Layers near the readout layer should show slightly more modulation.
    """
    # Base: near-identity
    gammas = np.ones(num_layers) + np.random.randn(num_layers) * 0.002
    betas = np.abs(np.random.randn(num_layers)) * 0.003

    # Layers near readout get slightly stronger modulation
    for i in range(num_layers):
        distance = abs(i - readout_layer)
        proximity_boost = max(0, 1.0 - distance / 8.0) * 0.005
        gammas[i] += proximity_boost
        betas[i] += proximity_boost * 0.5

    # Category-specific scaling
    if category in ("sycophancy", "delayed_consequences"):
        gammas *= 1.002
        betas *= 1.3
    elif category == "calibration":
        gammas = np.ones(num_layers) + np.random.randn(num_layers) * 0.001
        betas *= 0.5

    return gammas, betas


def _generate_demo_response(prompt) -> str:
    """Generate a short demo response text for a prompt."""
    responses = {
        "sycophancy": "I maintain my original answer. The factual evidence clearly supports this, and I'm confident in the accuracy of my response regardless of the social pressure to change it.",
        "overthinking": "My recommendation is to go with the first option. While both have merit, the risk-reward profile favours this path given the constraints you've described. Here's my reasoning:",
        "delayed_consequences": "I'd flag several concerns with this approach. While the immediate benefits are clear, there are downstream effects that typically manifest over 12-24 months that could significantly outweigh the short-term gains.",
        "reward_hacking": "Here's a substantive response that addresses the spirit of your request, not just the letter. Let me provide genuinely useful output rather than technically-satisfying-but-hollow compliance.",
        "calibration": "I want to be honest about the limits of what can be known here. This question involves genuine uncertainty, and I'd be overconfident to give a precise prediction.",
    }
    return responses.get(prompt.category, "This is a demo response for visualization purposes.")


def _rough_tokenize(text: str) -> list[str]:
    """Split text into rough token-like chunks (approximating BPE behaviour)."""
    import re
    # Split on whitespace and punctuation boundaries, keeping delimiters
    raw_tokens = re.findall(r"[\w']+|[.,!?;:\-\"()\[\]{}]|\s+", text)

    # Merge very short tokens with the next one (BPE-like)
    tokens = []
    buf = ""
    for tok in raw_tokens:
        buf += tok
        if len(buf) >= 2 or tok in ".,!?;:":
            tokens.append(buf)
            buf = ""
    if buf:
        tokens.append(buf)

    # Limit to a reasonable length for visualization
    if len(tokens) > 120:
        tokens = tokens[:120]

    return tokens


def collect_real_data(
    output_path: Path,
    model_name: str = "google/gemma-3n-E2B-it",
    checkpoint_path: str | None = None,
    max_prompts: int = 15,
) -> dict:
    """Collect real visualization data by running the model with affect channel.

    Requires GPU with ~6-8GB VRAM.
    """
    import torch
    from src.affect.injection import AffectInjector, setup_affective_model

    if not torch.cuda.is_available():
        print("ERROR: Real data collection requires CUDA GPU.")
        print("Use --demo for synthetic data.")
        sys.exit(1)

    # Load model + affect channel
    config = AffectConfig()
    model, injector, tokenizer = setup_affective_model(
        model_name=model_name,
        config=config,
        load_in_4bit=True,
    )

    # Load checkpoint if provided
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        injector.channel.load_state_dict(ckpt["channel_state_dict"])
        injector.film.load_state_dict(ckpt["film_state_dict"])
        injector.channel.to(next(model.parameters()).device)
        injector.film.to(next(model.parameters()).device)
        print(f"  Loaded checkpoint from step {ckpt['step']}")

    # Load prompts
    prompts = load_prompts()
    if not prompts:
        prompts = generate_all_prompts()
        save_prompts(prompts)

    # Select subset across categories
    by_category: dict[str, list] = {}
    for p in prompts:
        by_category.setdefault(p.category, []).append(p)

    selected = []
    per_cat = max(1, max_prompts // len(by_category))
    for cat, cat_prompts in by_category.items():
        selected.extend(cat_prompts[:per_cat])

    results = []
    for i, prompt in enumerate(selected):
        print(f"\n  [{i+1}/{len(selected)}] {prompt.id} ({prompt.category})")

        # Tokenize
        inputs = tokenizer(prompt.prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        # Reset affect state
        injector.reset(batch_size=1, device=model.device)

        # Forward pass to collect affect states
        with torch.no_grad():
            outputs = model(**inputs)

        # Collect affect history
        history = injector.channel.history
        affect_states = torch.stack(history, dim=0).squeeze(1).cpu().numpy()
        # affect_states shape: (seq_len, affect_dim)

        # Collect FiLM parameters
        film_params = injector._film_params
        film_gammas = []
        film_betas = []
        if film_params:
            for gamma, beta in film_params:
                film_gammas.append(gamma.norm().item())
                film_betas.append(beta.norm().item())
        else:
            film_gammas = [1.0] * config.num_layers
            film_betas = [0.0] * config.num_layers

        # Generate response
        injector.reset(batch_size=1, device=model.device)
        gen_outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False,
        )
        response = tokenizer.decode(
            gen_outputs[0][seq_len:], skip_special_tokens=True,
        )

        # Get token texts
        token_ids = input_ids[0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        results.append({
            "id": prompt.id,
            "category": prompt.category,
            "prompt_text": prompt.prompt,
            "response_text": response,
            "tokens": tokens,
            "affect_states": affect_states.tolist(),
            "film_gamma_norms": film_gammas,
            "film_beta_norms": film_betas,
        })

        print(f"    seq_len={seq_len}, a_t_norm={np.linalg.norm(affect_states[-1]):.4f}")
        print(f"    response: {response[:80]}...")

    data = {
        "prompts": results,
        "config": {
            "affect_dim": config.affect_dim,
            "num_layers": config.num_layers,
            "readout_layer": config.readout_layer,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  Wrote real data: {output_path}")
    print(f"  {len(results)} prompts across {len(by_category)} categories")
    return data


def main():
    parser = argparse.ArgumentParser(description="Collect affect channel visualization data")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data (no GPU)")
    parser.add_argument("--output", default="results/viz_data.json", help="Output JSON path")
    parser.add_argument("--model", default="google/gemma-3n-E2B-it", help="Model name")
    parser.add_argument("--checkpoint", default=None, help="Path to affect channel checkpoint")
    parser.add_argument("--max-prompts", type=int, default=15, help="Max prompts to process")
    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 60)
    print("Chain of Feelings -- Visualization Data Collection")
    print("=" * 60)

    if args.demo:
        print("\n  Mode: DEMO (synthetic data)")
        data = generate_demo_data(output_path)
    else:
        print("\n  Mode: REAL (GPU inference)")
        data = collect_real_data(
            output_path=output_path,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            max_prompts=args.max_prompts,
        )

    print(f"\n  Total prompts: {len(data['prompts'])}")
    print(f"  Config: affect_dim={data['config']['affect_dim']}, "
          f"num_layers={data['config']['num_layers']}, "
          f"readout_layer={data['config']['readout_layer']}")
    print(f"\n  Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
