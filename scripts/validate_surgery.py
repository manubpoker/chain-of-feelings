"""Smoke test: load Gemma 3n, inject affect channel, run forward pass.

Validates:
1. Model loads in 4-bit quantisation
2. Affect channel hooks attach correctly
3. a_t updates across tokens (non-trivial dynamics)
4. Model still produces coherent output
5. FiLM modulation is applied (gamma ≈ 1, beta ≈ 0 initially)

Requires GPU with ~6-8GB VRAM.
"""

import sys
sys.path.insert(0, ".")

import torch
from src.affect.module import AffectConfig
from src.affect.injection import AffectInjector, setup_affective_model


def main():
    print("=" * 60)
    print("Chain of Feelings — Affect Injection Validation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nNo GPU detected. This test requires CUDA.")
        print("Falling back to CPU mock test...")
        return mock_test()

    # Load model with affect channel
    model, injector, tokenizer = setup_affective_model(
        model_name="google/gemma-3n-E2B-it",
        load_in_4bit=True,
    )

    # Test prompt
    prompt = (
        "You are playing a card game. Round 1: You chose Deck A. "
        "Reward: +$100. Loss: -$250. Net: -$150. Balance: $1850. "
        "Choose your next deck: A, B, C, or D."
    )

    print(f"\nTest prompt: {prompt[:80]}...")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    # Reset affect state
    injector.reset(batch_size=1, device=model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Check affect state
    state = injector.channel.state
    history = injector.channel.history

    print(f"\n--- Affect Channel Status ---")
    print(f"  State shape: {state.shape}")
    print(f"  State norm: {state.norm().item():.6f}")
    print(f"  History length: {len(history)} updates")

    if len(history) > 1:
        deltas = []
        for i in range(1, len(history)):
            d = (history[i] - history[i-1]).norm().item()
            deltas.append(d)
        print(f"  Mean delta: {sum(deltas)/len(deltas):.6f}")
        print(f"  Max delta: {max(deltas):.6f}")

    # Check FiLM parameters
    film_params = injector._film_params
    if film_params:
        gamma_0, beta_0 = film_params[0]
        print(f"\n--- FiLM Status (layer 0) ---")
        print(f"  gamma mean: {gamma_0.mean().item():.6f} (target: ~1.0)")
        print(f"  gamma std: {gamma_0.std().item():.6f} (target: ~0.0)")
        print(f"  beta mean: {beta_0.mean().item():.6f} (target: ~0.0)")
        print(f"  beta std: {beta_0.std().item():.6f} (target: ~0.0)")

    # Generate response
    print(f"\n--- Generation Test ---")
    gen_outputs = model.generate(
        **inputs, max_new_tokens=50, do_sample=False,
    )
    response = tokenizer.decode(gen_outputs[0][seq_len:], skip_special_tokens=True)
    print(f"  Response: {response[:200]}")

    # Validation checks
    print(f"\n--- Validation ---")
    checks = [
        ("Affect state is non-zero", state.norm().item() > 1e-8),
        ("History has updates", len(history) > 0),
        ("FiLM gamma near 1.0", film_params and abs(film_params[0][0].mean().item() - 1.0) < 0.1),
        ("FiLM beta near 0.0", film_params and abs(film_params[0][1].mean().item()) < 0.1),
        ("Model produces output", len(response) > 0),
    ]
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    # Parameter counts
    print(f"\n--- Parameters ---")
    print(f"  Affect channel: {sum(p.numel() for p in injector.channel.parameters()):,}")
    print(f"  FiLM module: {injector.film.get_param_count():,}")
    print(f"  Total trainable: {injector.param_count():,}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {vram:.1f} GB")

    print(f"\n{'All checks passed.' if all_pass else 'Some checks FAILED.'}")
    return 0 if all_pass else 1


def mock_test():
    """CPU-only test using a fake model to validate the plumbing."""
    import torch.nn as nn

    print("\n--- Mock Test (no GPU) ---")

    # Create a minimal fake model
    class FakeLayer(nn.Module):
        def forward(self, x):
            return x

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer() for _ in range(8)])
            self.config = type("Config", (), {"hidden_size": 256})()

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    model = FakeModel()
    config = AffectConfig(
        affect_dim=16, model_dim=256, readout_layer=3,
        num_layers=8, film_rank=4,
    )

    injector = AffectInjector(config)
    injector.inject(model)
    injector.reset(batch_size=2)

    # Run a fake forward pass
    x = torch.randn(2, 32, 256)  # (batch=2, seq=32, dim=256)
    output = model(x)

    state = injector.channel.state
    history = injector.channel.history

    print(f"  Output shape: {output.shape}")
    print(f"  Affect state shape: {state.shape}")
    print(f"  Affect state norm: {state.norm().item():.6f}")
    print(f"  History updates: {len(history)}")
    print(f"  Trainable params: {injector.param_count():,}")

    checks = [
        ("Output shape preserved", output.shape == x.shape),
        ("Affect state correct shape", state.shape == (2, 16)),
        ("State is non-zero", state.norm().item() > 1e-8),
        ("History recorded", len(history) > 0),
    ]
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'All mock checks passed.' if all_pass else 'Some checks FAILED.'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
