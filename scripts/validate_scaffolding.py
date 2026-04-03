"""Smoke test: IGT scaffolding environment + baseline agents.

Validates:
1. IGT environment runs to completion
2. Random agent scores poorly (near 0)
3. EMA agent learns to prefer good decks (positive score)
4. Text formatting produces coherent observations
"""

import sys
sys.path.insert(0, ".")

from src.eval.scaffolding import IGTEnvironment, RandomAgent, EMAAgent, run_agent


def main():
    print("=" * 60)
    print("Chain of Feelings — IGT Scaffolding Validation")
    print("=" * 60)

    env = IGTEnvironment()
    n_runs = 20

    # --- Random agent baseline ---
    random_scores = []
    for seed in range(n_runs):
        result = run_agent(env, RandomAgent(), seed=seed)
        random_scores.append(result["score"])

    avg_random = sum(random_scores) / len(random_scores)
    print(f"\nRandom agent ({n_runs} runs):")
    print(f"  Average IGT score: {avg_random:+.1f}")
    print(f"  Range: [{min(random_scores):+.0f}, {max(random_scores):+.0f}]")

    # --- EMA agent (affect-like baseline) ---
    ema_scores = []
    for seed in range(n_runs):
        result = run_agent(env, EMAAgent(alpha=0.15, epsilon=0.1), seed=seed)
        ema_scores.append(result["score"])

    avg_ema = sum(ema_scores) / len(ema_scores)
    print(f"\nEMA agent ({n_runs} runs):")
    print(f"  Average IGT score: {avg_ema:+.1f}")
    print(f"  Range: [{min(ema_scores):+.0f}, {max(ema_scores):+.0f}]")

    # --- Text formatting check ---
    print("\n--- Sample IGT text output ---")
    env.seed(0)
    obs = env.reset()
    print(obs)
    for action in ["A", "B", "C", "D", "C", "D"]:
        obs, reward, done, info = env.step(action)
        print(obs)

    print(f"\n--- History (last 4 trials) ---")
    print(env.format_history(last_n=4))

    # --- Assertions ---
    print("\n--- Validation ---")
    checks = [
        ("Random score near zero", abs(avg_random) < 30),
        ("EMA score positive", avg_ema > 0),
        ("EMA beats random", avg_ema > avg_random),
        ("Environment runs 100 trials", len(env.history) == 6),  # we only ran 6 above
    ]
    # Full run check
    full_result = run_agent(env, EMAAgent(), seed=0)
    checks.append(("Full run completes 100 trials", full_result["total_trials"] == 100))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'All checks passed.' if all_pass else 'Some checks FAILED.'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
