"""Build the failure case evaluation set.

Generates ~44 curated prompts across 5 categories and saves as JSONL.
This is the initial seed set — expand to ~200 by adding more cases per category.
"""

import sys
sys.path.insert(0, ".")

from src.eval.failure_cases import generate_all_prompts, save_prompts


def main():
    print("=" * 60)
    print("Chain of Feelings — Building Evaluation Set")
    print("=" * 60)
    print()

    prompts = generate_all_prompts()

    print(f"\nTotal: {len(prompts)} prompts")
    print("\nSaving to data/failure_cases/...")
    save_prompts(prompts)

    print(f"\nDone. {len(prompts)} prompts saved across 5 categories.")
    print("\nTo expand to ~200 prompts, add more cases to each generator")
    print("function in src/eval/failure_cases.py")


if __name__ == "__main__":
    main()
