# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chain of Feelings: an explicit affect channel for transformer decision-making. Tests whether giving Gemma 3n a persistent "gut feeling" channel improves decisions in cases where reasoning models fail (sycophancy, overthinking, delayed consequences, reward hacking, miscalibration).

## Architecture

- **Variant A (PLE injection):** Forward hooks on Gemma 3n E2B inject a GRU-based affect state (16 dims) that modulates the residual stream via FiLM conditioning (rank-4). No model code fork — hooks compose with QLoRA.
- **Three-part loss:** task performance + affect bottleneck/stability regularisation + counterfactual somatic marker (contrastive loss on outcome-paired scenarios).
- **Ablation harness:** Context-manager that zeros/clamps/noises/shuffles the affect state to measure selective vs uniform collapse.

## Commands

```bash
# Install dependencies
uv sync

# Phase 1: Validate IGT scaffolding
uv run scripts/validate_scaffolding.py

# Phase 1: Build evaluation set
uv run scripts/build_eval_set.py

# Phase 2: Validate affect injection (needs GPU)
uv run scripts/validate_surgery.py

# Phase 4: Training (needs A100)
uv run scripts/run_training.py

# Phase 3: Ablation study
uv run scripts/run_ablation.py

# Phase 5: Interpretability
uv run scripts/run_interp.py
```

## Key Design Decisions

- **Forward hooks, not model fork:** Attach to `Gemma3nDecoderLayer` via `register_forward_hook`. Survives HF transformers updates, composable with PEFT.
- **GRU for affect state:** Ensures temporal persistence across tokens. MLP would compute a_t independently per token, defeating the "persistent channel" concept.
- **Rank-4 FiLM bottleneck:** Forces the affect channel to encode ~4 independent modulation directions, preventing it from becoming a general bypass.
- **Evaluation on reasoning-model failures:** IGT is scaffolding only. The real test is 5 categories of prompts where chain-of-thought makes things worse.

## Project Structure

- `src/eval/` — IGT scaffolding + failure case loader + baselines + metrics
- `src/affect/` — AffectChannel, AffectFiLM, PLE injection hooks
- `src/ablation/` — Ablation context manager + selectivity metrics
- `src/training/` — 3-part loss, training loop, scenario pair data
- `src/interp/` — Activation collection, SAE analysis, circuit tracing
- `scripts/` — Entry points for each phase
- `data/failure_cases/` — Curated JSONL prompts across 5 failure categories
