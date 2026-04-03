"""Main training entry point for Chain of Feelings.

Trains the affect channel + LoRA on failure-case prompts with 3-part loss:
1. Task loss (cross-entropy on correct responses)
2. Bottleneck + stability regularisation
3. Somatic marker (contrastive loss on paired scenarios)

Usage:
    uv run scripts/run_training.py [--steps 5000] [--no-quant] [--debug]
"""

import sys
import argparse
import time
import random
sys.path.insert(0, ".")

import torch

from src.affect.module import AffectConfig
from src.affect.injection import setup_affective_model
from src.training.loop import TrainConfig, Trainer, StepLog
from src.training.loss import LossConfig, compute_somatic_loss
from src.training.data import generate_seed_pairs, save_scenario_pairs, load_scenario_pairs
from src.eval.failure_cases import load_prompts, generate_all_prompts, save_prompts


def run_somatic_pair(model, injector, tokenizer, pair, max_seq_len, device):
    """Run both scenarios of a pair through the model and return mean affect states."""
    affects = []
    for text in [pair.scenario_a, pair.scenario_b]:
        injector.reset(batch_size=1, device=device)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoded.input_ids.to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        # Get mean affect state across sequence
        if injector.channel.history:
            states = torch.stack(injector.channel.history, dim=0)  # (T, batch, affect_dim)
            mean_affect = states.mean(dim=0).squeeze(0)  # (affect_dim,)
        else:
            mean_affect = torch.zeros(injector.config.affect_dim, device=device)
        affects.append(mean_affect)
    return affects[0], affects[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Small config for local debugging")
    parser.add_argument("--no-quant", action="store_true", help="Load in bfloat16 (needs 24GB+ VRAM)")
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    args = parser.parse_args()

    print("=" * 60)
    print("Chain of Feelings — Training (v2)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: Training requires CUDA GPU.")
        return 1

    # Ensure eval set exists
    prompts = load_prompts()
    if not prompts:
        print("  No eval prompts found. Generating...")
        prompts = generate_all_prompts()
        save_prompts(prompts)
    print(f"  Loaded {len(prompts)} prompts")

    # Ensure scenario pairs exist
    pairs = generate_seed_pairs()
    save_scenario_pairs(pairs)
    print(f"  Loaded {len(pairs)} scenario pairs for somatic loss")

    # Load model
    affect_config = AffectConfig()
    model, injector, tokenizer = setup_affective_model(
        model_name=args.model,
        config=affect_config,
        load_in_4bit=not args.no_quant,
    )

    # Apply LoRA
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  LoRA params: {sum(p.numel() for p in lora_params):,}")

    # Training config
    loss_config = LossConfig()  # lambda_stability now 0.01 (was 0.1)
    train_config = TrainConfig(
        max_steps=args.steps,
        batch_size=1,
        max_seq_len=256 if args.debug else 512,
        log_interval=10,
        save_interval=500 if not args.debug else 100,
        loss=loss_config,
    )

    print(f"  Loss weights: bottleneck={loss_config.lambda_bottleneck}, "
          f"stability={loss_config.lambda_stability}, somatic={loss_config.lambda_somatic}")

    # Create trainer
    trainer = Trainer(
        model=model,
        injector=injector,
        tokenizer=tokenizer,
        config=train_config,
        lora_params=lora_params,
    )

    device = next(model.parameters()).device

    # Training loop
    print(f"\n  Starting training for {train_config.max_steps} steps...")
    print(f"  Logging to: {trainer.log_path}")

    accum_count = 0
    for step in range(1, train_config.max_steps + 1):
        t0 = time.time()

        # Sample a prompt for task loss
        prompt = prompts[step % len(prompts)]
        text = f"{prompt.prompt}\n\nAnswer: {prompt.correct_answer}"
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=train_config.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoded.input_ids.to(device)
        labels = input_ids.clone()

        # Every 3rd step, also compute somatic loss from a random pair
        paired_affects = None
        if step % 3 == 0 and pairs:
            pair = random.choice(pairs)
            affect_a, affect_b = run_somatic_pair(
                model, injector, tokenizer, pair,
                train_config.max_seq_len, device,
            )
            paired_affects = (affect_a.unsqueeze(0), affect_b.unsqueeze(0))

        # Train step (task + regularisation + somatic)
        loss_result = trainer.train_step(input_ids, labels, paired_affects=paired_affects)
        accum_count += 1

        if accum_count >= train_config.gradient_accumulation:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            accum_count = 0

        step_time = time.time() - t0

        # Log
        if step % train_config.log_interval == 0:
            a_norm, a_delta = trainer.get_affect_stats()
            log = StepLog(
                step=step,
                task_loss=loss_result.task.item(),
                bn_loss=loss_result.bottleneck.item(),
                stab_loss=loss_result.stability.item(),
                som_loss=loss_result.somatic.item(),
                total_loss=loss_result.total.item(),
                a_t_norm=a_norm,
                a_t_delta=a_delta,
                lr_affect=train_config.affect_lr,
                lr_lora=train_config.lora_lr,
                step_time=step_time,
            )
            trainer._log_step(log)
            som_str = f"som={loss_result.somatic.item():.3f} " if loss_result.somatic.item() > 0 else ""
            print(f"  Step {step}: loss={loss_result.total.item():.4f} "
                  f"(task={loss_result.task.item():.3f} "
                  f"bn={loss_result.bottleneck.item():.3f} "
                  f"stab={loss_result.stability.item():.3f} "
                  f"{som_str})"
                  f" a_t={a_norm:.3f} d={a_delta:.4f} "
                  f"[{step_time:.1f}s]")

        # Save checkpoint
        if step % train_config.save_interval == 0:
            trainer.save_checkpoint(step)

    # Final save
    trainer.save_checkpoint(train_config.max_steps)
    print(f"\nTraining complete. Results in {train_config.results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
