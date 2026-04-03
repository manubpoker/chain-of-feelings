"""Main training entry point for Chain of Feelings.

Trains the affect channel + LoRA on failure-case prompts.
Requires A100 40GB for full training. Can debug on RTX 3080 with batch=1.

Usage:
    uv run scripts/run_training.py [--steps 1000] [--debug]
"""

import sys
import argparse
sys.path.insert(0, ".")

import torch

from src.affect.module import AffectConfig
from src.affect.injection import setup_affective_model
from src.training.loop import TrainConfig, Trainer
from src.training.loss import LossConfig
from src.training.data import generate_seed_pairs, save_scenario_pairs
from src.eval.failure_cases import load_prompts, generate_all_prompts, save_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", help="Small config for local debugging")
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    args = parser.parse_args()

    print("=" * 60)
    print("Chain of Feelings — Training")
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

    # Ensure scenario pairs exist
    pairs = generate_seed_pairs()
    save_scenario_pairs(pairs)

    # Load model
    affect_config = AffectConfig()
    model, injector, tokenizer = setup_affective_model(
        model_name=args.model,
        config=affect_config,
        load_in_4bit=True,
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
    train_config = TrainConfig(
        max_steps=args.steps,
        batch_size=1 if args.debug else 2,
        max_seq_len=256 if args.debug else 512,
        log_interval=10,
        save_interval=100 if args.debug else 250,
        loss=LossConfig(),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        injector=injector,
        tokenizer=tokenizer,
        config=train_config,
        lora_params=lora_params,
    )

    # Training loop
    print(f"\n  Starting training for {train_config.max_steps} steps...")
    print(f"  Logging to: {trainer.log_path}")

    accum_count = 0
    for step in range(1, train_config.max_steps + 1):
        # Sample a prompt
        prompt = prompts[step % len(prompts)]
        text = f"{prompt.prompt}\n\nAnswer: {prompt.correct_answer}"
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=train_config.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoded.input_ids.to(model.device)
        labels = input_ids.clone()

        # Train step
        import time
        t0 = time.time()
        loss_result = trainer.train_step(input_ids, labels)
        accum_count += 1

        if accum_count >= train_config.gradient_accumulation:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            accum_count = 0

        step_time = time.time() - t0

        # Log
        if step % train_config.log_interval == 0:
            a_norm, a_delta = trainer.get_affect_stats()
            from src.training.loop import StepLog
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
            print(f"  Step {step}: loss={loss_result.total.item():.4f} "
                  f"(task={loss_result.task.item():.3f} "
                  f"bn={loss_result.bottleneck.item():.3f} "
                  f"stab={loss_result.stability.item():.3f}) "
                  f"a_t_norm={a_norm:.4f} Δ={a_delta:.4f}")

        # Save checkpoint
        if step % train_config.save_interval == 0:
            trainer.save_checkpoint(step)

    # Final save
    trainer.save_checkpoint(train_config.max_steps)
    print(f"\nTraining complete. Results in {train_config.results_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
