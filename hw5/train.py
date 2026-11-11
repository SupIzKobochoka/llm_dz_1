import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from prepare import MODEL, REVISION
from utils import reward_fn

EXPERIMENTS = {
    "rl1": {"mode": "exact", "lr": 1e-5},
    "rl2": {"mode": "shaped", "lr": 1e-5},
    "rl3": {"mode": "shaped", "lr": 3e-6},
}


def export_full_model(adapter_dir: Path, final_dir: Path, tokenizer) -> None:
    print("Merging LoRA into text model...", flush=True)
    text_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    text_model = PeftModel.from_pretrained(
        text_model, str(adapter_dir)
    ).merge_and_unload()

    print("Loading original complete Qwen3.5 model...", flush=True)
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    source = text_model.state_dict()
    destination = full_model.state_dict()
    replacement = {}
    missing = []

    for name, value in source.items():
        candidates = [
            name,
            "language_model." + name,
            "model.language_model." + name,
            "model.language_model." + name.removeprefix("model."),
            "language_model." + name.removeprefix("model."),
            "model." + name,
        ]
        key = next(
            (
                k
                for k in dict.fromkeys(candidates)
                if k in destination and destination[k].shape == value.shape
            ),
            None,
        )

        if key is None:
            missing.append(name)
        else:
            replacement[key] = value

    if missing:
        raise RuntimeError(
            f"Cannot map {len(missing)} text model tensors to full model. "
            f"Examples: {missing[:8]}; full keys: {list(destination)[:8]}"
        )

    full_model.load_state_dict(replacement, strict=False)

    del destination, source, replacement, text_model
    gc.collect()

    print("Saving full merged model...", flush=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    full_model.save_pretrained(
        final_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(final_dir)

    config = json.loads((final_dir / "config.json").read_text())

    if "vision_config" not in config:
        raise RuntimeError("Export failed: checkpoint has no vision_config")

    if config.get("model_type") != "qwen3_5":
        raise RuntimeError(
            f"Unexpected model_type: {config.get('model_type')}"
        )

    print("Saved full checkpoint:", final_dir, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")

    exp = EXPERIMENTS[args.experiment]
    run_dir = Path("runs") / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "algorithm": "GRPO",
                "seed": 42,
                "steps": args.steps,
                "lr": exp["lr"],
                "reward": exp["mode"],
                "lora_rank": 8,
                "num_generations": 2,
                "max_completion_length": 192,
            },
            indent=2,
        )
        + "\n"
    )

    if not Path("data/train.jsonl").exists():
        raise FileNotFoundError(
            "data/train.jsonl not found; run python prepare.py first"
        )

    data = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl"},
        split="train",
    )

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    config = GRPOConfig(
        output_dir=str(run_dir / "trainer"),
        max_steps=args.steps,
        learning_rate=exp["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_completion_length=192,
        temperature=0.9,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        seed=42,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=data,
        processing_class=tok,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        reward_funcs=lambda completions, target, **kwargs: reward_fn(
            completions, target, exp["mode"], **kwargs
        ),
    )

    trainer.train()

    (run_dir / "log_history.json").write_text(
        json.dumps(
            trainer.state.log_history,
            indent=2,
            default=str,
        )
        + "\n"
    )

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    del trainer, model, data
    gc.collect()
    torch.cuda.empty_cache()

    export_full_model(
        adapter_dir,
        Path("checkpoints") / args.experiment,
        tok,
    )


if __name__ == "__main__":
    main()