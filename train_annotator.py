#!/usr/bin/env python3
"""Fine-tune an 8B model for Dafny annotation filling (Paper 2 methodology).

Paper: "dafny-annotator: AI-Assisted Dafny Program Verification" (arXiv 2411.15143)
Approach: QLoRA fine-tune Llama-3.1-8B-Instruct on DafnyBench hints_removed→ground_truth pairs.
Target: 50.6% on DafnyBench test set.

Usage:
  # Step 1: Prepare dataset
  python train_annotator.py --prepare-only --dataset wendy-sun/DafnyBench
  
  # Step 2: Train (requires GPU with ≥15GB VRAM for 8B in 4-bit)
  python train_annotator.py --train --model unsloth/Llama-3.1-8B-Instruct-bnb-4bit
  
  # Step 3: Evaluate on benchmark
  python train_annotator.py --eval --adapter /path/to/adapter
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompt — matches src/pipeline.py _ANNOTATE_SYSTEM exactly
# ---------------------------------------------------------------------------
ANNOTATE_SYSTEM = """You are a Dafny verification expert. The program below has a complete
code body but is MISSING loop invariants, decreases clauses, and proof assertions.

Your job: add the MINIMAL set of annotations (invariants, decreases, asserts) needed
for Dafny to verify the program.

RULES:
- The existing code body is CORRECT — do NOT change any executable statements.
- Only ADD: invariant clauses, decreases clauses, assert statements, ghost variables.
- Do NOT change method signatures, requires, or ensures clauses.
- Loop invariants must be: true on entry, preserved by the body, strong enough for postcondition.
- decreases must be non-negative integer that strictly decreases.
- Include bounds invariants for all array/sequence accesses.

Output ONLY the complete method/function bodies with annotations added.
Include the full original code plus your annotations. No markdown fences.
"""

ANNOTATE_USER_TEMPLATE = """Add loop invariants, decreases clauses, and proof assertions to make this Dafny program verify.

CURRENT PROGRAM (annotations removed — does NOT verify):
{hints_removed}

Add the missing annotations. Output the COMPLETE program with annotations.
"""


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(
    dataset_name: str,
    output_dir: Path,
    model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    max_seq_length: int = 2048,
):
    """Download DafnyBench, format as instruction-tuning data, filter by length.

    Uses the model's native chat template so training format matches inference exactly.
    """
    from datasets import load_dataset, get_dataset_split_names
    from transformers import AutoTokenizer

    print(f"Loading dataset: {dataset_name}")
    # DafnyBench only has "test" split — auto-detect available split
    try:
        splits = get_dataset_split_names(dataset_name)
    except Exception:
        splits = ["test"]
    print(f"Available splits: {splits}")
    split = "test" if "test" in splits else splits[0]
    ds = load_dataset(dataset_name, split=split)

    print(f"Total examples: {len(ds)}")
    print(f"Fields: {ds.column_names}")

    # Map field names — different HF repos use different names
    col_names = ds.column_names
    hint_col = None
    truth_col = None

    for candidate in ["hints_removed", "prompt", "input", "no_hints"]:
        if candidate in col_names:
            hint_col = candidate
            break
    for candidate in ["ground_truth", "completion", "output", "solution"]:
        if candidate in col_names:
            truth_col = candidate
            break

    if hint_col is None or truth_col is None:
        raise ValueError(
            f"Cannot find hints/ground_truth columns in {col_names}. "
            f"Expected: hints_removed/prompt and ground_truth/completion."
        )
    print(f"Hint column: {hint_col}, Truth column: {truth_col}")

    # Load tokenizer for the target model to get correct chat format
    print(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"Chat template type: {'Alpaca' if 'Instruction' in (tok.chat_template or '') else 'ChatML' if 'im_start' in (tok.chat_template or '') else 'Llama3' if 'start_header' in (tok.chat_template or '') else 'DeepSeek V2' if 'Assistant' in (tok.chat_template or '') else 'unknown'}")

    def format_example(example):
        hints = example[hint_col]
        truth = example[truth_col]
        user_msg = ANNOTATE_USER_TEMPLATE.format(hints_removed=hints)
        messages = [
            {"role": "system", "content": ANNOTATE_SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": truth},
        ]
        text = tok.apply_chat_template(messages, tokenize=False)
        tokens = tok(text, add_special_tokens=False)
        return {"text": text, "num_tokens": len(tokens["input_ids"])}

    print("Formatting examples...")
    ds = ds.map(format_example, remove_columns=col_names)

    lengths = ds["num_tokens"]
    import numpy as np
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Filter to fit in 2048 context
    filtered = ds.filter(lambda x: x["num_tokens"] <= max_seq_length)
    print(f"Filtered (≤{max_seq_length} tokens): {len(filtered)} examples "
          f"(dropped {len(ds) - len(filtered)})")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split 80/20 for train/eval (DafnyBench has no train split)
    split_ds = filtered.train_test_split(test_size=0.2, seed=3407)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    split_ds["train"].to_json(str(train_path))
    split_ds["test"].to_json(str(eval_path))
    print(f"Train: {len(split_ds['train'])} examples → {train_path}")
    print(f"Eval:  {len(split_ds['test'])} examples → {eval_path}")

    # Save metadata
    meta = {
        "dataset": dataset_name,
        "split_used": split,
        "hint_column": hint_col,
        "truth_column": truth_col,
        "total_examples": len(ds),
        "filtered_examples": len(filtered),
        "train_examples": len(split_ds["train"]),
        "eval_examples": len(split_ds["test"]),
        "max_seq_length": max_seq_length,
        "prompt_template": "Llama-3.1 chat (system + user + assistant)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Training (Unsloth QLoRA, matches Paper 2 parameters)
# ---------------------------------------------------------------------------
def train_model(
    model_name: str,
    train_data: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 8,
    max_seq_length: int = 2048,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
):
    """QLoRA fine-tune following Paper 2's hyperparameters."""
    import torch
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load 4-bit model
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )

    # Attach LoRA (Paper 2: r=8, alpha=8)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # Load formatted dataset
    ds = load_dataset("json", data_files=train_data, split="train")
    print(f"Training on {len(ds)} examples")

    # Training args (Paper 2: 3 epochs, lr=2e-4, batch=2 equivalent)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=10,
        logging_steps=5,
        learning_rate=learning_rate,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args,
    )

    print(f"Starting training ({num_epochs} epochs)...")
    trainer.train()

    # Save adapter
    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    adapter_dir: str,
    benchmark_dir: str,
    max_iters: int = 4,
):
    """Load fine-tuned model and run annotation benchmark."""
    import torch
    from unsloth import FastLanguageModel
    from pathlib import Path

    # Load model with adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)

    # Patch pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.local_llm import patch_pipeline_with_local_model
    from src.pipeline import run_annotate_pipeline

    adapter = patch_pipeline_with_local_model(
        model, tokenizer,
        model_type="deepseek_coder",
        max_new_tokens=1024,
        temperature=0.0,
    )

    # Run benchmark
    import json
    bench_dir = Path(benchmark_dir)
    with open(bench_dir / "benchmark.json") as f:
        problems = json.load(f)

    results = []
    for i, prob in enumerate(problems):
        hints_path = bench_dir / "problems" / prob["hints_file"]
        pid = prob["id"]
        print(f"[{i+1}/{len(problems)}] {pid[:50]} ...", end=" ", flush=True)

        try:
            import time
            start = time.time()
            ok, src, ver = run_annotate_pipeline(
                hints_path, max_iters=max_iters, llm_task="formal", verbose=False,
            )
            elapsed = time.time() - start
            status = "OK" if ok else "FAIL"
            print(f"{status} ({elapsed:.1f}s)")
            results.append({
                "id": pid, "type": prob["type"], "success": ok, "time": round(elapsed, 1),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": pid, "type": prob["type"], "success": False, "error": str(e)[:200]})

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"RESULTS: {ok}/{total} ({100*ok/total:.1f}%)")
    print(f"{'='*60}")

    for ptype in sorted(set(r["type"] for r in results)):
        t_res = [r for r in results if r["type"] == ptype]
        t_ok = sum(1 for r in t_res if r.get("success"))
        print(f"  {ptype}: {t_ok}/{len(t_res)} ({100*t_ok/len(t_res):.0f}%)")

    with open(bench_dir / "finetuned_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return ok, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune 8B model for Dafny annotation (Paper 2 methodology)"
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Download and format DafnyBench dataset, then exit"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Run QLoRA fine-tuning"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate fine-tuned model on benchmark"
    )
    parser.add_argument(
        "--dataset", default="wendy-sun/DafnyBench",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--model", default="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        help="Base model for fine-tuning (Paper 2 used Llama-3.1-8B, we recommend DeepSeek-Coder-7B)"
    )
    parser.add_argument(
        "--adapter", default="./dafny_annotator_lora",
        help="Path to save/load LoRA adapter"
    )
    parser.add_argument(
        "--benchmark-dir", default="./benchmark",
        help="Path to benchmark directory"
    )
    parser.add_argument(
        "--data-dir", default="./training_data",
        help="Directory for formatted training data"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
    )
    args = parser.parse_args()

    if args.prepare_only:
        prepare_dataset(args.dataset, Path(args.data_dir), model_name=args.model)
        print("\nDone. Now run: python train_annotator.py --train")
        return

    if args.train:
        data_path = Path(args.data_dir) / "train.jsonl"
        if not data_path.exists():
            print(f"Training data not found at {data_path}. Run --prepare-only first.")
            sys.exit(1)
        train_model(
            model_name=args.model,
            train_data=str(data_path),
            output_dir=args.adapter,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )
        print(f"\nDone. Now run: python train_annotator.py --eval --adapter {args.adapter}")
        return

    if args.eval:
        adapter_path = Path(args.adapter) / "adapter"
        if not adapter_path.exists():
            print(f"Adapter not found at {adapter_path}")
            sys.exit(1)
        evaluate_model(str(adapter_path), args.benchmark_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
