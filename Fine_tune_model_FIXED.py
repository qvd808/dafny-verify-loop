"""
CORRECTED Colab notebook for Dafny annotation fine-tuning.
Paper 2 methodology: Llama-3.1-8B-Instruct + QLoRA on DafnyBench.

Copy each cell into your Colab notebook, replacing the original cells.

KEY FIXES vs original notebook:
  1. Model: Llama-3.1-8B-Instruct (NOT DeepSeek-R1) — matches Paper 2
  2. Chat format: Uses model's native apply_chat_template (NOT manual ### User:)
  3. Output: No R1 think tags to worry about
  4. Dataset: Correct split (train, not test)
  5. Inference: Uses src.local_llm.patch_pipeline_with_local_model
"""

# ============================================================================
# CELL 1 (replaces cell-4): Load Llama-3.1-8B-Instruct (Paper 2's model)
# ============================================================================
CELL_1_LOAD_MODEL = """
import os
import torch
from unsloth import FastLanguageModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paper 2 uses Llama-3.1-8B-Instruct — standard instruct model, NOT a reasoning model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/deepseek-coder-7b-instruct-v1.5-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)

print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
"""

# ============================================================================
# CELL 2 (replaces cell-5): Attach LoRA and train
# ============================================================================
CELL_2_TRAIN = """
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments

# Attach LoRA (Paper 2: r=8, alpha=8)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=2048,
)

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/dafny_annotator_lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    logging_steps=5,
    learning_rate=2e-4,
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
    train_dataset=filtered_ds,   # from cell-2 (the reloaded dataset)
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=training_args,
)

trainer.train()

# Save adapter
model.save_pretrained("/content/drive/MyDrive/dafny_annotator_lora/adapter")
tokenizer.save_pretrained("/content/drive/MyDrive/dafny_annotator_lora/adapter")
print("Training complete.")
"""

# ============================================================================
# CELL 3 (replaces cell-7): Patch pipeline with local model (CORRECT WAY)
# ============================================================================
CELL_3_INFERENCE = """
import sys
from pathlib import Path

# Go to repo root
REPO = "/content/dafny-verify-loop"
%cd {REPO}
if str(Path(REPO)) not in sys.path:
    sys.path.insert(0, str(Path(REPO)))

# Set model to inference mode
FastLanguageModel.for_inference(model)

# Use the proper adapter — handles chat template, output cleaning, etc.
from src.local_llm import patch_pipeline_with_local_model

adapter = patch_pipeline_with_local_model(
    model,
    tokenizer,
    model_type="deepseek_coder",  # DeepSeek-Coder Alpaca format
    max_new_tokens=1024,
    temperature=0.0,           # greedy for reproducibility
)

print("Pipeline patched. Local model will handle all LLM calls.")
"""

# ============================================================================
# CELL 4 (replaces cell-8): Test on a single problem
# ============================================================================
CELL_4_TEST = """
from pathlib import Path
from src.pipeline import run_annotate_pipeline

# Test on a simple problem first
ok, source, verifier_output = run_annotate_pipeline(
    Path("benchmark/problems/Clover_abs_no_hints.dfy"),
    max_iters=4,
    llm_task="formal",
    verbose=True,
)
print("Success:", ok)

if not ok:
    print("\\n--- Last verifier output ---")
    print(verifier_output[:500])
"""

# ============================================================================
# CELL 5 (replaces cell-9): Dataset formatting with CORRECT field names
# ============================================================================
CELL_5_DATASET = """
from datasets import load_dataset

# Load DafnyBench — use TRAIN split (not test!)
# Try both common HuggingFace repos for this dataset
for repo in ["wendy-sun/DafnyBench", "catherinemei/DafnyBench"]:
    try:
        ds = load_dataset(repo, split="train", trust_remote_code=True)
        print(f"Loaded from {repo}: {len(ds)} examples")
        print(f"Columns: {ds.column_names}")
        break
    except Exception as e:
        print(f"{repo} failed: {e}")
        continue

# Detect correct field names
cols = ds.column_names
hint_col = None
truth_col = None
for c in ["hints_removed", "prompt", "input", "no_hints"]:
    if c in cols:
        hint_col = c
        break
for c in ["ground_truth", "completion", "output", "solution"]:
    if c in cols:
        truth_col = c
        break

print(f"Using hint_col='{hint_col}', truth_col='{truth_col}'")

_ANNOTATE_SYSTEM = \"\"\"You are a Dafny verification expert. The program below has a complete
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
\"\"\"

def format_annotate_example(example):
    hints = example[hint_col]
    truth = example[truth_col]
    user_msg = f\"\"\"Add loop invariants, decreases clauses, and proof assertions to make this Dafny program verify.

CURRENT PROGRAM (annotations removed — does NOT verify):
{hints}

Add the missing annotations. Output the COMPLETE program with annotations.
\"\"\"
    # Use model's native chat template for training
    messages = [
        {"role": "system", "content": _ANNOTATE_SYSTEM},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": truth},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(text, add_special_tokens=False)
    return {"text": text, "num_tokens": len(tokens["input_ids"])}

# Format and filter
ds = ds.map(format_annotate_example, remove_columns=cols)

import numpy as np
lengths = ds["num_tokens"]
print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, "
      f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
print(f"Examples <= 2048 tokens: {sum(1 for l in lengths if l <= 2048)} out of {len(lengths)}")

filtered_ds = ds.filter(lambda x: x["num_tokens"] <= 2048)
print(f"Filtered dataset: {len(filtered_ds)} examples")

# Save
filtered_ds.to_json("/content/drive/MyDrive/filtered_ds_llama3.jsonl")
print("Saved.")
"""

# ============================================================================
# CELL 6: Run full benchmark evaluation
# ============================================================================
CELL_6_BENCHMARK = """
import json, time
from pathlib import Path
from src.pipeline import run_annotate_pipeline

bench_dir = Path("benchmark")
with open(bench_dir / "benchmark.json") as f:
    problems = json.load(f)

results = []
for i, prob in enumerate(problems):
    hints_path = bench_dir / "problems" / prob["hints_file"]
    pid = prob["id"]
    print(f"[{i+1}/{len(problems)}] {pid[:50]} ...", end=" ", flush=True)
    
    try:
        start = time.time()
        ok, src, ver = run_annotate_pipeline(
            hints_path, max_iters=4, llm_task="formal", verbose=False,
        )
        elapsed = time.time() - start
        status = "OK" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        results.append({
            "id": pid, "type": prob["type"],
            "success": ok, "time": round(elapsed, 1),
        })
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"id": pid, "type": prob["type"], "success": False, "error": str(e)[:200]})

# Summary
total = len(results)
ok = sum(1 for r in results if r.get("success"))
print(f"\\n{'='*60}")
print(f"RESULTS: {ok}/{total} ({100*ok/total:.1f}%)")
print(f"{'='*60}")

for ptype in sorted(set(r["type"] for r in results)):
    t_res = [r for r in results if r["type"] == ptype]
    t_ok = sum(1 for r in t_res if r.get("success"))
    print(f"  {ptype}: {t_ok}/{len(t_res)} ({100*t_ok/len(t_res):.0f}%)")

with open(bench_dir / "finetuned_results.json", "w") as f:
    json.dump(results, f, indent=2)
"""


if __name__ == "__main__":
    print("Copy each CELL_* variable into your Colab notebook.")
    print(f"\nCELL_1_LOAD_MODEL: {len(CELL_1_LOAD_MODEL)} chars")
    print(f"CELL_2_TRAIN: {len(CELL_2_TRAIN)} chars")
    print(f"CELL_3_INFERENCE: {len(CELL_3_INFERENCE)} chars")
    print(f"CELL_4_TEST: {len(CELL_4_TEST)} chars")
    print(f"CELL_5_DATASET: {len(CELL_5_DATASET)} chars")
    print(f"CELL_6_BENCHMARK: {len(CELL_6_BENCHMARK)} chars")
