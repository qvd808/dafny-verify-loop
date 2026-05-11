"""Local LLM adapter for ProofGen pipeline.

Supports (all instruction/instruct models — NO reasoning/think tags):
- deepseek-coder-7b-instruct-v1.5  ← RECOMMENDED for Dafny (code-native, 7B)
- DeepSeek-Coder-V2-Lite-Instruct  (16B MoE, stronger, untested w/ QLoRA)
- Llama-3.1-8B-Instruct           (Paper 2 baseline, 50.6% on DafnyBench)
- Qwen2.5-Coder-7B-Instruct       (code-specialized alternative)
- DeepSeek-R1-Distill-Llama-8B    (reasoning model, NOT recommended)

Usage (Colab):
    from src.local_llm import patch_pipeline_with_local_model
    patch_pipeline_with_local_model(model, tokenizer, model_type="deepseek_coder")

    # Then run as normal:
    from src.pipeline import run_annotate_pipeline
    ok, _, _ = run_annotate_pipeline(...)
"""

from __future__ import annotations

import re
import warnings
from typing import Any


class LocalLLMAdapter:
    """Adapter wrapping a HF model+tokenizer to match llm.chat_completion interface."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        model_type: str = "llama3",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_type = model_type

    def __call__(
        self,
        system: str,
        user: str,
        *,
        llm_task: str = "formal",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_sec: float = 120.0,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_new_tokens

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        raw = self._run_generate(prompt, temp, max_tok)
        return self._clean_output(raw)

    def _run_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        # Don't skip_special_tokens — SentencePiece models (DeepSeek-Coder)
        # encode spaces as special tokens (▁), which get stripped otherwise,
        # producing "methodAbs" instead of "method Abs".
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Manually strip only the EOS token and padding
        eos = self.tokenizer.eos_token
        if eos and generated_text.endswith(eos):
            generated_text = generated_text[:-len(eos)]
        # Also strip common EOS variants
        for eos_variant in ["</s>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]:
            generated_text = generated_text.replace(eos_variant, "")
        return generated_text.strip()

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip chat markers and artifacts from generated output.

        Handles all major instruct model families:
        - DeepSeek-Coder v1.5: ### Instruction: / ### Response: (Alpaca)
        - DeepSeek-Coder V2-Lite: User: / Assistant:
        - DeepSeek-R1:  <think>... response (reasoning)
        - Llama 3.1: <|start_header_id|>...<|eot_id|>
        - Qwen: <|im_start|>...<|im_end|>
        """
        # ---- R1 reasoning model: extract content after  response ----
        m = re.search(r"response\s*(.*?)$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            # Strip  <think> blocks if no  response found
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # ---- Strip chat turn markers (model trying to start new turn) ----
        stop_markers = [
            # DeepSeek-Coder v1.5 (Alpaca-style)
            "### Instruction:", "### Instruction",
            "### Response:", "### Response",
            "### Human:", "### Assistant:",
            "### User:",
            # DeepSeek V2 / R1
            "<｜Assistant｜>", "<｜User｜>", "<｜System｜>",
            "Assistant:", "User:", "System:",
            # Llama 3.1
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|eot_id|>",
            # Qwen
            "<|im_start|>assistant", "<|im_start|>user",
            "<|im_start|>system", "<|im_end|>",
            # Generic
            "### Assistant:", "### User:", "### Human:",
        ]
        for marker in stop_markers:
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx].strip()

        # ---- Strip trailing incomplete marker fragments ----
        incomplete = [
            "<|start_header", "<|eot", "<|im_start",
            "<｜", "### Instruct", "### Respon",
        ]
        for frag in incomplete:
            if text.rstrip().endswith(frag):
                idx = text.rfind(frag)
                text = text[:idx].strip()

        return text.strip()


def patch_pipeline_with_local_model(
    model: Any,
    tokenizer: Any,
    *,
    model_type: str = "llama3",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> LocalLLMAdapter:
    """Monkey-patch src.llm.chat_completion to use a local model."""
    from . import llm
    adapter = LocalLLMAdapter(
        model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, temperature=temperature,
        model_type=model_type,
    )
    llm.chat_completion = adapter
    return adapter
