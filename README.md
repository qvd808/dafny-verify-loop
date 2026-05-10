# ProofGen — Spec-Driven Dafny Proof Synthesis (Limited-Model Optimized)

Given a Dafny specification (method signature + requires/ensures), generate a verified method body using LLMs optimized for limited-capability models.

## Key Innovations for Limited Models

The integra-v4 SpecLock pipeline struggles with limited models (Llama 3.3, Mistral Small, etc.) because it asks the LLM to simultaneously:
1. Design the algorithm
2. Invent loop invariants
3. Write correct Dafny syntax
4. Prove termination

**ProofGen decomposes proof generation into narrow, focused stages:**

| Stage | Task | Why it helps limited models |
|-------|------|-----------------------------|
| **1. Analyze** | Understand the spec (no code) | Separates reasoning from coding |
| **2. Skeleton** | Write code structure with placeholder comments | Constrains syntax decisions |
| **3. Invariants** | Fill in loop invariants + decreases | Focuses on the hardest part in isolation |
| **4. Targeted Repair** | Fix specific error types (postcondition, invariant, termination) | Error-specific prompts are more effective |
| **5. Holistic Retry** | Full-context repair (fallback) | Last resort with maximum context |

Additional optimizations:
- **Few-shot examples**: Automatically selects structurally similar solved problems
- **Intelligent error parsing**: Classifies Dafny errors into categories (postcondition, invariant_entry, invariant_maintained, decreases, bounds, etc.)
- **Progressive temperature**: Lower temperature for analysis (0.1), higher for creative repair (0.3)
- **Truncated error output**: Limits verifier output to 2000 chars to avoid context overflow
- **Skeleton parse check**: Validates syntax before spending tokens on invariants

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify infrastructure (no LLM needed)
python -m src run specs/binary_search.yaml --golden
python -m src run specs/max_index.yaml --golden
python -m src run specs/reverse_seq.yaml --golden

# Check LLM provider configuration
python -m src providers

# Run with LLM (requires .env with API keys)
python -m src run specs/binary_search.yaml --max-iters 12 -v -o out/binary_search.dfy
```

## Spec Format

Same as integra-v4 SpecLock:

```yaml
# specs/my_problem.yaml
id: my_problem
name: Human-readable name
description: >
  What the method should do.
dafny_template: problems/my_problem/spec.template.dfy
placeholder: "<<<BODY>>>"
```

```dafny
// problems/my_problem/spec.template.dfy
method MyMethod(params) returns (results)
  requires ...
  ensures ...
{
<<<BODY>>>
}
```

## Project Structure

```
test_dafny/
  specs/                  # YAML spec files (same format as integra-v4)
  problems/               # Dafny template files with <<<BODY>>>
  src/
    __init__.py
    __main__.py           # CLI entry point
    pipeline.py           # Multi-stage pipeline orchestrator
    verifier.py           # Dafny verify wrapper + intelligent error parsing
    llm.py                # OpenAI-compatible chat with provider fallback
    providers.py          # Provider registry (Groq, NVIDIA, SambaNova, etc.)
    examples.py           # Few-shot example library
    golden.py             # Reference bodies for smoke testing
    prompts/
      analyze.py          # Phase 1: Spec analysis prompts
      skeleton.py         # Phase 2: Skeleton generation prompts
      invariants.py       # Phase 3: Invariant synthesis prompts
      repair.py           # Phase 4-5: Targeted + holistic repair prompts
  requirements.txt
```

## Comparison with integra-v4

| Feature | integra-v4 SpecLock | ProofGen |
|---------|---------------------|----------|
| Single-pass generation | Yes (generate_body node) | No (multi-stage) |
| Error feedback | Raw verifier output | Parsed + classified errors |
| Few-shot examples | None | Automatic selection by problem type |
| Invariant synthesis | LLM must guess | Derived from spec analysis |
| Repair strategy | Generic retry | Error-type-specific prompts |
| LangGraph dependency | Yes | No (stdlib only) |
| Skeleton validation | No | Yes (parse check before invariants) |

## LLM Providers

Configure API keys in `.env`:

```bash
GROQ_API_KEY=...
NVIDIA_API_KEY=...
SAMBANOVA_API_KEY=...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=...
CEREBRAS_API_KEY=...
```

Provider fallback is automatic — if one returns 429/5xx/404, the next is tried.

**Formal task mode** (`--llm-task formal`, default) hoists providers with `preferred_for_formal=True` first (NVIDIA, SambaNova, Mistral).
