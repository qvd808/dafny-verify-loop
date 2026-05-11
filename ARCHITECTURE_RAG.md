# ProofGen RAG Architecture — Design Plan

Status: **DRAFT for review** | Date: 2026-05-10

---

## 1. Problem Statement

We need to improve ProofGen's success rate (currently 47.1% on annotation, untested at scale on body generation) using two orthogonal strategies:

| Strategy | Cost | Solves |
|----------|------|--------|
| **RAG** — Retrieve similar verified proofs as few-shot examples | One-time indexing cost | Algorithm design, invariant patterns |
| **Fine-tune 8B** — Train a local model on Dafny annotations | GPU hours (Colab) | Invariant filling, repair loop |

The question is: *does RAG alone get us far enough?* If yes, we skip fine-tuning. If no, we combine both.

---

## 2. RAG Architecture

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (run once)                          │
│                                                                 │
│  500+ verified Dafny programs ──► Parse ──► Embed ──► FAISS    │
│  (DafnyBench ground_truth)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE (per problem)                         │
│                                                                 │
│  New spec ──► Embed query ──► FAISS search ──► top-k matches   │
│                                                  │              │
│                                                  ▼              │
│                              ┌──────────────────────────┐       │
│                              │  Formatted few-shot       │       │
│                              │  examples in prompt       │       │
│                              └──────────┬───────────────┘       │
│                                         ▼                       │
│                              LLM generates body/invariants      │
│                                         │                       │
│                                         ▼                       │
│                              dafny verify → pass/fail           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 What gets indexed

Each DafnyBench ground_truth file is parsed into a structured document with **three embedding views**:

```
Document
├── spec_view      ← "method Foo(x: int) returns (y: int)
│                       requires x > 0
│                       ensures y == x * 2"
│                   
├── body_view      ← "var i := 0;
│                      while i < n
│                        invariant ...
│                        decreases ...
│                      { i := i + 1; }"
│                   
└── invariant_view ← "invariant 0 <= i <= n
│                      invariant acc == sum_range(0, i)
│                      decreases n - i"
│
└── metadata        ← {type: linear_scan, loops: 1, uses_arrays: true, ...}
```

Three separate FAISS indices, one per view. Different pipeline phases query different views:

| Pipeline Phase | Queries | Retrieved view | Why |
|---------------|---------|----------------|-----|
| 1. Analyze | spec_view | spec_view | Understand what similar specs required |
| 2. Skeleton | spec_view | body_view | See how similar algorithms were structured |
| 3. Invariants | body_view | invariant_view | Match loop structure → get correct invariants |
| 4. Repair | body_view + error | invariant_view | Find near-miss invariants that were fixed |

### 2.3 Parsing strategy

Not a full Dafny parser (too complex). A regex-based extractor that identifies:

```
Capture groups:
  method <name>(<params>) returns (<returns>)
    requires <clauses...>
    ensures <clauses...>
  {
    <body>
  }

  predicate <name>(<params>)
    reads <frame>
  { <body> }

  while <cond>
    invariant <clauses...>
    decreases <expr>
  { <body> }

  ghost var <name> := <expr>;
  assert <expr>;
  lemma <name>(<params>) ...
```

Edge cases handled:
- Multiple methods per file → indexed separately
- Helper predicates → indexed alongside their method
- Nested loops → invariants grouped by nesting level
- Comments → stripped before embedding

### 2.4 Embedding model

**Primary choice: LLM API embeddings (NVIDIA NV-Embed-QA or equivalent)**

Rationale:
- Understands formal specification semantics (`forall`, `ensures`, `invariant`)
- Code-specific embedders (CodeBERT, UniXcoder) have never seen Dafny training data
- One-time cost: 500 docs × ~1K tokens × $0.0001/1K tokens ≈ **$0.05 total**
- 4096-dim vectors, excellent for semantic search

**Fallback: Hybrid BM25 + embedding**

For queries that are nearly syntactically identical to indexed documents (common in Dafny — many binary search implementations look similar), BM25 catches exact keyword matches that embeddings might miss:

```
score = 0.7 * cosine_similarity(query_embedding, doc_embedding)
      + 0.3 * BM25(query_tokens, doc_tokens)
```

### 2.5 Metadata filtering

Before FAISS search, apply filters to narrow the candidate pool:

```python
filters = {
    "has_nested_loops": True/False,     # Match loop complexity
    "uses_arrays": True/False,           # Match data structure
    "uses_seqs": True/False,
    "uses_recursion": True/False,
    "num_loops": (min, max),            # Match loop count
    "problem_type": "linear_scan",       # Optional: if we can classify
}
```

This prevents retrieving a recursive solution when the problem needs a loop, or a nested-loop solution when the problem only needs one loop.

---

## 3. Query Strategies

### 3.1 Body generation query

**Input**: Spec template (method signature + requires + ensures, no body)

**Query construction**:
```
Query = f"{method_signature}\n{requires}\n{ensures}"
```

**Expected retrieval**: Methods with similar signatures and postconditions. E.g., querying `BinarySearch(a: array<int>, x: int)` should retrieve other binary search implementations and potentially linear search (structurally similar — scan with elimination).

### 3.2 Annotation filling query

**Input**: Hints-removed program (body exists, invariants missing)

**Query construction**:
```
# Extract the loop skeleton without invariants
skeleton = extract_while_loops(body)  # "while lo < hi { var mid := ...; if ... }"
Query = f"{ensures}\n{skeleton}"
```

**Expected retrieval**: Programs with similar loop structure AND similar postconditions → their invariants are likely applicable.

### 3.3 Repair query

**Input**: Failing program + verifier error output

**Query construction**:
```
error_type = classify_error(verifier_output)  # "postcondition", "invariant_maintained", etc.
Query = f"{error_type}\n{failing_invariant_or_condition}"
```

**Expected retrieval**: Programs where a similar invariant was needed to fix a similar error.

---

## 4. Integration with ProofGen Pipeline

### 4.1 Modified pipeline

```
                    ┌──────────────────┐
                    │   New spec.yaml   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  RAG LOOKUP       │  ◄── one embedding call
                    │  Query: spec      │
                    │  Returns: top-3   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Retrieved│  │ Retrieved│  │ Retrieved│
        │ Example 1│  │ Example 2│  │ Example 3│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                   ┌────────▼─────────┐
                   │  PHASE 1          │  ◄── API call #1 (70B)
                   │  Analyze spec     │
                   │  (prompt includes │
                   │   RAG examples)   │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  PHASE 2          │  ◄── API call #2 (70B)
                   │  Generate skeleton│
                   │  (guided by RAG   │
                   │   body patterns)  │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  RAG LOOKUP       │  ◄── second embedding call
                   │  Query: skeleton  │
                   │  Returns: top-3   │
                   │  invariants       │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  PHASE 3          │  ◄── LOCAL fine-tuned 8B
                   │  Fill invariants  │      (or API if not trained)
                   │  (prompt includes │
                   │   matched invs)   │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  PHASE 4          │  ◄── LOCAL fine-tuned 8B
                   │  Targeted repair  │      (0-5 iterations, free)
                   │  (RAG repair      │
                   │   examples)       │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  dafny verify     │
                   │  → pass / fail    │
                   └──────────────────┘
```

### 4.2 API call budget

| Step | API calls | Cost per problem |
|------|-----------|-----------------|
| Initial RAG lookup (embed query) | 1 embedding | ~$0.0001 |
| Phase 1 (analyze) | 1 LLM (70B) | ~$0.002 |
| Phase 2 (skeleton) | 1 LLM (70B) | ~$0.002 |
| Invariant RAG lookup | 1 embedding | ~$0.0001 |
| Phase 3 (invariants) | **Local (free)** | $0 |
| Phase 4 (repair × N) | **Local (free)** | $0 |
| **Total per problem** | **2 LLM + 2 embedding** | **~$0.004** |

At this cost, you can run **250 problems for ~$1**.

---

## 5. Data Model

### 5.1 Indexed document schema

```python
@dataclass
class DafnyDocument:
    # Identity
    doc_id: str                    # e.g., "Clover_binary_search"
    source_file: str               # original .dfy path
    
    # Extracted segments
    methods: list[MethodInfo]
    
    # Embedding views (stored as numpy arrays in FAISS)
    spec_embedding: np.ndarray     # 4096-dim float32
    body_embedding: np.ndarray     # 4096-dim float32
    invariant_embedding: np.ndarray # 4096-dim float32
    
    # Metadata for filtering
    problem_type: str              # binary_search | linear_scan | recursive | ...
    num_loops: int
    has_nested_loops: bool
    uses_arrays: bool
    uses_seqs: bool
    uses_sets: bool
    uses_recursion: bool
    max_loop_depth: int
    verified: bool                 # always True for ground_truth


@dataclass
class MethodInfo:
    name: str
    params: list[tuple[str, str]]           # [(name, type)]
    returns: list[tuple[str, str]]
    requires: list[str]
    ensures: list[str]
    body: str
    invariants: list[list[str]]             # per-loop
    decreases: list[str | None]             # per-loop
    helper_predicates: list[str]
    helper_lemmas: list[str]
```

### 5.2 FAISS index layout

```
faiss_index/
├── spec_index.faiss          # 500 × 4096 float32 (~8 MB)
├── body_index.faiss          # 500 × 4096 float32 (~8 MB)
├── invariant_index.faiss     # 500 × 4096 float32 (~8 MB)
├── metadata.json             # 500 records (~2 MB)
├── documents.json            # Full extracted text (~20 MB)
└── bm25_index.pkl            # BM25 token index (~5 MB)
```

Total: ~50 MB. Fast to load, fast to search (<10ms per query on CPU).

---

## 6. Prompt Format with RAG

### 6.1 Example: Phase 1 prompt with RAG

```
SYSTEM: You are a formal methods expert analyzing Dafny specifications.

RETRIEVED SIMILAR VERIFIED PROGRAMS:
─── Example 1 (binary_search, verified) ───
method BinarySearch(arr: array<int>, key: int) returns (index: int)
  requires sorted(arr)
  ensures index >= 0 ==> arr[index] == key
  ensures index < 0 ==> key not in arr
// [full body + invariants from ground_truth]

─── Example 2 (linear_search, verified) ───
method LinearSearch(arr: array<int>, key: int) returns (index: int)
  ensures index >= 0 ==> arr[index] == key
  ensures index < 0 ==> key not in arr
// [full body + invariants from ground_truth]

─── TARGET PROBLEM ───
{predicate Sorted(a: array<int>) ...}
method BinarySearch(a: array<int>, x: int) returns (index: int)
  requires Sorted(a)
  ensures index >= 0 ==> index < a.Length && a[index] == x
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x

Analyze this specification. Identify algorithm type, invariants needed, and decreases clause.
```

### 6.2 Example: Phase 3 prompt with RAG

```
SYSTEM: You fill loop invariants for Dafny methods.

RETRIEVED INVARIANT PATTERNS FOR SIMILAR LOOPS:
─── Pattern 1 (binary search, lo/hi bounds) ───
while lo < hi
  invariant 0 <= lo <= hi <= a.Length
  invariant forall k :: 0 <= k < lo ==> a[k] < x
  invariant forall k :: hi <= k < a.Length ==> a[k] > x
  decreases hi - lo

─── TARGET SKELETON ───
while lo < hi
  /* invariant: ??? */
  /* decreases: ??? */
{
  var mid := lo + (hi - lo) / 2;
  if a[mid] < x { lo := mid + 1; }
  else if x < a[mid] { hi := mid; }
  else { return mid; }
}

Fill in the correct invariants and decreases clause following the pattern above.
```

---

## 7. Implementation Plan

### Phase A: Build the index (1-2 days)

| Step | Effort | Description |
|------|--------|-------------|
| A1 | 2h | Dafny regex parser — extract methods, specs, bodies, invariants |
| A2 | 1h | Metadata classifier — tag each document with problem_type, loops, arrays/segs/sets |
| A3 | 1h | Embedding generation — call NVIDIA embedding API for all 500 documents |
| A4 | 1h | FAISS index construction — build 3 indices + BM25 |
| A5 | 1h | Validation — spot-check retrieval quality on 10 known queries |

### Phase B: Integrate with pipeline (1 day)

| Step | Effort | Description |
|------|--------|-------------|
| B1 | 2h | RAG lookup module — query embedding + FAISS search + metadata filter |
| B2 | 1h | Prompt formatter — inject retrieved examples into existing prompt templates |
| B3 | 2h | Pipeline integration — wire RAG lookups into phases 1, 2, 3 |
| B4 | 1h | Evaluation — re-run 17-problem benchmark with RAG enabled |

### Phase C: Fine-tuning (if needed) (2-3 days)

| Step | Effort | Description |
|------|--------|-------------|
| C1 | 2h | Training data preparation — 500 annotation pairs from DafnyBench |
| C2 | 3h | QLoRA fine-tuning on Colab T4 — unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit |
| C3 | 1h | GGUF conversion — quantize for local inference |
| C4 | 1h | Pipeline integration — wire local model into phases 3-4 |
| C5 | 1h | Evaluation — re-run benchmark with fine-tuned model |

### Phase D: Optimization (ongoing)

- Hybrid retrieval tuning (BM25 weight, k value, filter thresholds)
- Prompt template refinement based on failure analysis
- Add more documents to index (DafnySynth, custom problems)
- Monitor retrieval quality over time

---

## 8. Expected Impact

Based on the literature and the architecture above, here's my projection:

| Configuration | Annotation (17-prob) | Body gen (estimate) |
|---------------|---------------------|---------------------|
| Baseline (current, no RAG) | 47.1% | ~20-30% (untested) |
| + RAG only | 55-65% | ~40-50% |
| + RAG + fine-tuned 8B | 65-75% | ~45-55% |
| + RAG + fine-tuned 8B + voting (pass@5) | 80-85% | ~55-65% |

The ceiling without frontier models (Claude/GPT-5 class) is probably around 80-85%. The remaining 15-20% of problems require genuine mathematical insight that current open-weight models don't have — things like non-linear arithmetic lemmas, complex permutation invariants, or termination proofs for non-obvious well-founded orders.

---

## 9. Open Questions for Review

1. **Embedding model**: Use NVIDIA's embedding API, or try a local code embedder first (faster iteration, no cost)? The local option would mean using something like CodeBERT or UniXcoder — lower quality but zero cost and offline.

2. **Fine-tuning priority**: Should we build RAG first and evaluate, THEN decide if fine-tuning is needed? Or start both in parallel? My vote: RAG first (2 days), evaluate, then decide.

3. **Base model for fine-tuning**: DeepSeek-R1-Distill-Llama-8B (reasoning model) vs Qwen2.5-Coder-7B (code model)? The reasoning model might overthink simple invariant filling — worth a quick test before committing.

4. **Index maintenance**: Should the RAG index be a static artifact, or should we add successfully generated proofs back into the index (self-improving)? The latter introduces risk of contamination but could improve retrieval for novel problem types.

5. **Minimum viable retrieval**: Can we start with just BM25 (keyword matching) instead of embeddings? Dafny is keyword-dense — `ensures forall k :: 0 <= k < a.Length ==> a[k] <= a[i]` might retrieve well with BM25 alone. This would let us test the RAG pipeline in hours instead of days.

---

*End of architecture plan. Ready for review.*
