# RAG for Body Generation — Concrete Walkthrough

## The Question

> "How do we test if RAG actually helps generate method bodies?"

We test it by taking a spec, running it through the pipeline twice — once with RAG, once without — and comparing the output. If RAG retrieves a structurally similar verified proof and the LLM produces a correct body in fewer attempts (or succeeds where the baseline fails), RAG works.

Below I walk through a real example step by step.

---

## Step 0 — The Setup

### What's in the index

We've indexed 500 verified Dafny programs from DafnyBench. Each program is parsed into three views:

```
Document: "Clover_binary_search"
  spec_view:   "method BinarySearch(a: array<int>, key: int) returns (idx: int)
                  requires sorted(a)
                  ensures idx >= 0 ==> a[idx] == key
                  ensures idx < 0 ==> key not in a"

  body_view:   "var lo := 0;
                var hi := a.Length;
                while lo < hi
                  var mid := lo + (hi - lo) / 2;
                  if a[mid] < key { lo := mid + 1; }
                  else if key < a[mid] { hi := mid; }
                  else { return mid; }
                return -1;"

  invariant_view: "invariant 0 <= lo <= hi <= a.Length
                   invariant forall k :: 0 <= k < lo ==> a[k] < key
                   invariant forall k :: hi <= k < a.Length ==> a[k] > key
                   decreases hi - lo"

  metadata: {type: "binary_search", loops: 1, has_nested: false,
             uses_arrays: true, uses_seqs: false}
```

### The test problem (NOT in the index)

We take a problem that is similar-but-different from anything in the index. For this walkthrough, let's use a **"find first occurrence"** variant — a binary search that finds the *leftmost* occurrence of a target in a sorted array with duplicates:

```dafny
// NOT in the index — this is our test problem
predicate Sorted(a: array<int>)
  reads a
{
  forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
}

method FindFirst(a: array<int>, x: int) returns (index: int)
  requires Sorted(a)
  ensures index >= 0 ==> index < a.Length && a[index] == x
  ensures index >= 0 ==> forall k :: 0 <= k < index ==> a[k] != x
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x
{
  <<<BODY>>>
}
```

This is NOT in DafnyBench. It's similar to binary search but needs a different loop invariant (must track the earliest occurrence).

---

## Step 1 — Embed the Query

We take the spec (signature + requires + ensures) and embed it into a 4096-dim vector.

```
Query text:
"method FindFirst(a: array<int>, x: int) returns (index: int)
   requires Sorted(a)
   ensures index >= 0 ==> index < a.Length && a[index] == x
   ensures index >= 0 ==> forall k :: 0 <= k < index ==> a[k] != x
   ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x"

→ NVIDIA NV-Embed-QA → [0.023, -0.451, 0.891, ..., 0.112]  (4096 floats)
```

---

## Step 2 — FAISS Search

We search the `spec_view` index for the top-3 most similar documents:

```
Query vector: [0.023, -0.451, 0.891, ..., 0.112]
                    │
                    ▼ cosine similarity against all 500 documents
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
  Rank 1          Rank 2          Rank 3
  score: 0.94     score: 0.87     score: 0.81
  │               │               │
  │               │               └── "LinearSearch" (Clover)
  │               │                   method LinearSearch(a, key) returns (idx)
  │               │                     ensures idx >= 0 ==> a[idx] == key
  │               │                     ensures idx < 0 ==> key not in a
  │               │                   // Similar ensures, but linear scan pattern
  │               │
  │               └── "binary_search" (Dafny-demo)
  │                   method BinarySearch(a, key) returns (index)
  │                     ensures index >= 0 ==> a[index] == key
  │                     ensures index < 0 ==> key not in a
  │                   // Standard binary search, no duplicate handling
  │
  └── "binary_search" (Clover)
      method BinarySearch(a, key) returns (idx)
        ensures idx >= 0 ==> a[idx] == key
        ensures idx < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != key
      // Closest match — identical postcondition structure
```

Then we apply metadata filters. The query uses `array<int>` → filter `uses_arrays: true`. Query has single return value → filter `num_loops: (0, 2)`. The top 3 all pass.

---

## Step 3 — Retrieve Full Documents

For each ranked match, we pull the full document from storage. Here's what Rank 1 looks like:

```dafny
// ===== RETRIEVED EXAMPLE 1 (score: 0.94) =====
// Source: Clover_binary_search (from DafnyBench ground_truth)

predicate sorted(a: array<int>)
  reads a
{
  forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
}

method BinarySearch(a: array<int>, key: int) returns (idx: int)
  requires sorted(a)
  ensures idx >= 0 ==> idx < a.Length && a[idx] == key
  ensures idx < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != key
{
  var lo := 0;
  var hi := a.Length;
  while lo < hi
    invariant 0 <= lo <= hi <= a.Length
    invariant forall k :: 0 <= k < lo ==> a[k] < key
    invariant forall k :: hi <= k < a.Length ==> a[k] > key
    decreases hi - lo
  {
    var mid := (lo + hi) / 2;
    if a[mid] < key {
      lo := mid + 1;
    } else if key < a[mid] {
      hi := mid;
    } else {
      return mid;
    }
  }
  return -1;
}
```

---

## Step 4 — Build the Prompt (WITH RAG)

Now we inject the retrieved examples into the Phase 1 (Analyze) prompt:

````
SYSTEM:
You are a formal methods expert analyzing Dafny specifications.
Your job is to read a method specification (requires/ensures) and identify:
1. What algorithm structure is needed (loop, recursion, or direct computation)
2. What the loop invariants or recursive lemma must capture
3. What decreases clause is needed
4. The key proof obligations

Below are VERIFIED solutions to similar problems. Study them, then analyze
the TARGET problem.

═══════════════════════════════════════════
RETRIEVED EXAMPLE 1 — Binary Search (standard)
═══════════════════════════════════════════
// Full ground_truth source code shown here...
[the Clover binary search from above]

KEY INSIGHT FROM THIS EXAMPLE:
- The loop invariant partitions the array: elements left of `lo` are < key,
  elements right of `hi` are > key
- The bounds invariant `0 <= lo <= hi <= a.Length` prevents index errors
- decreases `hi - lo` ensures termination because search space shrinks

═══════════════════════════════════════════
RETRIEVED EXAMPLE 2 — Binary Search (Dafny-demo variant)
═══════════════════════════════════════════
// Another binary search with slightly different style...

═══════════════════════════════════════════
RETRIEVED EXAMPLE 3 — Linear Search
═══════════════════════════════════════════
// Linear scan — similar postconditions but different algorithm...

═══════════════════════════════════════════
TARGET PROBLEM
═══════════════════════════════════════════
predicate Sorted(a: array<int>)
  reads a
{
  forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
}

method FindFirst(a: array<int>, x: int) returns (index: int)
  requires Sorted(a)
  ensures index >= 0 ==> index < a.Length && a[index] == x
  ensures index >= 0 ==> forall k :: 0 <= k < index ==> a[k] != x
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x

Analyze this specification.
````

---

## Step 5 — Compare: WITH RAG vs WITHOUT RAG

### Without RAG (current pipeline)

The LLM sees only the spec. It must reason from scratch:

```
LLM ANALYSIS (no RAG):
"## 1. Algorithm Structure
 Binary search — the array is sorted, so we can eliminate half the
 search space each iteration.

 ## 3. Loop Invariant Candidates
 invariant 0 <= lo <= hi <= a.Length
 invariant forall k :: 0 <= k < lo ==> a[k] < x
 invariant forall k :: hi <= k < a.Length ==> a[k] > x
 // NOTE: This is the STANDARD binary search invariant.
 // It does NOT handle the "first occurrence" requirement."
```

The LLM produces the standard binary search body. When we verify it, it FAILS on the second postcondition: `forall k :: 0 <= k < index ==> a[k] != x`. The standard invariant says `a[k] < x` for k < lo, which is stronger than needed but doesn't guarantee `a[k] != x` at the found position if there are duplicates.

The repair loop then needs to figure out: we need `a[mid] == x` to continue searching left, not return immediately. This requires multiple repair iterations.

### With RAG

The LLM sees the standard binary search AND understands the extra requirement from the target spec:

```
LLM ANALYSIS (with RAG):
"## 1. Algorithm Structure
 Binary search, same core as Example 1. HOWEVER, the target spec has
 an additional ensures: elements before the found index must NOT equal x.
 This means when we find a[mid] == x, we cannot return immediately —
 we must continue searching left to find the FIRST occurrence.

 ## 3. Loop Invariant Candidates (adapted from Example 1)
 The standard partition invariants apply:
   invariant 0 <= lo <= hi <= a.Length
   invariant forall k :: 0 <= k < lo ==> a[k] < x
   invariant forall k :: hi <= k < a.Length ==> a[k] > x

 ADDITIONALLY, we need to track that we've seen x:
   invariant index == -1 || (a[index] == x && forall k :: 0 <= k < index ==> a[k] != x)

 And when a[mid] == x, instead of returning, we set index := mid; hi := mid;
 This continues searching left while preserving the invariant."
```

The LLM now produces:

```dafny
var lo := 0;
var hi := a.Length;
index := -1;
while lo < hi
  invariant 0 <= lo <= hi <= a.Length
  invariant forall k :: 0 <= k < lo ==> a[k] < x
  invariant forall k :: hi <= k < a.Length ==> a[k] > x
  invariant index == -1 || (0 <= index < a.Length && a[index] == x &&
            forall k :: 0 <= k < index ==> a[k] != x)
  decreases hi - lo
{
  var mid := lo + (hi - lo) / 2;
  if a[mid] < x {
    lo := mid + 1;
  } else if x < a[mid] {
    hi := mid;
  } else {
    index := mid;
    hi := mid;  // continue searching left
  }
}
```

This verifies on the first or second attempt.

---

## Step 6 — How We Measure Success

### The comparison protocol

For a set of N test problems (NOT in the index):

```
For each problem:
  1. Run WITHOUT RAG: pipeline → count repair iterations → record pass/fail
  2. Run WITH RAG:    pipeline → count repair iterations → record pass/fail
  3. Compare:
     - Did RAG succeed where no-RAG failed?
     - Did RAG reduce repair iterations?
     - Did RAG produce better invariant quality (judged by human or by
       diff against ground truth)?
```

### Concrete metrics

| Metric | Without RAG | With RAG | What we learn |
|--------|-------------|----------|---------------|
| Pass@1 rate | e.g. 3/10 | e.g. 6/10 | Does RAG make first attempts more likely to succeed? |
| Avg repair iterations | e.g. 4.2 | e.g. 2.1 | Does RAG reduce the repair burden? |
| Pass@8 rate | e.g. 5/10 | e.g. 8/10 | Does RAG enable solving problems that were unsolvable before? |
| Invariant precision | e.g. 60% exact match | e.g. 80% exact match | Are retrieved invariants being correctly adapted? |
| Retrieval relevance | — | e.g. 0.85 avg score | Are we retrieving the right documents? |

### Test set construction

Critical: the test problems must NOT be in the index. Options:

1. **Hold-out from DafnyBench**: Index 450 problems, test on the remaining 50. Simple but risks overfitting to DafnyBench's distribution.

2. **Problems from testdafny110**: Index on DafnyBench, test on testdafny110's 110 programs. Different source = stronger test.

3. **Hand-crafted variants**: Take known problems and change one requirement (like we did with FindFirst). Tests generalization.

4. **The user's own specs**: The 3 problems already in `test_dafny/specs/`. If RAG improves these, that's a practical win.

### What "RAG is working" looks like

```
Problem: FindFirst (duplicate-aware binary search)
──────────────────────────────────────────────
Without RAG:  3 attempts → FAIL (postcondition not proved)
With RAG:     1 attempt  → PASS  (correctly adapted invariant)
                                 
Retrieved:    Clover_binary_search (0.94), Dafny-demo_binary_search (0.87),
              Clover_linear_search (0.81)
              
What RAG provided: The structural template for binary search invariants.
The LLM only had to add ONE extra invariant for the "first occurrence"
tracking — it didn't have to invent binary search from scratch.
```

---

## Step 7 — The Minimal Viable Test

Before building the full RAG system, here's the quickest way to validate the idea:

### 30-minute smoke test

```
1. Pick 3 problems from our /specs/ directory
2. For each, manually find the top-2 similar DafnyBench programs
   (use grep to search by keywords: "binary", "max", "reverse")
3. Hard-code those retrieved examples into the prompt
4. Run the pipeline
5. Compare against the results we already have (all 3 passed without RAG,
   but measure: fewer repair iterations? Better invariants?)
```

If even hard-coded retrieval helps (fewer repairs, better invariants), then building the FAISS index is worth it. If there's no difference on already-solvable problems, we need harder test problems first.

### 2-hour validation test

```
1. Build a BM25-only index (keyword search, no embeddings)
2. Test on 10 held-out DafnyBench problems
3. Compare with-RAG vs without-RAG using the metrics above
4. If BM25 alone helps, embeddings will help more
```

BM25 is a 30-minute build. It won't capture semantic similarity ("first occurrence" ≈ "lower bound") but it WILL retrieve structurally identical programs (both use `while lo < hi` with `a[mid]`).

---

## Summary

The test boils down to:

```
For each test problem:
  ┌──────────────────────────────────────────────┐
  │ 1. Embed spec → FAISS search → top-k docs     │
  │ 2. Inject retrieved docs into prompt           │
  │ 3. Run pipeline (analyze → skeleton → invars)  │
  │ 4. dafny verify                                │
  │ 5. Compare: pass/fail, iterations, quality     │
  │                                                │
  │ Control: same pipeline, same LLM, NO RAG       │
  │ (just the generic few-shot examples we already  │
  │  have in examples.py)                          │
  └──────────────────────────────────────────────┘
```

The hypothesis: **RAG succeeds because it constrains the LLM's search space.** Instead of "invent a binary search from scratch," the task becomes "adapt this verified binary search to the target spec's additional requirement." That adaptation is a much smaller cognitive leap for a limited model.
