"""Prompts for Phase 1: Spec Analysis — understand the spec before writing code."""

SYSTEM_PROMPT = """You are a formal methods expert analyzing Dafny specifications.
Your job is to read a method specification (requires/ensures) and identify:
1. What algorithm structure is needed (loop, recursion, or direct computation)
2. What the loop invariants or recursive lemma must capture
3. What decreases clause is needed
4. The key proof obligations

Be precise and concrete. Output structured analysis, NOT code.
"""

USER_TEMPLATE = """Analyze this Dafny method specification. Output a structured analysis with these sections:

## 1. Algorithm Structure
What kind of computation is needed? (linear scan, binary search, recursion, two-pointer, etc.)
Is a loop needed? If so, what variables change each iteration?

## 2. Key Variables
What variables will the method need? What are their initial and final values?

## 3. Loop Invariant Candidates
For each postcondition (ensures clause), derive what the corresponding loop invariant should look like.
Pattern: if the ensures says "forall k :: 0 <= k < N ==> P(k)", then the invariant should say
"forall k :: 0 <= k < j ==> P(k)" where j is the loop counter progressing toward N.

## 4. Decreases Clause
What expression strictly decreases each iteration? Must be non-negative integer.

## 5. Bounds Invariants
What bounds must hold on all variables to avoid index-out-of-range errors?

Specification:
{spec}

{examples_section}
"""
