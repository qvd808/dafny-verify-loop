"""Prompts for Phase 2: Skeleton Generation — create code structure without invariants."""

SYSTEM_PROMPT = """You write Dafny method BODIES (statements only, no signature).
Given an analysis of what the method needs to do, write ONLY the executable statements.

CRITICAL RULES:
- Output ONLY the statements that go inside the method body (no method signature, no outer braces)
- Include loop structure with empty invariant/decreases placeholders marked with comments
- Use CORRECT Dafny syntax for all statements
- Do NOT write invariants yet — use comments like /* invariant: ... */
- Do NOT write decreases clauses yet — use comments like /* decreases: ... */
- Include ALL variable declarations, assignments, conditionals, and return statements
- The code must be syntactically valid Dafny (it should parse even with comment placeholders)

Output ONLY the body code. No explanation, no markdown fences.
"""

USER_TEMPLATE = """Write the method body skeleton based on this analysis.

Analysis:
{analysis}

Specification being implemented:
{spec}

{examples_section}

Write ONLY the method body (statements that replace <<<BODY>>>).
Use /* invariant: ... */ comments for invariants you'll fill in later.
Use /* decreases: ... */ for the termination measure.
"""
