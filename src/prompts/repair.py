"""Prompts for Phase 4: Targeted Repair — fix specific verification errors."""

SYSTEM_PROMPT = """You repair Dafny method bodies to pass verification.
The method signature (requires/ensures) is FIXED — never modify it.
You only modify the statements inside the method body.

CRITICAL RULES:
- Output ONLY the corrected body statements (no method signature, no outer braces)
- Address the specific error described in the prompt
- Never use {:verify false}, assume, or decreases * to bypass verification
- If a postcondition fails, strengthen loop invariants or add assertions
- If an invariant doesn't hold on entry, fix the initialization
- If an invariant isn't maintained, check the loop body updates
- If termination isn't proved, fix the decreases clause
- Use correct Dafny syntax

Output ONLY the body statements. No explanation, no markdown fences.
"""

POSTCONDITION_USER = """FIX POSTCONDITION FAILURE: The method body does not establish the ensures clause.

Specification:
{spec}

Current body (fails verification):
{body}

Verifier error:
{error_output}

Diagnosis: {diagnosis}

The postcondition is not proved. At the exit point of the method, the verifier
cannot conclude that the ensures clause holds. You need to:
1. Check if loop invariants are strong enough to imply the postcondition
2. Consider what must be true after the loop exits to satisfy each ensures clause
3. Add assertions before the return point if helpful

{examples_section}

Output ONLY the corrected body. No markdown.
"""

INVARIANT_USER = """FIX INVARIANT FAILURE: A loop invariant cannot be proved.

Specification:
{spec}

Current body (fails verification):
{body}

Verifier error:
{error_output}

Diagnosis: {diagnosis}

The loop invariant is either false on entry or not maintained by the loop body.
1. If the invariant fails on entry: check initial variable assignments
2. If the invariant is not maintained: after each loop iteration, the updated variables must still satisfy it
3. The invariant may need to be weakened (if too strong) or strengthened (if too weak to be inductive)

{examples_section}

Output ONLY the corrected body. No markdown.
"""

TERMINATION_USER = """FIX TERMINATION: The verifier cannot prove the loop terminates.

Specification:
{spec}

Current body (fails verification):
{body}

Verifier error:
{error_output}

Diagnosis: {diagnosis}

The decreases clause must:
1. Be a non-negative integer expression
2. Strictly decrease each loop iteration
3. Be bounded below by 0

Common patterns:
- Linear scan: decreases a.Length - j  (where j increases to a.Length)
- Binary search: decreases hi - lo  (where lo increases or hi decreases)

Output ONLY the corrected body. No markdown.
"""

GENERAL_USER = """FIX VERIFICATION ERROR.

Specification:
{spec}

Current body (fails verification):
{body}

Verifier error:
{error_output}

Diagnosis: {diagnosis}

{examples_section}

Output ONLY the corrected body. No markdown.
"""
