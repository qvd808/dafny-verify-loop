"""Dafny verification wrapper with intelligent error parsing."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerificationResult:
    passed: bool
    raw_output: str
    # Parsed error details for targeted repair
    error_lines: list[str] = field(default_factory=list)
    error_types: list[str] = field(default_factory=list)  # e.g. "postcondition", "invariant", "decreases", "syntax"
    failing_assertions: list[str] = field(default_factory=list)
    suggestion: str = ""


# Patterns for parsing Dafny verifier output
_ERROR_PATTERNS = [
    # Postcondition violation
    (r"this postcondition might not hold", "postcondition"),
    (r"a postcondition.*could not be proved", "postcondition"),
    (r"ensures clause might not hold", "postcondition"),
    # Loop invariant issues
    (r"this loop invariant might not hold(?: on entry)?", "invariant_entry"),
    (r"this loop invariant might not be maintained", "invariant_maintained"),
    (r"invariant might not hold", "invariant"),
    # Termination
    (r"cannot prove termination", "decreases"),
    (r"decreases clause might not", "decreases"),
    (r"cannot prove.*decreases", "decreases"),
    # Assertion
    (r"assertion might not hold", "assertion"),
    (r"this assertion could not be proved", "assertion"),
    # Precondition
    (r"a precondition.*might not hold", "precondition"),
    (r"requires clause might not hold", "precondition"),
    # Index out of bounds
    (r"index out of range", "bounds"),
    (r"array index.*might be out", "bounds"),
    (r"sequence index.*might be out", "bounds"),
    # Division/modulo by zero
    (r"possible division by zero", "div_zero"),
    # General verification failure
    (r"Verification of .+ timed out", "timeout"),
    (r"verified, 0 errors", "success"),
]


def run_dafny_verify(dfy_source: str) -> VerificationResult:
    """Run dafny verify and parse the output for targeted repair hints."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        candidate = root / "candidate.dfy"
        candidate.write_text(dfy_source, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["dafny", "verify", str(candidate)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                raw_output="Dafny verification timed out after 120s.",
                error_types=["timeout"],
                suggestion="The proof is likely too complex or has non-terminating verification. Simplify.",
            )

        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0

        if ok:
            return VerificationResult(passed=True, raw_output=out, suggestion="")

        return _parse_errors(out)


def _parse_errors(output: str) -> VerificationResult:
    """Parse Dafny output to identify specific error types and locations."""
    lines = output.split("\n")
    error_lines: list[str] = []
    error_types: list[str] = []
    failing_assertions: list[str] = []

    # Collect error lines (lines with file:line:col: pattern)
    for line in lines:
        if re.search(r"\.dfy\(\d+,\d+\)", line) or "Error" in line or "error" in line:
            error_lines.append(line.strip())

    # Classify errors
    detected = set()
    for line in lines:
        line_lower = line.lower()
        for pattern, etype in _ERROR_PATTERNS:
            if re.search(pattern, line_lower):
                detected.add(etype)
                break

    error_types = list(detected)

    # Extract specific failing assertions/postconditions
    for line in lines:
        stripped = line.strip()
        if "might not hold" in stripped.lower() or "could not be proved" in stripped.lower():
            # Try to extract the assertion text
            m = re.search(r'"(.*?)"', stripped)
            if m:
                failing_assertions.append(m.group(1))

    # Generate targeted suggestion
    suggestion = _generate_suggestion(error_types, failing_assertions, output)

    return VerificationResult(
        passed=False,
        raw_output=output,
        error_lines=error_lines,
        error_types=error_types,
        failing_assertions=failing_assertions,
        suggestion=suggestion,
    )


def _generate_suggestion(error_types: list[str], failing: list[str], _output: str) -> str:
    """Generate a targeted repair suggestion based on parsed error types."""
    parts: list[str] = []

    if "syntax" in error_types or "Error" in str(error_types):
        parts.append("SYNTAX: Check for missing semicolons, incorrect Dafny syntax, or unclosed braces.")

    if "postcondition" in error_types:
        parts.append(
            "POSTCONDITION FAILURE: The body does not establish the ensures clause. "
            "Strengthen loop invariants or add assertions before the return point. "
            "Think: at the exit point, what facts must be true to imply the postcondition?"
        )
        if failing:
            parts.append(f"  Failing condition(s): {', '.join(failing)}")

    if "invariant_entry" in error_types:
        parts.append(
            "INVARIANT ENTRY: A loop invariant is not true before the first iteration. "
            "Check that initial variable assignments satisfy all invariants."
        )

    if "invariant_maintained" in error_types:
        parts.append(
            "INVARIANT MAINTENANCE: The loop body does not preserve all invariants. "
            "After each iteration, the invariant must still hold. "
            "Check variable updates and consider strengthening the invariant."
        )

    if "invariant" in error_types and "invariant_entry" not in error_types and "invariant_maintained" not in error_types:
        parts.append(
            "INVARIANT ISSUE: A loop invariant cannot be proved. Ensure invariants are "
            "inductive (true on entry and preserved by each iteration) and strong enough "
            "to prove the postcondition."
        )

    if "decreases" in error_types:
        parts.append(
            "TERMINATION: Add or fix the decreases clause. Use a non-negative integer "
            "expression that strictly decreases each iteration (e.g., hi - lo, a.Length - j)."
        )

    if "bounds" in error_types:
        parts.append(
            "ARRAY/SEQ BOUNDS: An index might be out of range. Add bounds invariants "
            "(e.g., 0 <= i < a.Length) and check all array/sequence accesses."
        )

    if "precondition" in error_types:
        parts.append(
            "PRECONDITION: A requires clause of a called method/function is not satisfied. "
            "Check that arguments to recursive calls or helper methods meet their preconditions."
        )

    if "assertion" in error_types:
        parts.append(
            "ASSERTION: An assert statement cannot be proved. Either the assertion is false "
            "or the verifier needs stronger preceding invariants/assertions."
        )

    if "timeout" in error_types:
        parts.append(
            "TIMEOUT: Verification took too long. The proof is likely too complex. "
            "Consider simplifying the algorithm or adding intermediate assertions."
        )

    if not parts:
        parts.append(
            "VERIFICATION FAILED: Review the Dafny output carefully. "
            "Common issues: missing loop invariants, incorrect decreases clause, "
            "or body logic that doesn't satisfy the specification."
        )

    return " | ".join(parts)
