"""Multi-stage proof synthesis pipeline optimized for limited LLMs.

Modes:
  - pipeline (default): 5-stage decomposition (analyze → skeleton → invariants → repair → holistic)
  - direct: single-pass generation + targeted repair (like integra-v4 but with better prompts)

Stages:
  1. Spec Analysis — understand what the proof needs
  2. Skeleton Generation — create code structure without invariants
  3. Invariant Synthesis — fill in loop invariants and decreases
  4. Verification + Targeted Repair — fix specific errors
  5. (fallback) Holistic Retry — if targeted repair fails
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import examples as exlib
from . import llm
from .prompts import analyze as analyze_prompts
from .prompts import invariants as inv_prompts
from .prompts import repair as repair_prompts
from .prompts import skeleton as skel_prompts
from .verifier import VerificationResult, run_dafny_verify


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml_spec(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_template(template_rel: str, body: str, placeholder: str = "<<<BODY>>>") -> str:
    root = project_root()
    text = (root / template_rel).read_text(encoding="utf-8")
    if placeholder not in text:
        raise ValueError(f"Placeholder {placeholder!r} not found in {template_rel}")
    return text.replace(placeholder, body)


def _extract_body(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace from LLM output."""
    s = raw.strip()
    # Try to extract from markdown code fences
    m = re.match(r"^```(?:dafny)?\s*\n(.*)\n```\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # Also try: content between ```dafny and ``` not at exact edges
    m2 = re.search(r"```(?:dafny)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m2:
        s = m2.group(1).strip()
    return s


def _select_examples(spec_text: str, analysis_text: str = "") -> list[exlib.Example]:
    """Select relevant few-shot examples based on spec characteristics."""
    combined = (spec_text + " " + analysis_text).lower()
    selected: list[exlib.Example] = []

    if "sorted" in combined or "binary" in combined:
        selected.extend(exlib.get_examples_by_type("binary_search"))
    if "max" in combined or "maximum" in combined or "largest" in combined:
        selected.extend(exlib.get_examples_by_type("linear_scan"))
    if "reverse" in combined or "rev" in combined:
        selected.extend(exlib.get_examples_by_type("recursive"))
    if "sum" in combined or "accumulat" in combined:
        selected.extend(exlib.get_examples_by_type("linear_scan"))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for ex in selected:
        if ex[3] not in seen:
            seen.add(ex[3])
            unique.append(ex)
    return unique[:2] or exlib.get_all_examples()[:2]


def _llm_call(system: str, user: str, llm_task: str = "formal", temperature: float = 0.2) -> str:
    """Wrapper for LLM chat completion."""
    try:
        return llm.chat_completion(system, user, llm_task=llm_task, temperature=temperature)
    except RuntimeError as e:
        if "No LLM API keys" in str(e):
            raise
        raise


# ---------------------------------------------------------------------------
# Pipeline stages (decomposed mode)
# ---------------------------------------------------------------------------

def stage_analyze(
    spec_text: str,
    meta: dict,
    llm_task: str = "formal",
) -> str:
    """Phase 1: Analyze the spec to understand proof requirements."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text))

    user = analyze_prompts.USER_TEMPLATE.format(
        spec=spec_text,
        examples_section=examples_text,
    )

    # For limited models, lower temperature for analysis (more deterministic)
    raw = _llm_call(analyze_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.1)
    return raw.strip()


def stage_skeleton(
    spec_text: str,
    analysis: str,
    meta: dict,
    llm_task: str = "formal",
) -> str:
    """Phase 2: Generate code skeleton from analysis."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text, analysis))

    user = skel_prompts.USER_TEMPLATE.format(
        analysis=analysis,
        spec=spec_text,
        examples_section=examples_text,
    )
    raw = _llm_call(skel_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.15)
    return _extract_body(raw)


def stage_invariants(
    spec_text: str,
    skeleton: str,
    analysis: str,
    llm_task: str = "formal",
) -> str:
    """Phase 3: Fill in loop invariants and decreases clauses."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text, analysis))

    # Extract invariant hints from analysis
    hint_lines = []
    for line in analysis.split("\n"):
        if "invariant" in line.lower() or "decreases" in line.lower():
            hint_lines.append(line)
    hint = "\n".join(hint_lines) if hint_lines else ""

    user = inv_prompts.USER_TEMPLATE.format(
        spec=spec_text,
        skeleton=skeleton,
        examples_section=examples_text,
        analysis_hint=f"Relevant hints from analysis:\n{hint}" if hint else "",
    )
    raw = _llm_call(inv_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.15)
    return _extract_body(raw)


def stage_targeted_repair(
    spec_text: str,
    body: str,
    ver_result: VerificationResult,
    llm_task: str = "formal",
) -> str:
    """Phase 4: Targeted repair based on parsed error types."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text))

    error_types = ver_result.error_types
    diagnosis = ver_result.suggestion

    if "postcondition" in error_types:
        user_template = repair_prompts.POSTCONDITION_USER
    elif any(t in error_types for t in ("invariant", "invariant_entry", "invariant_maintained")):
        user_template = repair_prompts.INVARIANT_USER
    elif "decreases" in error_types:
        user_template = repair_prompts.TERMINATION_USER
    else:
        user_template = repair_prompts.GENERAL_USER

    user = user_template.format(
        spec=spec_text,
        body=body,
        error_output=ver_result.raw_output[:2000],  # Truncate for limited context
        diagnosis=diagnosis,
        examples_section=examples_text,
    )
    raw = _llm_call(repair_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.2)
    return _extract_body(raw)


def stage_holistic_retry(
    spec_text: str,
    body: str,
    ver_result: VerificationResult,
    meta: dict,
    attempt: int,
    llm_task: str = "formal",
) -> str:
    """Phase 5: Holistic retry with full error context (fallback)."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text))

    system = """You are a Dafny expert. The method body below fails verification.
Fix ALL errors so the body satisfies the specification.
Output ONLY the corrected method body (no method signature, no outer braces).
Do NOT use {:verify false}, assume, or decreases *.
"""
    user = f"""Fix this Dafny method body to pass verification.

Specification:
{spec_text}

Current body (fails verification):
{body}

Verifier errors (attempt {attempt}):
{ver_result.raw_output[:3000]}

Diagnosis: {ver_result.suggestion}

{examples_text}

Output ONLY the corrected body statements. No markdown.
"""
    raw = _llm_call(system, user, llm_task=llm_task, temperature=0.3)
    return _extract_body(raw)


# ---------------------------------------------------------------------------
# Direct mode (single-pass with better prompts)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = """You are an expert Dafny programmer.
Implement ONLY the method body for the given specification.

RULES:
- Output ONLY the statements that replace <<<BODY>>> (no method signature, no outer braces).
- Use correct Dafny syntax: loops need invariant and decreases for total correctness.
- Include all necessary loop invariants — they must be true on entry, preserved by the loop body,
  and strong enough to prove the postcondition.
- For array/sequence problems, include bounds invariants.
- Use decreases clauses for all loops and recursive calls.

FEW-SHOT PATTERNS (common proof structures):
- Linear scan with accumulator: invariant acc == f(processed_range), decreases N - j
- Linear scan finding max: invariant forall k :: 0 <= k < j ==> a[k] <= a[max_i]
- Binary search: invariants about partition left of lo and right of hi
- Recursion: ensure recursive call meets precondition

Return ONLY the body statements, no markdown fences.
"""


def stage_direct_generate(
    spec_text: str,
    meta: dict,
    llm_task: str = "formal",
) -> str:
    """Single-pass body generation with few-shot examples."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(spec_text))

    user_parts = [
        f"Problem id: {meta.get('id', '')}",
        f"Name: {meta.get('name', '')}",
        f"Description: {meta.get('description', '')}",
        "",
        "Template (replace <<<BODY>>> with your implementation):",
        spec_text,
        "",
        examples_text,
    ]
    user = "\n".join(user_parts)
    raw = _llm_call(_DIRECT_SYSTEM, user, llm_task=llm_task, temperature=0.2)
    return _extract_body(raw)


# ---------------------------------------------------------------------------
# Main pipeline dispatcher
# ---------------------------------------------------------------------------

def run_pipeline(
    spec_path: Path,
    *,
    max_iters: int = 12,
    use_golden: bool = False,
    llm_task: str = "formal",
    verbose: bool = False,
    skip_analysis: bool = False,
    mode: str = "pipeline",
) -> tuple[bool, str, str]:
    """Run the optimized proof synthesis pipeline.

    Args:
        spec_path: Path to spec YAML file
        max_iters: Maximum repair attempts
        use_golden: Use bundled reference body (skip LLM entirely)
        llm_task: Provider ordering hint ("formal" or "default")
        verbose: Print stage details to stderr
        skip_analysis: Skip Phase 1 (spec analysis)
        mode: "pipeline" (multi-stage) or "direct" (single-pass + repair)

    Returns (success, final_dafny_source, last_verifier_output).
    """
    meta = load_yaml_spec(spec_path)
    pid = meta["id"]
    template_rel = meta["dafny_template"]
    placeholder = meta.get("placeholder", "<<<BODY>>>")

    root = project_root()
    spec_text = (root / template_rel).read_text(encoding="utf-8")

    # Golden path: use pre-verified reference body
    if use_golden:
        from . import golden as golden_bodies
        body = golden_bodies.BODIES.get(pid, "")
        if not body:
            raise RuntimeError(f"No golden body for '{pid}'")
        source = render_template(template_rel, body, placeholder)
        ver_result = run_dafny_verify(source)
        return ver_result.passed, source, ver_result.raw_output

    # Dispatch to selected mode
    if mode == "direct":
        return _run_direct_mode(
            meta, spec_text, template_rel, placeholder,
            max_iters, llm_task, verbose,
        )
    else:
        return _run_pipeline_mode(
            meta, spec_text, template_rel, placeholder,
            max_iters, llm_task, verbose, skip_analysis,
        )


def _run_direct_mode(
    meta: dict,
    spec_text: str,
    template_rel: str,
    placeholder: str,
    max_iters: int,
    llm_task: str,
    verbose: bool,
) -> tuple[bool, str, str]:
    """Direct mode: single-pass generation + targeted repair loop."""
    if verbose:
        print("=" * 60, file=sys.stderr)
        print("DIRECT MODE: single-pass + targeted repair", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    body = stage_direct_generate(spec_text, meta, llm_task=llm_task)
    if verbose:
        print("Initial generation:", file=sys.stderr)
        print(body, file=sys.stderr)

    source = render_template(template_rel, body, placeholder)
    ver_result = run_dafny_verify(source)

    if ver_result.passed:
        if verbose:
            print("VERIFIED on first attempt!", file=sys.stderr)
        return True, source, ver_result.raw_output

    last_body = body
    for i in range(max_iters):
        if verbose:
            print(f"\n--- Repair attempt {i + 1}/{max_iters} ---", file=sys.stderr)
            print(f"Errors: {ver_result.error_types}", file=sys.stderr)
            print(f"Diagnosis: {ver_result.suggestion[:200]}", file=sys.stderr)

        # Use targeted repair first, then holistic
        if i < 4:
            body = stage_targeted_repair(spec_text, last_body, ver_result, llm_task=llm_task)
        else:
            body = stage_holistic_retry(spec_text, last_body, ver_result, meta, i + 1, llm_task=llm_task)

        source = render_template(template_rel, body, placeholder)
        ver_result = run_dafny_verify(source)
        last_body = body

        if ver_result.passed:
            if verbose:
                print("VERIFIED!", file=sys.stderr)
                print(source, file=sys.stderr)
            return True, source, ver_result.raw_output

        if verbose:
            print(f"  -> Still failing: {', '.join(ver_result.error_types)}", file=sys.stderr)

    return False, source, ver_result.raw_output


def _run_pipeline_mode(
    meta: dict,
    spec_text: str,
    template_rel: str,
    placeholder: str,
    max_iters: int,
    llm_task: str,
    verbose: bool,
    skip_analysis: bool,
) -> tuple[bool, str, str]:
    """Full multi-stage pipeline mode."""
    # ---- Phase 1: Analyze spec ----
    if verbose:
        print("=" * 60, file=sys.stderr)
        print("PHASE 1: Spec Analysis", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    if skip_analysis:
        analysis = "No analysis performed (--skip-analysis)."
    else:
        try:
            analysis = stage_analyze(spec_text, meta, llm_task=llm_task)
        except RuntimeError:
            analysis = "LLM unavailable for analysis. Proceeding with direct generation."

    if verbose:
        print(analysis, file=sys.stderr)

    # ---- Phase 2: Generate skeleton ----
    if verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 2: Skeleton Generation", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    skeleton = stage_skeleton(spec_text, analysis, meta, llm_task=llm_task)
    if verbose:
        print(skeleton, file=sys.stderr)

    # Verify that skeleton at least parses
    skeleton_source = render_template(template_rel, skeleton, placeholder)
    ver_result = run_dafny_verify(skeleton_source)
    if verbose:
        print(f"\nSkeleton parse check: {'OK' if _parses(ver_result) else 'SYNTAX ERROR'}", file=sys.stderr)

    # If skeleton has syntax errors, retry once
    if not _parses(ver_result):
        if verbose:
            print("Skeleton has syntax errors, retrying...", file=sys.stderr)
        skeleton = stage_skeleton(spec_text, analysis, meta, llm_task=llm_task)
        skeleton_source = render_template(template_rel, skeleton, placeholder)
        ver_result = run_dafny_verify(skeleton_source)

    # ---- Phase 3: Synthesize invariants ----
    if verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 3: Invariant Synthesis", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    body = stage_invariants(spec_text, skeleton, analysis, llm_task=llm_task)
    if verbose:
        print(body, file=sys.stderr)

    # ---- Phase 4: Verify and targeted repair loop ----
    source = render_template(template_rel, body, placeholder)
    ver_result = run_dafny_verify(source)

    if verbose and ver_result.passed:
        print("\nVERIFIED on first attempt!", file=sys.stderr)

    repair_count = 0
    holistic_count = 0
    last_body = body

    while not ver_result.passed and repair_count < max_iters:
        repair_count += 1

        if verbose:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"REPAIR ATTEMPT {repair_count}/{max_iters}", file=sys.stderr)
            print(f"Error types: {ver_result.error_types}", file=sys.stderr)
            print(f"Diagnosis: {ver_result.suggestion[:200]}", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)

        # Targeted repair first (up to 4 times), then holistic
        if repair_count <= 4:
            body = stage_targeted_repair(spec_text, last_body, ver_result, llm_task=llm_task)
        else:
            holistic_count += 1
            body = stage_holistic_retry(spec_text, last_body, ver_result, meta, holistic_count, llm_task=llm_task)

        source = render_template(template_rel, body, placeholder)
        ver_result = run_dafny_verify(source)
        last_body = body

        if verbose:
            status = "VERIFIED" if ver_result.passed else f"Still failing: {', '.join(ver_result.error_types)}"
            print(f"  -> {status}", file=sys.stderr)
            if ver_result.passed:
                print(source, file=sys.stderr)

    return ver_result.passed, source, ver_result.raw_output


def _parses(ver_result: VerificationResult) -> bool:
    """Check if the Dafny code at least parses (no syntax errors)."""
    output = ver_result.raw_output.lower()
    if "parse error" in output or "syntax error" in output:
        return False
    if "error" in output and "verified, 0 errors" not in output:
        if any(term in output for term in ("might not hold", "cannot prove", "postcondition", "invariant", "decreases")):
            return True
    return True

# ============================================================================
# Annotate mode: fill in missing annotations (invariants, decreases, asserts)
# for programs that already have the code body.
# ============================================================================

_ANNOTATE_SYSTEM = """You are a Dafny verification expert. The program below has a complete
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


def stage_annotate(
    hints_content: str,
    ground_content: str,
    llm_task: str = "formal",
) -> str:
    """Fill in missing annotations for a program that has the body but lacks invariants/decreases."""
    # Select relevant examples
    examples_text = exlib.format_examples_for_prompt(_select_examples(hints_content))

    user = f"""Add loop invariants, decreases clauses, and proof assertions to make this Dafny program verify.

CURRENT PROGRAM (annotations removed — does NOT verify):
{hints_content}

{examples_text}

Add the missing annotations. Output the COMPLETE program with annotations.
"""
    raw = _llm_call(_ANNOTATE_SYSTEM, user, llm_task=llm_task, temperature=0.1)
    return raw.strip()


def stage_annotate_repair(
    hints_content: str,
    current_content: str,
    ver_result: VerificationResult,
    llm_task: str = "formal",
) -> str:
    """Targeted repair for annotation filling."""
    examples_text = exlib.format_examples_for_prompt(_select_examples(hints_content))

    diagnosis = ver_result.suggestion

    user = f"""Fix the verification errors in this Dafny program by adjusting the annotations.

ORIGINAL PROGRAM (without annotations):
{hints_content}

CURRENT PROGRAM (with annotations, fails verification):
{current_content[:4000]}

VERIFIER ERRORS:
{ver_result.raw_output[:2000]}

DIAGNOSIS: {diagnosis}

{examples_text}

Adjust the annotations (invariants, decreases, asserts) to fix these errors.
Do NOT change the executable code. Output the COMPLETE program.
"""
    raw = _llm_call(_ANNOTATE_SYSTEM, user, llm_task=llm_task, temperature=0.2)
    return raw.strip()


def run_annotate_pipeline(
    hints_path: Path,
    *,
    max_iters: int = 8,
    llm_task: str = "formal",
    verbose: bool = False,
) -> tuple[bool, str, str]:
    """Run the annotation-filling pipeline on a DafnyBench-style problem.

    Args:
        hints_path: Path to the hints_removed .dfy file
        max_iters: Maximum repair attempts
        llm_task: Provider ordering hint
        verbose: Print details to stderr

    Returns (success, final_dafny_source, last_verifier_output).
    """
    hints_content = hints_path.read_text(encoding="utf-8", errors="replace")

    if verbose:
        print(f"--- Annotating: {hints_path.name[:60]} ---", file=sys.stderr)

    # Generate annotations
    annotated = stage_annotate(hints_content, "", llm_task=llm_task)

    # Strip markdown fences
    annotated = _extract_body(annotated)
    if not annotated or len(annotated) < len(hints_content) * 0.5:
        # If output looks truncated, try again
        annotated = stage_annotate(hints_content, "", llm_task=llm_task)
        annotated = _extract_body(annotated)

    ver_result = run_dafny_verify(annotated)

    if ver_result.passed:
        if verbose:
            print("  VERIFIED on first attempt!", file=sys.stderr)
        return True, annotated, ver_result.raw_output

    current = annotated
    for i in range(max_iters):
        if verbose:
            print(f"  Repair {i+1}/{max_iters}: {ver_result.error_types}", file=sys.stderr)

        current = stage_annotate_repair(hints_content, current, ver_result, llm_task=llm_task)
        current = _extract_body(current)
        ver_result = run_dafny_verify(current)

        if ver_result.passed:
            if verbose:
                print(f"  VERIFIED after {i+1} repairs!", file=sys.stderr)
            return True, current, ver_result.raw_output

    return False, current, ver_result.raw_output
