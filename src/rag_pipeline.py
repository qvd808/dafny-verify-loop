"""RAG-augmented pipeline: same stages as pipeline.py but with retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

from . import examples as exlib
from . import llm
from .pipeline import (
    _extract_body,
    _parses,
    render_template,
    load_yaml_spec,
    project_root,
    run_dafny_verify,
)
from .prompts import analyze as analyze_prompts
from .prompts import invariants as inv_prompts
from .prompts import repair as repair_prompts
from .prompts import skeleton as skel_prompts
from .rag_retrieve import RAGRetriever, format_retrieved_for_prompt, get_retriever
from .verifier import VerificationResult


def _llm_call(system: str, user: str, llm_task: str = "formal", temperature: float = 0.2) -> str:
    try:
        return llm.chat_completion(system, user, llm_task=llm_task, temperature=temperature)
    except RuntimeError:
        raise


# ---------------------------------------------------------------------------
# RAG-augmented pipeline stages
# ---------------------------------------------------------------------------

def stage_analyze_rag(
    spec_text: str,
    meta: dict,
    retriever: RAGRetriever,
    llm_task: str = "formal",
) -> str:
    """Phase 1 with RAG: analyze spec with retrieved similar verified programs."""
    # Retrieve similar specs
    retrieved = retriever.search_for_body_gen(spec_text, k=3)
    rag_text = format_retrieved_for_prompt(retrieved, max_docs=3)
    
    # Also include static few-shot examples
    examples_text = exlib.format_examples_for_prompt(
        exlib.get_all_examples()[:2]
    )

    user = analyze_prompts.USER_TEMPLATE.format(
        spec=spec_text,
        examples_section=f"{rag_text}\n\n{examples_text}",
    )

    raw = _llm_call(analyze_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.1)
    return raw.strip()


def stage_skeleton_rag(
    spec_text: str,
    analysis: str,
    meta: dict,
    retriever: RAGRetriever,
    llm_task: str = "formal",
) -> str:
    """Phase 2 with RAG: generate skeleton with retrieved body patterns."""
    retrieved = retriever.search_for_body_gen(spec_text, k=2)
    rag_text = format_retrieved_for_prompt(retrieved, max_docs=2)

    user = skel_prompts.USER_TEMPLATE.format(
        analysis=analysis,
        spec=spec_text,
        examples_section=f"{rag_text}\n\n// Key: follow the structure of retrieved examples above.",
    )
    raw = _llm_call(skel_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.15)
    return _extract_body(raw)


def stage_invariants_rag(
    spec_text: str,
    skeleton: str,
    analysis: str,
    retriever: RAGRetriever,
    llm_task: str = "formal",
) -> str:
    """Phase 3 with RAG: fill invariants with retrieved invariant patterns."""
    # Retrieve matching invariants based on skeleton structure
    retrieved = retriever.search_for_invariants(skeleton, spec_text, k=3)
    rag_text = format_retrieved_for_prompt(retrieved, max_docs=3)

    # Extract invariant hints from analysis
    hint_lines = []
    for line in analysis.split("\n"):
        if "invariant" in line.lower() or "decreases" in line.lower():
            hint_lines.append(line)
    hint = "\n".join(hint_lines) if hint_lines else ""

    user = inv_prompts.USER_TEMPLATE.format(
        spec=spec_text,
        skeleton=skeleton,
        examples_section=f"{rag_text}",
        analysis_hint=f"Relevant hints from analysis:\n{hint}" if hint else "",
    )
    raw = _llm_call(inv_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.15)
    return _extract_body(raw)


def stage_targeted_repair_rag(
    spec_text: str,
    body: str,
    ver_result: VerificationResult,
    retriever: RAGRetriever,
    llm_task: str = "formal",
) -> str:
    """Phase 4 with RAG: targeted repair with retrieval for similar error fixes."""
    # Retrieve documents matching the error pattern
    error_query = f"{ver_result.suggestion}\n{body[:500]}"
    retrieved = retriever.search_for_repair(error_query, body, k=2)
    rag_text = format_retrieved_for_prompt(retrieved, max_docs=2)

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
        error_output=ver_result.raw_output[:2000],
        diagnosis=diagnosis,
        examples_section=rag_text,
    )
    raw = _llm_call(repair_prompts.SYSTEM_PROMPT, user, llm_task=llm_task, temperature=0.2)
    return _extract_body(raw)


# ---------------------------------------------------------------------------
# RAG pipeline runner (decomposed mode)
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    spec_path: Path,
    *,
    max_iters: int = 12,
    use_golden: bool = False,
    llm_task: str = "formal",
    verbose: bool = False,
    skip_analysis: bool = False,
) -> tuple[bool, str, str]:
    """Run RAG-augmented multi-stage proof synthesis.

    Returns (success, final_dafny_source, last_verifier_output).
    """
    meta = load_yaml_spec(spec_path)
    pid = meta["id"]
    template_rel = meta["dafny_template"]
    placeholder = meta.get("placeholder", "<<<BODY>>>")

    root = project_root()
    spec_text = (root / template_rel).read_text(encoding="utf-8")

    if use_golden:
        from . import golden as golden_bodies
        body = golden_bodies.BODIES.get(pid, "")
        if not body:
            raise RuntimeError(f"No golden body for '{pid}'")
        source = render_template(template_rel, body, placeholder)
        ver_result = run_dafny_verify(source)
        return ver_result.passed, source, ver_result.raw_output

    retriever = get_retriever()

    # ---- Phase 1: Analyze with RAG ----
    if verbose:
        print("=" * 60, file=sys.stderr)
        print("PHASE 1 (RAG): Spec Analysis", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    if skip_analysis:
        analysis = "No analysis performed."
    else:
        analysis = stage_analyze_rag(spec_text, meta, retriever, llm_task=llm_task)

    if verbose:
        print(analysis[:2000], file=sys.stderr)

    # ---- Phase 2: Skeleton with RAG ----
    if verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 2 (RAG): Skeleton Generation", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    skeleton = stage_skeleton_rag(spec_text, analysis, meta, retriever, llm_task=llm_task)
    if verbose:
        print(skeleton[:1500], file=sys.stderr)

    # Parse check
    skeleton_source = render_template(template_rel, skeleton, placeholder)
    ver_result = run_dafny_verify(skeleton_source)
    if verbose:
        print(f"Skeleton parse: {'OK' if _parses(ver_result) else 'SYNTAX ERROR'}", file=sys.stderr)

    if not _parses(ver_result):
        skeleton = stage_skeleton_rag(spec_text, analysis, meta, retriever, llm_task=llm_task)

    # ---- Phase 3: Invariants with RAG ----
    if verbose:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PHASE 3 (RAG): Invariant Synthesis", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    body = stage_invariants_rag(spec_text, skeleton, analysis, retriever, llm_task=llm_task)
    if verbose:
        print(body[:1500], file=sys.stderr)

    # ---- Phase 4: Verify + targeted repair with RAG ----
    source = render_template(template_rel, body, placeholder)
    ver_result = run_dafny_verify(source)

    if verbose and ver_result.passed:
        print("\nVERIFIED on first attempt (with RAG)!", file=sys.stderr)

    repair_count = 0
    last_body = body

    while not ver_result.passed and repair_count < max_iters:
        repair_count += 1

        if verbose:
            print(f"\nREPAIR {repair_count}/{max_iters}: {ver_result.error_types}", file=sys.stderr)
            print(f"  Diagnosis: {ver_result.suggestion[:150]}", file=sys.stderr)

        body = stage_targeted_repair_rag(spec_text, last_body, ver_result, retriever, llm_task=llm_task)

        source = render_template(template_rel, body, placeholder)
        ver_result = run_dafny_verify(source)
        last_body = body

        if verbose:
            status = "VERIFIED" if ver_result.passed else f"Still: {ver_result.error_types}"
            print(f"  -> {status}", file=sys.stderr)

    return ver_result.passed, source, ver_result.raw_output
