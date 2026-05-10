"""CLI entry point for ProofGen — spec-driven Dafny proof synthesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import project_root, run_pipeline
from .providers import provider_chain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProofGen — spec-driven Dafny proof synthesis optimized for limited LLMs"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # providers subcommand
    providers_p = sub.add_parser(
        "providers",
        help="Print which fallback providers have API keys configured",
    )
    providers_p.add_argument(
        "--llm-task",
        choices=("formal", "default"),
        default="formal",
        help="Ordering: formal hoists preferred_for_formal providers first",
    )

    # run subcommand
    run_p = sub.add_parser("run", help="Run proof synthesis pipeline for one spec YAML")
    run_p.add_argument(
        "spec_yaml",
        type=Path,
        help="Path to specs/*.yaml (relative to cwd or absolute)",
    )
    run_p.add_argument(
        "--max-iters",
        type=int,
        default=12,
        help="Max repair attempts (default 12)",
    )
    run_p.add_argument(
        "--golden",
        action="store_true",
        help="Use bundled reference body (no LLM); smoke-tests pipeline + Dafny",
    )
    run_p.add_argument(
        "--llm-task",
        choices=("formal", "default"),
        default="formal",
        help="Provider ordering for LLM calls",
    )
    run_p.add_argument(
        "-o", "--out",
        type=Path,
        default=None,
        help="Write final .dfy on success",
    )
    run_p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each pipeline stage output to stderr",
    )
    run_p.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the initial spec analysis phase",
    )
    run_p.add_argument(
        "--mode",
        choices=("pipeline", "direct"),
        default="pipeline",
        help=(
            "pipeline: 5-stage decomposition (analyze → skeleton → invariants → repair → holistic). "
            "direct: single-pass generation + targeted repair (faster, less guided)."
        ),
    )

    args = parser.parse_args()

    if args.cmd == "providers":
        names = []
        seen = set()
        for p in provider_chain(args.llm_task):
            if p.enabled and p.name not in seen:
                seen.add(p.name)
                names.append(p.name)
        print(f"Provider fallback order: {names or '(none configured)'}")
        envs = list(dict.fromkeys(p.api_key_env for p in provider_chain(args.llm_task) if p.enabled))
        print("Required env vars (chain order): " + ", ".join(envs))
        return

    if args.cmd == "run":
        spec_path = args.spec_yaml
        if not spec_path.is_absolute():
            spec_path = Path.cwd() / spec_path
        if not spec_path.is_file():
            candidate = project_root() / args.spec_yaml
            if candidate.is_file():
                spec_path = candidate

        if not spec_path.is_file():
            print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
            sys.exit(1)

        ok, source, ver_out = run_pipeline(
            spec_path,
            max_iters=args.max_iters,
            use_golden=args.golden,
            llm_task=args.llm_task,
            verbose=args.verbose,
            skip_analysis=args.skip_analysis,
            mode=args.mode,
        )

        print(ver_out)
        if ok:
            print("\n=== VERIFIED OK ===")
            if args.out:
                args.out.write_text(source, encoding="utf-8")
                print(f"Wrote verified program to {args.out}")
            sys.exit(0)

        print("\n=== VERIFICATION FAILED ===", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
