"""Evaluation script for ProofGen on the DafnyBench-based benchmark."""

import sys, json, time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_annotate_pipeline

BENCH_DIR = Path(__file__).resolve().parent
PROBLEMS_DIR = BENCH_DIR / "problems"

def main():
    with open(BENCH_DIR / "benchmark.json") as f:
        meta = json.load(f)

    print(f"Benchmark: {len(meta)} problems")
    print(f"Types: { {t: sum(1 for m in meta if m['type']==t) for t in sorted(set(m['type'] for m in meta))} }")
    print()

    results = []
    for i, m in enumerate(meta):
        hints_path = PROBLEMS_DIR / m['hints_file']
        pid = m['id']
        ptype = m['type']

        print(f"[{i+1}/{len(meta)}] {pid[:50]} ({ptype}) ...", end=" ", flush=True)

        try:
            start = time.time()
            ok, source, ver_out = run_annotate_pipeline(
                hints_path,
                max_iters=8,
                llm_task="formal",
                verbose=False,
            )
            elapsed = time.time() - start

            status = "OK" if ok else "FAIL"
            print(f"{status} ({elapsed:.1f}s)")
            results.append({
                "id": pid,
                "type": ptype,
                "success": ok,
                "time": round(elapsed, 1),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": pid,
                "type": ptype,
                "success": False,
                "error": str(e)[:100],
            })

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r.get('success'))
    print(f"\n{'='*60}")
    print(f"RESULTS: {ok}/{total} ({100*ok/total:.1f}%)")
    print(f"{'='*60}")

    # Per-type breakdown
    for ptype in sorted(set(r['type'] for r in results)):
        t_results = [r for r in results if r['type'] == ptype]
        t_ok = sum(1 for r in t_results if r.get('success'))
        print(f"  {ptype}: {t_ok}/{len(t_results)} ({100*t_ok/len(t_results):.0f}%)")

    # Save results
    with open(BENCH_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {BENCH_DIR / 'results.json'}")

if __name__ == "__main__":
    main()
