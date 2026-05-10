"""Evaluate ProofGen pipeline: given spec (YAML + Dafny template with <<<BODY>>>), 
generate and verify the body."""

import sys, json, time, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BENCH_DIR = Path(__file__).resolve().parent

def run_one(yaml_path, max_iters=8, verbose=False):
    """Run the ProofGen pipeline on one spec YAML."""
    from src.pipeline import run_pipeline
    
    ok, source, ver_out = run_pipeline(
        yaml_path,
        max_iters=max_iters,
        use_golden=False,
        llm_task="formal",
        verbose=verbose,
        mode="pipeline",  # Use the full multi-stage pipeline
    )
    return ok, source, ver_out

def main():
    with open(BENCH_DIR / "benchmark.json") as f:
        meta = json.load(f)
    
    print(f"Body Generation Benchmark: {len(meta)} problems")
    types_summary = {}
    for m in meta:
        types_summary[m['type']] = types_summary.get(m['type'], 0) + 1
    print(f"Types: {types_summary}")
    print()
    
    results = []
    for i, m in enumerate(meta):
        pid = m['id']
        ptype = m['type']
        yaml_path = BENCH_DIR / "specs" / f"{pid}.yaml"
        
        if not yaml_path.exists():
            print(f"[{i+1}/{len(meta)}] {pid[:45]} ... SKIP (no yaml)")
            results.append({"id": pid, "type": ptype, "success": False, "error": "no yaml"})
            continue
        
        print(f"[{i+1}/{len(meta)}] {pid[:45]} ({ptype}) ...", end=" ", flush=True)
        
        try:
            start = time.time()
            ok, source, ver_out = run_one(yaml_path, max_iters=8, verbose=False)
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
            print(f"ERROR: {str(e)[:80]}")
            results.append({
                "id": pid,
                "type": ptype,
                "success": False,
                "error": str(e)[:100],
            })
    
    # Summary
    total = len(results)
    ok_count = sum(1 for r in results if r.get('success'))
    print(f"\n{'='*60}")
    print(f"BODY GENERATION RESULTS: {ok_count}/{total} ({100*ok_count/total:.1f}%)")
    print(f"{'='*60}")
    
    for ptype in sorted(set(r['type'] for r in results)):
        t_res = [r for r in results if r['type'] == ptype]
        t_ok = sum(1 for r in t_res if r.get('success'))
        print(f"  {ptype:15s}: {t_ok}/{len(t_res)} ({100*t_ok/len(t_res):.0f}%)")
    
    # Avg time
    times = [r['time'] for r in results if 'time' in r]
    if times:
        print(f"\nAvg time: {sum(times)/len(times):.1f}s per problem")
    
    with open(BENCH_DIR / "results_body.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {BENCH_DIR / 'results_body.json'}")

if __name__ == "__main__":
    main()
