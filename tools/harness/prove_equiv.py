"""Equivalence proofs for every pinned and holdout mutant, with no GPU and no inputs.

The proofs don't depend on inputs, so they can run before (or alongside) a search and be handed to goal_report.py:
  1. metal -O2 IR identical (source path normalised), then air-opt -O3 IR identical;
  2. z3_equiv: bit-vector SMT over the mutated expression;
  3. dead_store: strong liveness says the changed value never reaches an output, memory or control flow.
Every row is written as it finishes; a rerun skips rows already in the output (resume).

Usage: python3 -I prove_equiv.py <prep dir> [--jobs N] [--out FILE] [--only NAME,NAME] [--retry-not-proved]
  4. z3_fp: float-aware SMT (.safe semantics, both denormal-flush choices); a robust difference is kept as "witness".
Output rows: name, sample, kind, line, col, orig, rep, resolution ("equivalent" | "not_proved" | "error"), proof.
"""
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlog  # noqa: E402
import resolve_survivors as rs  # noqa: E402
import z3_equiv  # noqa: E402
import dead_store  # noqa: E402
import z3_fp  # noqa: E402

LOG = None


def key(r):
    return (r["name"], r["line"], r["col"], r["orig"], r["rep"])


def prove_shader(d, todo):
    name = os.path.basename(d)
    try:
        msl = open(os.path.join(d, "ref.metal")).read()
        out = []
        cache = {}
        with tempfile.TemporaryDirectory() as tmp:
            ref_ir = rs.ir(os.path.join(d, "ref.metal"), tmp, "ref")
            ref_o3 = None
            for k, m in enumerate(todo):
                row = dict(m, name=name)
                try:
                    text = rs.mutant_text(msl, m)
                except ValueError as e:
                    row.update(resolution="error", error=str(e))
                    out.append(row)
                    continue
                mp = os.path.join(tmp, f"m{k}.metal")
                open(mp, "w").write(text)
                mir = rs.ir(mp, tmp, f"m{k}") if ref_ir is not None else None
                if mir is not None and mir == ref_ir:
                    out.append(dict(row, resolution="equivalent", proof="metal -O2 IR identical"))
                    continue
                if mir is not None:
                    if ref_o3 is None:
                        ref_o3 = rs.ir_o3(ref_ir, tmp, "ref") or ""
                    mo3 = rs.ir_o3(mir, tmp, f"m{k}")
                    if ref_o3 and mo3 is not None and mo3 == ref_o3:
                        out.append(dict(row, resolution="equivalent", proof="air-opt -O3 IR identical"))
                        continue
                ok, why = z3_equiv.prove_equivalent(msl, m)
                if ok:
                    out.append(dict(row, resolution="equivalent", proof=why))
                    continue
                try:
                    fok, fwhy, fwit = z3_fp.prove_equivalent(msl, m, timeout_s=20)
                except Exception as e:  # prover failure is "not proved", never a proof
                    fok, fwhy, fwit = None, f"z3_fp error: {e}", None
                if fok:
                    out.append(dict(row, resolution="equivalent", proof="z3_fp: " + str(fwhy)))
                    continue
                if fok is False and fwit:
                    row["witness"] = fwit                  # robust statement-level difference: steering target
                dok, dwhy = dead_store.prove_dead(msl, m, cache)
                if dok:
                    out.append(dict(row, resolution="equivalent", proof=dwhy))
                    continue
                out.append(dict(row, resolution="not_proved", z3=why, z3_fp=fwhy, dead=dwhy, ir_compiled=mir is not None))
        return out
    except Exception as e:  # one bad shader never stops the batch
        fail = runlog.exception(e, LOG, f"prove {name}")
        return [dict(m, name=name, resolution="error", failure=fail) for m in todo]


def main(argv):
    global LOG
    if not argv:
        print(__doc__)
        return 2
    prep = argv[0]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 4
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(prep, "..", "equiv_all.jsonl")
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    retry = "--retry-not-proved" in argv   # re-run rows not proved before (e.g. after adding a prover)
    LOG = runlog.setup(out_path, "prove_equiv", verbose="--verbose" in argv)
    rs.LOG = LOG
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("resolution") == "equivalent" or (r.get("resolution") == "not_proved" and not retry):
                done.add(key(r))
    work = []
    for name in sorted(os.listdir(prep)):
        d = os.path.join(prep, name)
        if only and name not in only or not os.path.exists(os.path.join(d, "ref.metal")):
            continue
        todo = []
        for sample in ("pinned", "holdout"):
            p = os.path.join(d, f"mutants_{sample}.json")
            if os.path.exists(p):
                for m in json.load(open(p)):
                    m = dict(m, sample=sample)
                    if key(dict(m, name=name)) not in done:
                        todo.append(m)
        if todo:
            work.append((d, todo))
    n = sum(len(t) for _, t in work)
    LOG.info("prove_equiv: %d mutants in %d shaders (%d already done), jobs %d", n, len(work), len(done), jobs)
    c = Counter()
    with ThreadPoolExecutor(jobs) as ex, open(out_path, "a") as f:
        futs = [ex.submit(prove_shader, d, t) for d, t in work]
        for i, fut in enumerate(as_completed(futs), 1):
            for row in fut.result():
                c[(row.get("sample"), row["resolution"])] += 1
                f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(futs):
                LOG.info("  %d/%d shaders: %s", i, len(futs), dict(c))
    LOG.info("prove_equiv done: %s", dict(c))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
