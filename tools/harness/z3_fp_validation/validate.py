"""Validate z3_fp on run_150_v3: soundness on killed mutants, power on unobservable/unresolved survivors.

    python3 validate.py [out.jsonl] [--resume]     (--resume skips rows already in out.jsonl and appends)
    python3 validate.py out.jsonl --summary
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/harness")
import z3_fp  # noqa: E402

WORK = os.environ.get("GAME_ROOT", ".") + "/harness_work"
RUN = f"{WORK}/run_150_v3.jsonl"
RESOLVED = f"{WORK}/run_150_v3.jsonl.resolved.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results.jsonl")
RESUME = "--resume" in sys.argv

_msl = {}


def msl_of(name):
    if name not in _msl:
        _msl.clear()
        _msl[name] = open(f"{WORK}/prep_h/{name}/ref.metal").read()
    return _msl[name]


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                      # file still being appended: skip a partial last line
    return rows


def main():
    jobs = []
    for shader in read_jsonl(RUN):
        for m in shader.get("mutants", []):
            if m.get("verdict") == "killed":
                jobs.append(("sound:killed", shader["name"], m))
    resolved = read_jsonl(RESOLVED)
    for r in resolved:
        res = r.get("resolution")
        if res in ("killed_by_hunt", "killed_by_gate"):
            jobs.append((f"sound:{res}", r["name"], r))
        elif res in ("unobservable", "unresolved"):
            jobs.append((f"power:{res}", r["name"], r))
        elif res == "equivalent":
            jobs.append(("xcheck:equivalent", r["name"], r))
    jobs.sort(key=lambda j: j[1])
    done = set()
    if RESUME and os.path.exists(OUT):
        done = {(r["cat"], r["name"], r["line"], r["col"], r["rep"]) for r in read_jsonl(OUT)}
    t0 = time.time()
    with open(OUT, "a" if RESUME else "w") as out:
        for i, (cat, name, m) in enumerate(jobs):
            if (cat, name, m["line"], m["col"], m["rep"]) in done:
                continue
            t = time.time()
            ok, why, wit = z3_fp.prove_equivalent(msl_of(name), m)
            rec = {"cat": cat, "name": name, "kind": m["kind"], "line": m["line"], "col": m["col"], "orig": m["orig"],
                   "rep": m["rep"], "src": m["src"], "ok": ok, "why": why, "witness": wit,
                   "secs": round(time.time() - t, 2)}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if cat.startswith("sound") and ok is True:
                print("UNSOUND", json.dumps(rec), flush=True)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    summarize(OUT)


def summarize(path):
    rows = read_jsonl(path)
    by = collections.defaultdict(collections.Counter)
    reasons = collections.Counter()
    for r in rows:
        v = {True: "proved", False: "witness", None: "none+witness" if r["witness"] else "none"}[r["ok"]]
        by[r["cat"]][v] += 1
        by[r["cat"]]["total"] += 1
        if r["cat"].startswith("power") and r["ok"] is None:
            why = r["why"]
            if why.startswith("out of scope: call") or why.startswith("out of scope: member") or \
                    why.startswith("out of scope: unknown type"):
                why = " ".join(why.split()[:5])
            reasons[why] += 1
    for cat in sorted(by):
        print(cat, dict(by[cat]))
    print("unsound (True on a killed mutant):",
          sum(1 for r in rows if r["cat"].startswith("sound") and r["ok"] is True))
    print("power None reasons:")
    for why, n in reasons.most_common(25):
        print(f"  {n:4} {why}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--summary":
        summarize(sys.argv[1])
    else:
        main()
