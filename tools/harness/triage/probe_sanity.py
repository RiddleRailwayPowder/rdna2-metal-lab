"""Probe sanity: on statements whose block ran (and whose lane guard fired, if any), every outermost ternary that is
not inside a short-circuit operand must have fired t or f. Reports the fraction that did.
    python3 -I probe_sanity.py
"""
import os  # noqa: E401
import json, os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taint
PREP = os.environ.get("GAME_ROOT", ".") + "/harness_work/prep_h"
ok = bad = 0
badlist = []
for f in glob.glob("cov/*.json"):
    if f.endswith(".ref.json") or f.endswith(".inputs.json"):
        continue
    cov = json.load(open(f)); name = cov["shader"]; hits = set(cov["cov_hits"])
    arms = json.load(open(os.path.join("probe", os.path.basename(f)[:-5], "arms.json")))
    t = taint.tree_for(os.path.join(PREP, name, "ref.metal"))
    lines = open(os.path.join(PREP, name, "ref.metal")).read().split("\n")
    for li, recs in arms.items():
        li = int(li)
        b = t.block_of.get(li)
        if b is None or b not in hits:
            continue
        g = t.guard_of.get(li)
        if g is not None and g not in hits:
            continue
        spans = [(a[0], c[1]) for _, _, a, c in recs]
        for (tt, ff, a, c) in recs:
            inner = any(s0 <= a[0] - 2 and c[1] <= s1 and (s0, s1) != (a[0], c[1]) for s0, s1 in spans)
            if inner:
                continue
            pre = lines[li][:a[0]]
            if "&&" in pre or "||" in pre:
                continue
            if tt in hits or ff in hits:
                ok += 1
            else:
                bad += 1; badlist.append((name, li + 1))
print("outermost ternaries on executed statements:", ok + bad, "fired:", ok, "never fired:", bad)
print(badlist[:10])
