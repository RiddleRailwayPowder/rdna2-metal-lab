"""Recompute family/role in rows.jsonl with the current classify.token_context (no taint rerun).
    python3 -I relabel.py   # rewrites rows.jsonl in place
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import classify
rows = [json.loads(l) for l in open(os.path.join(HERE, "rows.jsonl"))]
cache = {}
for r in rows:
    if r["name"] not in cache:
        cache[r["name"]] = open(os.path.join(classify.PREP, r["name"], "ref.metal")).read().split("\n")
    line = cache[r["name"]][r["line"] - 1]
    r["family"], r["role"] = classify.token_context(line, r["col"], r["orig"], r["kind"])
with open(os.path.join(HERE, "rows.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
