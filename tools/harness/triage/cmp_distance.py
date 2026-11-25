"""Closest operand distance at the mutated comparison, over inputs.json: one cmplog.metal run per shader
(A only, --cmplog 1). Distance = |bits(a) - bits(b)| of the closest pair the shader logged for that comparison id
(ULP distance for floats of one sign). A threshold mutant K -> K+1 needs the operand at K or K+1.
    python3 -I cmp_distance.py > cmp_distance.json
"""
import os  # noqa: E401
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("GAME_ROOT", ".") + "/harness_work"
P = WORK + "/prep_h/"
H = os.environ.get("GAME_ROOT", ".") + "/tools/harness/harness"
key = lambda n: "".join(c if c.isalnum() else "_" for c in n)

def run(n):
    d = P + n
    cmd = [H, "--meta", d + "/meta.json", "--a", d + "/cmplog.metal", "--b", "none", "--inputs",
           os.path.join(HERE, "cov", key(n) + ".inputs.json"), "--all", "1", "--coverage", "1", "--cmplog", "1"]
    if os.path.exists(d + "/vs.metal"):
        cmd += ["--vs", d + "/vs.metal"]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=WORK).stdout)
    best = {}
    for v in out.get("variants", []):
        for i, a, b in v.get("cmp", []):
            dist = abs(a - b)
            best[i] = min(best.get(i, 1 << 33), dist)
    return n, best

names = json.load(open(os.path.join(HERE, "cmplog_names.json")))
cache = os.path.join(HERE, "cmp_dist_cache.json")
dist = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(cache)).items()} if os.path.exists(cache) else {}
todo = [n for n in names if n not in dist]
with ThreadPoolExecutor(2) as ex:
    dist.update(dict(ex.map(run, todo)))
json.dump(dist, open(cache, "w"))
rows = [json.loads(l) for l in open(os.path.join(HERE, "rows.jsonl"))]
out = []
CL = re.compile(r"sbx_cl\((\d+)u, ")
for r in rows:
    if r["group"] not in ("unobservable", "unresolved") or r["role"] != "boundary" or r["name"] not in dist:
        continue
    a = open(P + r["name"] + "/ref.metal").read().split("\n")
    b = open(P + r["name"] + "/cmplog.metal").read().split("\n")
    off = [i for i, l in enumerate(b) if " main0(" in l][0] - [i for i, l in enumerate(a) if " main0(" in l][0]
    cl = b[r["line"] - 1 + off]
    ids = []
    for m in CL.finditer(cl):
        # args of this sbx_cl call
        k, depth = m.end(), 1
        while depth and k < len(cl):
            depth += cl[k] == "("
            depth -= cl[k] == ")"
            k += 1
        if re.search(r"\b" + r["orig"] + r"u\b", cl[m.end():k]):
            ids.append(int(m.group(1)))
    ds = [dist[r["name"]].get(i) for i in ids]
    known = [x for x in ds if x is not None]
    out.append({"name": r["name"], "line": r["line"], "orig": r["orig"], "group": r["group"], "cause": r["cause"],
                "family": r["family"], "cmp_ids": ids, "min_distance": min(known) if known else None,
                "ran": bool(known)})
json.dump(out, sys.stdout, indent=1)
