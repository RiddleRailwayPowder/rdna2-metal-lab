"""Re-run every killed_by_hunt mutant (snapshot) against ref.metal on the shader's current inputs.json (--all 1).
A kill that doesn't reproduce is flaky or its input wasn't saved. 2 processes max, cwd = harness_work.
    python3 -I recheck_hunt.py resolved_snapshot.jsonl > recheck_hunt.json
"""
import os  # noqa: E401
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("GAME_ROOT", ".") + "/harness_work"
H = os.environ.get("GAME_ROOT", ".") + "/tools/harness/harness"
sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/harness")
import resolve_survivors as rs  # mutant_text only (read-only use)

def one(r):
    d = os.path.join(WORK, "prep_h", r["name"])
    msl = open(os.path.join(d, "ref.metal")).read()
    p = os.path.join(HERE, "recheck", f"{abs(hash((r['name'], r['line'], r['col'])))}.metal")
    open(p, "w").write(rs.mutant_text(msl, r))
    cmd = [H, "--meta", os.path.join(d, "meta.json"), "--a", os.path.join(d, "ref.metal"), "--b", p,
           "--inputs", os.path.join(d, "inputs.json"), "--all", "1"]
    if os.path.exists(os.path.join(d, "vs.metal")):
        cmd += ["--vs", os.path.join(d, "vs.metal")]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=WORK).stdout
    try:
        j = json.loads(out)
        return {"name": r["name"], "line": r["line"], "col": r["col"], "status": j.get("status"), "equal": j.get("equal"),
                "diff_variants": [v["i"] for v in j.get("variants", []) if v.get("equal") is False]}
    except Exception as e:
        return {"name": r["name"], "line": r["line"], "error": repr(e)}

rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r["resolution"] == "killed_by_hunt"]
os.makedirs(os.path.join(HERE, "recheck"), exist_ok=True)
with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(one, rows))
json.dump(res, sys.stdout, indent=1)
