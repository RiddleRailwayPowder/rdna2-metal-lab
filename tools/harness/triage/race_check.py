"""Race check: rerun plain ref.metal (A only) on every compute shader with survivors and compare per-variant output
hashes with the probe run in cov/. A mismatch = output depends on scheduling (or on the probe). 2 processes max.
    python3 -I race_check.py > race.json
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_cov as cc

def one(r):
    k = cc.key(r["name"])
    d = os.path.join(cc.PREP, r["name"])
    ref = cc.harness(os.path.join(d, "meta.json"), os.path.join(d, "ref.metal"), os.path.join(cc.OUT, k + ".inputs.json"), r["stage"], d)
    pr = json.load(open(os.path.join(cc.OUT, k + ".json")))
    if ref.get("status") != "ok":
        return r["name"], {"status": ref.get("status")}
    diff = [i for i, (a, b) in enumerate(zip(ref["variants"], pr["variants"])) if a["hashes"] != b["hashes"]]
    outs = sorted({o for i in diff for o in ref["variants"][i]["hashes"] if ref["variants"][i]["hashes"][o] != pr["variants"][i]["hashes"].get(o)})
    return r["name"], {"variants": len(ref["variants"]), "differing_variants": diff, "outputs": outs}

ts = [r for r in cc.targets() if r["stage"] == "cs"]
with ThreadPoolExecutor(2) as ex:
    res = dict(ex.map(one, ts))
json.dump(res, sys.stdout, indent=1)
