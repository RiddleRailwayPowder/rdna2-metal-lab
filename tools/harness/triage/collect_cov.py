"""Union coverage per shader over its inputs.json: one harness run per shader (A only, --coverage 1, --all 1), on a
select-probe copy of ref.metal (selprobe.py) so the same run also reports which arm of every ternary was taken.
Block and lane-guard ids are the same as ref.metal's; the probe only adds coverage stores. Falls back to ref.metal
if the probe doesn't compile. Runs with cwd = harness_work so meta.json's relative fssm path resolves, as in
run_set/resolve_survivors. At most 2 harness processes at a time.

    python3 -I collect_cov.py [--validate N]   # --validate: also run plain ref.metal on N shaders and compare
Output: cov/<key>.json (harness JSON + "shader", "probe": bool), probe/<key>/arms.json
"""
import os  # noqa: E401
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selprobe  # noqa: E402

ROOT = os.environ.get("GAME_ROOT", ".")
WORK = ROOT + "/harness_work"
PREP = WORK + "/prep_h"
H = ROOT + "/tools/harness/harness"
OUT = os.path.join(HERE, "cov")


def targets():
    rows = [json.loads(l) for l in open(WORK + "/run_150_v3.jsonl")]
    return [r for r in rows if r.get("status") == "ok" and any(m["verdict"] == "survived" for m in r.get("mutants", []))]


def key(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def harness(meta, a, inputs, stage, d):
    cmd = [H, "--meta", meta, "--a", a, "--b", "none", "--inputs", inputs, "--all", "1", "--coverage", "1"]
    if stage == "ps":
        cmd += ["--vs", os.path.join(d, "vs.metal")]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=WORK)
        return json.loads(p.stdout)
    except Exception as e:
        return {"status": "error", "reason": repr(e)}


def run(r, validate=False):
    name = r["name"]
    d = os.path.join(PREP, name)
    k = key(name)
    out = os.path.join(OUT, k + ".json")
    if os.path.exists(out) and not validate:
        return name, "cached"
    inputs = os.path.join(OUT, k + ".inputs.json")
    if not os.path.exists(inputs):
        # snapshot: the resolver may append to inputs.json while we run
        open(inputs, "w").write(open(os.path.join(d, "inputs.json")).read())
    if validate:
        res = harness(os.path.join(d, "meta.json"), os.path.join(d, "ref.metal"), inputs, r["stage"], d)
        json.dump(res, open(os.path.join(OUT, k + ".ref.json"), "w"))
        return name, res.get("status")
    try:
        pm, pmeta, _ = selprobe.instrument(os.path.join(d, "ref.metal"), os.path.join(d, "meta.json"),
                                           os.path.join(HERE, "probe", k))
        res = harness(pmeta, pm, inputs, r["stage"], d)
        res["probe"] = True
    except Exception as e:
        res = {"status": "probe_error", "reason": repr(e)}
    if res.get("status") != "ok":
        reason = str(res.get("reason") or res.get("status"))[:300]
        res = harness(os.path.join(d, "meta.json"), os.path.join(d, "ref.metal"), inputs, r["stage"], d)
        res["probe"] = False
        res["probe_failure"] = reason
    res["shader"] = name
    json.dump(res, open(out, "w"))
    return name, f"{res.get('status')} probe={res.get('probe')}"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ts = targets()
    if "--validate" in sys.argv:
        n = int(sys.argv[sys.argv.index("--validate") + 1])
        ts = ts[::max(1, len(ts) // n)][:n]
        with ThreadPoolExecutor(2) as ex:
            for name, st in ex.map(lambda r: run(r, True), ts):
                print("validate", st, name, flush=True)
        sys.exit(0)
    with ThreadPoolExecutor(2) as ex:
        for name, st in ex.map(run, ts):
            print(st, name, flush=True)
