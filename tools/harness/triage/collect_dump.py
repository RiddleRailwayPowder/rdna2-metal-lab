"""Output degeneracy per pixel shader: distinct output hashes across inputs.json variants (from cov/<key>.json) and
distinct pixel values in dumped colour targets. The harness dumps only variant 0 and only with a B shader, so each
dumped variant is its own run (--b ref.metal, single-variant inputs file). Up to 3 variants per shader, chosen
for distinct colour hashes. At most 2 harness processes at a time; cwd = harness_work (fssm path).

    python3 -I collect_dump.py <names file>     # one shader name per line
Output: dump/<key>/v<i>/a_color<k>.bin (32x32 RGBA float32), degeneracy.json
"""
import os  # noqa: E401
import json, os, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GAME_ROOT", ".")
WORK = ROOT + "/harness_work"
PREP = WORK + "/prep_h"
H = ROOT + "/tools/harness/harness"


def key(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def jobs_for(name):
    cov = json.load(open(os.path.join(HERE, "cov", key(name) + ".json")))
    variants = json.load(open(os.path.join(HERE, "cov", key(name) + ".inputs.json")))
    seen, pick = set(), []
    for row in cov.get("variants", []):
        h = tuple(sorted((k, v) for k, v in row["hashes"].items() if k.startswith("color")))
        if h not in seen and len(pick) < 3:
            seen.add(h)
            pick.append(row["i"])
    return [(name, i, variants[i]) for i in pick]


def run(job):
    name, i, variant = job
    d = os.path.join(PREP, name)
    od = os.path.join(HERE, "dump", key(name), f"v{i}")
    os.makedirs(od, exist_ok=True)
    inp = os.path.join(od, "inputs.json")
    json.dump([variant], open(inp, "w"))
    cmd = [H, "--meta", os.path.join(d, "meta.json"), "--a", os.path.join(d, "ref.metal"), "--b",
           os.path.join(d, "ref.metal"), "--vs", os.path.join(d, "vs.metal"), "--inputs", inp, "--dump", od]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=WORK)
    return name, i, p.returncode


def pixels(path):
    b = open(path, "rb").read()
    n = len(b) // 16
    return [struct.unpack_from("<4I", b, 16 * k) for k in range(n)]


def summarize(names):
    out = {}
    for name in names:
        cov = json.load(open(os.path.join(HERE, "cov", key(name) + ".json")))
        hashes = {}
        written = set()
        for row in cov.get("variants", []):
            written |= set(row.get("written", []))
            for k, v in row["hashes"].items():
                hashes.setdefault(k, set()).add(v)
        rec = {"variants": len(cov.get("variants", [])), "written": sorted(written),
               "distinct_hashes": {k: len(v) for k, v in hashes.items()}, "pixels": {}}
        base = os.path.join(HERE, "dump", key(name))
        if os.path.isdir(base):
            per = {}
            for v in sorted(os.listdir(base)):
                for f in os.listdir(os.path.join(base, v)):
                    if f.startswith("a_color") and f.endswith(".bin"):
                        px = pixels(os.path.join(base, v, f))
                        per.setdefault(f[2:-4], []).append(px)
            for o, lists in per.items():
                allpx = [p for l in lists for p in l]
                rec["pixels"][o] = {"dumped_variants": len(lists),
                                    "distinct_rgba": len(set(allpx)),
                                    "distinct_per_variant": [len(set(l)) for l in lists],
                                    "distinct_per_channel": [len(set(p[c] for p in allpx)) for c in range(4)]}
        out[name] = rec
    return out


if __name__ == "__main__":
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    ps = [n for n in names if os.path.exists(os.path.join(PREP, n, "vs.metal"))]
    jobs = [j for n in ps for j in jobs_for(n)]
    with ThreadPoolExecutor(2) as ex:
        for name, i, rc in ex.map(run, jobs):
            print(rc, name, i, flush=True)
    json.dump(summarize(names), open(os.path.join(HERE, "degeneracy.json"), "w"), indent=1)
