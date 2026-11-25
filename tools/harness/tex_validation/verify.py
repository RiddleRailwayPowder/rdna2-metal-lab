#!/usr/bin/env python3
"""Verification runs for harness_tex vs harness (at most 2 harness processes at a time).

Per shader: old x2 (noise floor), new x2 (cross-process determinism), new A-vs-A in one process,
all with --b none/--all 1/--coverage 1 over the shader's inputs.json. Results: runs/<tag>/<shader>.json.
"""
import os  # noqa: E401
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("GAME_ROOT", ".")
PREP = os.path.dirname(os.path.abspath(__file__)) + "/snap"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/runs"
SHADERS = [l.strip() for l in open(os.environ.get("SHADERS", os.path.dirname(os.path.abspath(__file__)) + "/shaders.txt")) if l.strip()]
RUNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["old1", "old2", "new1", "new2", "newab"]


def cmd(tag, s):
    d = f"{PREP}/{s}"
    binary = ROOT + "/tools/harness/" + ("harness" if "old" in tag else "harness_tex")
    b = d + "/ref.metal" if tag.endswith("ab") else "none"
    c = [binary, "--meta", d + "/meta.json", "--a", d + "/ref.metal", "--b", b, "--inputs", d + "/inputs.json",
         "--all", "1", "--coverage", "1", "--compile-timeout", "120"]
    meta = json.load(open(d + "/meta.json"))
    if meta.get("stage") == "ps":
        c += ["--vs", d + "/vs.metal"]
    return c


def run(job):
    tag, s = job
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    out = f"{OUT}/{tag}/{s}.json"
    env = dict(os.environ)
    if tag.startswith("val"):
        env["MTL_SHADER_VALIDATION"] = "1"
        env["MTL_DEBUG_LAYER"] = "1"
    t = time.time()
    p = subprocess.run(cmd(tag, s), capture_output=True, text=True, env=env, timeout=1800, cwd=ROOT + "/harness_work")
    with open(out, "w") as f:
        f.write(p.stdout)
    with open(out + ".err", "w") as f:
        f.write(p.stderr)
    return f"{tag} {s} exit {p.returncode} {time.time() - t:.1f}s"


jobs = [(t, s) for s in SHADERS for t in RUNS]
with ThreadPoolExecutor(max_workers=2) as ex:
    for line in ex.map(run, jobs):
        print(line, flush=True)
