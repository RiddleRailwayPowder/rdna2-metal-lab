"""Mac census: translated SPIR-V -> (Palmetto set-0 fix) -> SPIRV-Cross MSL -> Apple metal compiler (AIR).
Usage: python3 -I metal_census.py <spirv dir> <work dir> <out.jsonl> [--jobs N]"""
import json, os, subprocess, sys, tempfile, re
from concurrent.futures import ThreadPoolExecutor
SET0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream", "palmetto", "tools", "spirv_metal_set0.py")
src, work, out = sys.argv[1], sys.argv[2], sys.argv[3]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
os.makedirs(work, exist_ok=True)
def first_err(t):
    for line in t.splitlines():
        if "error" in line.lower():
            return re.sub(r"^.*?error:\s*", "", line).strip()[:200]
    return (t.strip().splitlines() or [""])[-1][:200]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402
LOG = runlog.setup(out, "metal_census")


def run(cmd):
    """subprocess result shape the steps below expect; every call and failure goes to the log."""
    r = runlog.run(cmd, 300, LOG, " ".join(os.path.basename(str(c)) for c in cmd[:2]) + " " + os.path.basename(str(cmd[-1])))
    class P: pass
    p = P(); p.returncode = r["rc"] if r["rc"] is not None else -1
    p.stdout, p.stderr = r.get("stdout") or "", (r.get("stderr") or "") + (" [timed out]" if r["timed_out"] else "")
    return p
def one(name):
    stem = name[:-4]; spv = os.path.join(src, name)
    fixed = os.path.join(work, stem + ".m.spv"); msl = os.path.join(work, stem + ".metal"); air = os.path.join(work, stem + ".air")
    r = {"name": stem}
    try:
        p = run([sys.executable, "-I", SET0, spv, fixed])
        if p.returncode: r.update(step="set0", err=first_err(p.stderr + p.stdout)); return r
        p = run(["spirv-cross", "--msl", "--msl-version", "30100", "--msl-argument-buffers", "--msl-argument-buffer-tier", "1",
                 "--msl-decoration-binding", fixed, "--output", msl])
        if p.returncode: r.update(step="spirv-cross", err=first_err(p.stderr + p.stdout)); return r
        p = run(["xcrun", "metal", "-std=metal3.1", "-c", msl, "-o", air])
        if p.returncode: r.update(step="metal", err=first_err(p.stderr + p.stdout)); return r
        r.update(step="ok", msl_lines=sum(1 for _ in open(msl)), air_bytes=os.path.getsize(air))
    except Exception as e:                # one shader must never stop the census
        r.update(step="exception", err=runlog.exception(e, LOG, f"census {stem}")["exception"])
    finally:
        for f in (fixed, air):
            try: os.remove(f)
            except OSError: pass
    return r
names = sorted(n for n in os.listdir(src) if n.endswith(".spv"))
with open(out, "w") as f, ThreadPoolExecutor(jobs) as ex:
    for i, r in enumerate(ex.map(one, names), 1):
        f.write(json.dumps(r) + "\n")
        if i % 500 == 0: print(f"  {i}/{len(names)}", flush=True)
from collections import Counter
rs = [json.loads(l) for l in open(out)]
print(len(rs), "modules:", dict(Counter(r["step"] for r in rs)))
for (step, err), n in Counter((r["step"], r.get("err", "")) for r in rs if r["step"] != "ok").most_common(15):
    print(f"  {n:6d}  {step}: {err}")
