"""Separate the causes of a low mutation kill rate (run_set.py) on pixel shaders that wrote output.

Per shader:
  positive control: one colour output overwritten with 12345 just before `return out;`. Must be caught,
                    or the harness plumbing is broken.
  swap vs loud:     the same float helper sites mutated two ways: operator swap (spvFMul -> spvFAdd ...) and
                    "loud" (the whole call returns 12345). Loud killed + swap survived = reached code, weak mutant.
                    Both survive = the code is unreached or its result never reaches an output.

Usage: python3 -I diagnose_kill_rate.py <prep dir> <run_set results.jsonl> [--shaders N] [--sites N]
"""
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_set  # noqa: E402
from run_set import harness, main_body  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlog  # noqa: E402

LOUD = "\ntemplate<typename T, typename U> static inline T spvLoud(T a, U b) { return T(12345.0); }\n"
SWAP = {"spvFMul(": "spvFAdd(", "spvFAdd(": "spvFSub(", "spvFSub(": "spvFAdd("}


def variants(msl, rng, n_sites):
    lo, _ = main_body(msl)
    body = msl[lo:]
    out = []
    ret = body.rfind("return out;")
    m = re.search(r"out\.(m_\d+)\s*=", body)
    if ret > 0 and m:
        line = f"out.{m.group(1)} = decltype(out.{m.group(1)})(12345);\n    "
        out.append(("positive", None, msl[:lo + ret] + line + msl[lo + ret:]))
    sites = [x for x in re.finditer(r"spvF(Mul|Add|Sub)\(", body)]
    rng.shuffle(sites)
    for i, s in enumerate(sites[:n_sites]):
        a, b = lo + s.start(), lo + s.end()
        out.append(("swap", i, msl[:a] + SWAP[s.group(0)] + msl[b:]))
        loud = msl[:a] + "spvLoud(" + msl[b:]
        out.append(("loud", i, loud.replace("using namespace metal;\n", "using namespace metal;\n" + LOUD, 1)))
    return out


def one(d, n_sites):
    ref = os.path.join(d, "ref.metal")
    msl = open(ref).read()
    rng = random.Random(os.path.basename(d))
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, (kind, site, text) in enumerate(variants(msl, rng, n_sites)):
            p = os.path.join(tmp, f"v{k}.metal")
            open(p, "w").write(text)
            r = harness(d, ref, p)
            st = r.get("status")
            if st != "ok":
                run_set.LOG.info("diagnose %s %s site %s: %s in %s: %s", os.path.basename(d), kind, site, st,
                                 r.get("phase"), run_set.first_error(r))
            v = "invalid" if st in ("compile_b", "pipeline_b") else ("survived" if st == "ok" and r.get("equal") else "killed")
            res.append((kind, site, v))
    return os.path.basename(d), res


def main(argv):
    prep, results = argv[0], argv[1]
    run_set.LOG = runlog.setup(results + ".diagnose", "diagnose")
    n_sh = int(argv[argv.index("--shaders") + 1]) if "--shaders" in argv else 40
    n_sites = int(argv[argv.index("--sites") + 1]) if "--sites" in argv else 3
    recs = [json.loads(l) for l in open(results)]
    pick = [r["name"] for r in recs if r.get("stage") == "ps" and r.get("status") == "ok"
            and r.get("deterministic") and r.get("exercised")][:n_sh]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda n: one(os.path.join(prep, n), n_sites), pick))
    pos = Counter(v for _, res in out for k, _, v in res if k == "positive")
    pairs = Counter()
    for _, res in out:
        by = {}
        for k, s, v in res:
            if s is not None:
                by.setdefault(s, {})[k] = v
        for s, kv in by.items():
            if "invalid" in kv.values():
                pairs["invalid"] += 1
            else:
                pairs[f"swap {kv.get('swap')}, loud {kv.get('loud')}"] += 1
    print(f"{len(pick)} pixel shaders that wrote output")
    print("positive control:", dict(pos))
    print("same site, swap vs loud:")
    for k, v in pairs.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
