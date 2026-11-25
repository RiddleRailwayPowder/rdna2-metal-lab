"""Tables for the triage report from rows.jsonl (classify.py) and degeneracy.json (collect_dump.py).
    python3 -I summary.py > summary.txt
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.environ.get("GAME_ROOT", ".") + "/harness_work/prep_h"
rows = [json.loads(l) for l in open(os.path.join(HERE, "rows.jsonl"))]
deg = json.load(open(os.path.join(HERE, "degeneracy.json"))) if os.path.exists(os.path.join(HERE, "degeneracy.json")) else {}
CAUSES = ["nondeterministic", "no_output", "site_unreached", "site_in_dead_arm", "dead_value", "consumer_unhit", "masked_by_select",
          "control_only", "reaches_output"]


def degenerate(name):
    d = deg.get(name)
    if not d or not d["pixels"]:
        return None
    vals = [p["distinct_rgba"] for o, p in d["pixels"].items()]
    return max(vals) <= 16


def table(rs, by):
    c = Counter((r["cause"], by(r)) for r in rs)
    cols = sorted(set(by(r) for r in rs), key=str)
    print("cause".ljust(20), *[str(x).rjust(14) for x in cols], "total".rjust(7))
    for cause in CAUSES:
        n = [c[(cause, x)] for x in cols]
        if sum(n):
            print(cause.ljust(20), *[str(x).rjust(14) for x in n], str(sum(n)).rjust(7))
    print("total".ljust(20), *[str(sum(c[(k, x)] for k in CAUSES)).rjust(14) for x in cols], str(len(rs)).rjust(7))


tgt = [r for r in rows if r["group"] in ("unobservable", "unresolved")]
print("== target rows by cause x resolution")
table(tgt, lambda r: r["group"])
print("\n== target rows by cause x stage")
table(tgt, lambda r: r["stage"])
print("\n== target rows by cause x kind")
table(tgt, lambda r: r["kind"])
print("\n== control groups by cause")
table([r for r in rows if r["group"] in ("killed", "killed_by_hunt", "equivalent")], lambda r: r["group"])
print("\n== pending survivors (resolver not reached yet) by cause")
table([r for r in rows if r["group"] == "pending"], lambda r: r["stage"])

print("\n== family x role, target rows, per cause")
for cause in CAUSES:
    rs = [r for r in tgt if r["cause"] == cause]
    if not rs:
        continue
    c = Counter((r["family"], r["role"], r["group"][:5]) for r in rs)
    print(cause, dict(sorted(c.items(), key=lambda x: -x[1])))

print("\n== reaches_output: degeneracy of the shader's outputs (dumped variants, max distinct RGBA <= 16)")
rs = [r for r in tgt if r["cause"] == "reaches_output"]
print(Counter((degenerate(r["name"]), r["group"][:5]) for r in rs))
print("out sinks:", Counter(tuple(r["out_sinks_sel"]) != () for r in rs))

def sub(r):
    if r["role"] == "boundary":
        return "exact_boundary"
    if r["role"] == "ulp":
        return "ulp_constant"
    if r["family"] in ("f16_pack", "f16_unpack", "denorm_flush", "nan_inf_convert", "float_exponent_bits"):
        return "float_regime"
    if r["family"] in ("descriptor_bits", "wave_lane_mask", "u24_mul", "int_other"):
        return "int_bits_regime"
    return "float_op"


print("\n== reaches_output sub-causes x resolution x degenerate")
rs = [r for r in tgt if r["cause"] == "reaches_output"]
print(Counter((sub(r), r["group"][:5]) for r in rs))
print(Counter((sub(r), degenerate(r["name"])) for r in rs))
print("all target rows by family/role idiom (any cause):", Counter(r["family"] for r in tgt))
print("pending reaches_output sub:", Counter(sub(r) for r in rows if r["group"] == "pending" and r["cause"] == "reaches_output"))
for sc in ["exact_boundary", "ulp_constant", "float_regime", "int_bits_regime", "float_op"]:
    k = 0
    seen = set()
    for r in rs:
        if sub(r) == sc and r["name"] not in seen and k < 3:
            seen.add(r["name"]); k += 1
            src = open(os.path.join(PREP, r["name"], "ref.metal")).read().split("\n")[r["line"] - 1].strip()
            print(sc, "|", r["name"], r["line"], r["orig"], "->", r["rep"], r["group"], r["family"], "|", src[:200])

print("\n== control_only: discard decisions")
print(Counter((r["discard_ctrl"], r["group"][:5]) for r in tgt if r["cause"] == "control_only"))

print("\n== dead_store.py agreement (dead_value rows)")
print(Counter((r["cause"] == "dead_value", r["dead_store"]) for r in rows))

print("\n== top shaders by target rows")
byn = defaultdict(list)
for r in tgt:
    byn[r["name"]].append(r)
for n, rs in sorted(byn.items(), key=lambda x: -len(x[1]))[:15]:
    d = deg.get(n, {})
    px = {o: p["distinct_rgba"] for o, p in d.get("pixels", {}).items()}
    print(len(rs), rs[0]["stage"], n, dict(Counter(r["cause"] for r in rs)), "px", px,
          "hashes", d.get("distinct_hashes"))

print("\n== examples")
for cause in CAUSES:
    rs = [r for r in tgt if r["cause"] == cause]
    seen = set()
    k = 0
    for r in rs:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        src = open(os.path.join(PREP, r["name"], "ref.metal")).read().split("\n")[r["line"] - 1].strip()
        print(cause, "|", r["name"], r["line"], r["kind"], r["orig"], "->", r["rep"], r["group"], r["family"], r["role"],
              "| sinks_blk", r["sinks_blk"][:4], "sel", r["sinks_sel"][:4], "|", src[:220])
        k += 1
        if k == 4:
            break
