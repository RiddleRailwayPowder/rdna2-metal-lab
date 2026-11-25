#!/usr/bin/env python3
"""Compare runs/<tag>/<shader>.json between harness (old) and harness_tex (new)."""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/snap"
HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = [l.strip() for l in open(HERE + "/shaders.txt") if l.strip()]


def load(tag, s):
    p = f"{HERE}/runs/{tag}/{s}.json"
    try:
        return json.load(open(p))
    except Exception as e:
        return {"status": f"unreadable ({e})"}


def rowkey(v, outputs_only=False):
    k = (tuple(sorted(v["hashes"].items())), tuple(v["written"]))
    return k if outputs_only else k + (tuple(v["hits"]),)


def distinct(r):
    """Distinct output hashes across variants, summed over outputs (memory included) and the same without memory."""
    seen = {}
    for v in r.get("variants", []):
        for n, h in v["hashes"].items():
            seen.setdefault(n, set()).add(h)
    return sum(len(x) for x in seen.values()), sum(len(x) for n, x in seen.items() if n != "memory")


def hits(r):
    u = set()
    for v in r.get("variants", []):
        u |= set(v["hits"])
    return u


rows = []
for s in SHADERS:
    inputs = json.load(open(f"{ROOT}/{s}/inputs.json"))
    hv = [bool(v.get("hash")) for v in inputs]
    o1, o2, n1, n2, ab = (load(t, s) for t in ("old1", "old2", "new1", "new2", "newab"))
    st = {t: r.get("status") for t, r in zip(("old1", "old2", "new1", "new2", "newab"), (o1, o2, n1, n2, ab))}
    bad = {t: x for t, x in st.items() if x != "ok"}
    V = lambda r: r.get("variants", [])
    noise_old = sum(rowkey(a) != rowkey(b) for a, b in zip(V(o1), V(o2)))
    noise_new = sum(rowkey(a) != rowkey(b) for a, b in zip(V(n1), V(n2)))
    ab_unequal = sum(1 for v in V(ab) if not v.get("equal", False))
    # variants without hash entries must match old exactly (outputs, written, coverage ids)
    nohash_diff = sum(rowkey(a) != rowkey(b) for a, b, h in zip(V(o1), V(n1), hv) if not h)
    hash_out_changed = sum(rowkey(a, True) != rowkey(b, True) for a, b, h in zip(V(o1), V(n1), hv) if h)
    hash_hits_changed = sum(tuple(a["hits"]) != tuple(b["hits"]) for a, b, h in zip(V(o1), V(n1), hv) if h)
    new_outputs = sorted({n for v in V(n1) for n in v["hashes"]} - {n for v in V(o1) for n in v["hashes"]})
    ho, hn = hits(o1), hits(n1)
    rows.append(dict(shader=s, stage=json.load(open(f"{ROOT}/{s}/meta.json")).get("stage"), variants=len(hv),
                     hash_variants=sum(hv), bad=bad, noise_old=noise_old, noise_new=noise_new, ab_unequal=ab_unequal,
                     nohash_diff=nohash_diff, hash_out_changed=hash_out_changed, hash_hits_changed=hash_hits_changed,
                     distinct_old=distinct(o1), distinct_new=distinct(n1), cov_old=len(ho), cov_new=len(hn),
                     cov_gained=len(hn - ho), cov_lost=len(ho - hn), new_outputs=new_outputs,
                     loop_old=sum(v["loop_guard"] for v in V(o1)), loop_new=sum(v["loop_guard"] for v in V(n1))))
json.dump(rows, open(HERE + "/compare.json", "w"), indent=1)
hdr = "shader stage var hashvar noiseO noiseN abNeq nohashDiff outChg hitChg distinctO distinctN covO covN +cov -cov loopO loopN bad newOut"
print(hdr)
for r in rows:
    print(r["shader"][:48], r["stage"], r["variants"], r["hash_variants"], r["noise_old"], r["noise_new"], r["ab_unequal"],
          r["nohash_diff"], r["hash_out_changed"], r["hash_hits_changed"], r["distinct_old"], r["distinct_new"],
          r["cov_old"], r["cov_new"], r["cov_gained"], r["cov_lost"], r["loop_old"], r["loop_new"], r["bad"] or "", r["new_outputs"])
