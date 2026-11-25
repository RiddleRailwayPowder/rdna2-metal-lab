"""Classify surviving mutants (unobservable / unresolved rows of the resolver) into causes, with a control group of
killed mutants run through the same ladder.

Inputs: a snapshot of run_150_v3.jsonl.resolved.jsonl (deduped), run_150_v3.jsonl, cov/<key>.json and
probe/<key>/arms.json from collect_cov.py.

Ladder (first match wins), per mutant:
  nondeterministic   shader not deterministic in run_set (racing)
  no_output          the shader wrote no output (colour, depth, texture, memory) on any input variant
  site_unreached     the site's block or its lane guard never fired over inputs.json
  site_in_dead_arm   the mutated token sits in a ternary arm never taken (or a ternary never evaluated)
  dead_value         static def-use: the value reaches no sink on any path (liveness proof should close it)
  consumer_unhit     forward def-use restricted to executed blocks/guards reaches no sink; the unrestricted one does
  masked_by_select   ... also restricted to ternary arms that were taken: no sink
  control_only       the only sinks left are branch/loop/switch conditions (or a discard decision)
  reaches_output     a data sink (colour/depth output, store, texture write, atomic) is reachable through executed
                     code and taken arms: the value is masked arithmetically, needs an exact value, or the
                     outputs are degenerate (sub-split by token idiom and output diversity)

    python3 -I classify.py <resolved snapshot> > rows.jsonl
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taint  # noqa: E402
import dead_store as ds  # noqa: E402  (taint put tools/harness on sys.path)

WORK = os.environ.get("GAME_ROOT", ".") + "/harness_work"
PREP = WORK + "/prep_h"


def key(name):
    return "".join(c if c.isalnum() else "_" for c in name)


# comparison operators only: `>>` / `<<` shifts and template brackets are not comparisons
CMP_BEFORE = re.compile(r"(<=|>=|==|!=|(?<![<>])<|(?<![<>-])>)\s*(?:as_type<float>\()?$")
CMP_AFTER = re.compile(r"^u?\s*(<=|>=|==|!=|<(?![<=])|>(?![>=]))")
CMP_AFTER_F = re.compile(r"^u?\)\s*(<=|>=|==|!=|<(?![<=])|>(?![>=]))")   # only after `as_type<float>(K`


def token_context(line, col, orig, kind):
    """(family, role): family = float/bit idiom the token belongs to; role = boundary | regime | ulp | op."""
    before = line[:col].rstrip()
    after = line[col + len(orig):]
    win = line[max(0, col - 160): col + 160]
    if kind == "float_op":
        args = line[col: col + 200]
        if "864026624u" in args[:120] or ("& 1023u" in args[:60]):
            fam = "f16_unpack"
        elif orig == "fma(" and re.match(r"fma\(as_type<float>\(\d+u\)|fma\([^,]+, as_type<float>\(\d+u\), as_type<float>\(\d+u\)\)", args):
            fam = "const_fma"
        else:
            fam = "float_op"
        return fam, "op"
    # integer constant
    is_cmp = (bool(CMP_BEFORE.search(before)) or bool(CMP_AFTER.match(after))
              or (before.endswith("as_type<float>(") and bool(CMP_AFTER_F.match(after))))
    in_float_lit = before.endswith("as_type<float>(")
    fam = None
    if re.search(r">> 13u\) & 1023u\) \| 32256u", line) or ("947912704u" in line and "31743u" in line):
        fam = "f16_pack"
    elif "& 31744u) == 0u" in win or "864026624u" in win or "as_type<half2>" in win:
        fam = "f16_unpack"
    elif "abs(" in win and "< as_type<float>(8388608u)" in win and orig in ("8388608", "2147483648", "0"):
        fam = "denorm_flush"
    elif re.search(r"== 0\.0\) \|\| \(_\d+ == 0\.0\)\) \? 0\.0|== 0\.0\) \? 0\.0 : spvFMul", win):
        fam = "legacy_mul"
    elif "isnan(" in win or "2139095041u" in win or re.search(r"2147483647u\) > 2139095040u", win) or "1333788672u" in win:
        fam = "nan_inf_convert"
    elif ">> 23u" in win or "126u -" in win or "8388607u" in win or "| 8388608u" in win:
        fam = "float_exponent_bits"
    elif "16777215u" in win:
        fam = "u24_mul"
    elif re.search(r">= 32u\) \?|& 31u\)\) & 1u|gl_SubgroupInvocationID|spvSubgroup|spvQuad|simd_|gl_LocalInvocationID", win):
        fam = "wave_lane_mask"
    elif re.search(r"extract_bits\(|4294967292u|4294967295u, 4294967295u|& 65535u", win):
        fam = "descriptor_bits"
    elif in_float_lit:
        fam = "float_constant"
    else:
        fam = "int_other"
    if is_cmp:
        role = "boundary"
    elif in_float_lit and fam in ("float_constant", "legacy_mul", "int_other", "const_fma"):
        role = "ulp"
    else:
        role = "regime"
    return fam, role


def effective_lines(path, arms, hits):
    lines = open(path).read().split("\n")
    eff = {}
    for li, recs in arms.items():
        li = int(li)
        s = list(lines[li])
        for t, f, (a0, a1), (b0, b1) in recs:
            if t not in hits:
                s[a0:a1] = " " * (a1 - a0)
            if f not in hits:
                s[b0:b1] = " " * (b1 - b0)
        eff[li] = "".join(s).strip()
    return eff, lines


def classify(row, ctx):
    name = row["name"]
    path = os.path.join(PREP, name, "ref.metal")
    hits, arms, eff, lines, nondet, no_output = ctx
    out = {k: row.get(k) for k in ("name", "kind", "line", "col", "orig", "rep", "resolution", "verdict", "why")}
    line = lines[row["line"] - 1]
    assert line[row["col"]: row["col"] + len(row["orig"])] == row["orig"], (name, row["line"])
    out["family"], out["role"] = token_context(line, row["col"], row["orig"], row["kind"])
    full = taint.taint(path, row["line"])
    blk = taint.taint(path, row["line"], hits)
    sel = taint.taint(path, row["line"], hits, eff)
    out["site_block"], out["site_guard"] = full["site_block"], full["site_guard"]
    # arms containing the token
    dead_arm = None
    for t, f, (a0, a1), (b0, b1) in arms.get(str(row["line"] - 1), []):
        c = row["col"]
        if a0 <= c < a1 and t not in hits:
            dead_arm = ("true", t, f)
        elif b0 <= c < b1 and f not in hits:
            dead_arm = ("false", t, f)
        if dead_arm:
            break
    kinds = lambda r: sorted(Counter(k for k, _ in r["sinks"]).items())
    out["sinks_full"], out["sinks_blk"], out["sinks_sel"] = kinds(full), kinds(blk), kinds(sel)
    data_sel = [(k, l) for k, l in sel["sinks"] if k not in ("control", "return")]
    ctrl_sel = [(k, l) for k, l in sel["sinks"] if k == "control"]
    out["discard_ctrl"] = any(l in sel["discard_ifs"] for _, l in ctrl_sel)
    out["data_sink_lines_sel"] = sorted(set(l + 1 for _, l in data_sel))[:6]
    out["out_sinks_sel"] = sorted(set(lines[l].strip()[:40] for k, l in data_sel if k == "out"))
    try:
        dead, why = ds.prove_dead(open(path).read(), row, cache=ctx_cache.setdefault(name, {}))
    except Exception as e:  # pragma: no cover
        dead, why = None, repr(e)
    out["dead_store"] = dead
    label_site = bool(re.match(r"(case .*|default):$|\} while \(", line.strip()))
    if label_site:                       # switch label / do-while tail: a control site the tree walk doesn't visit
        for r_ in (full, blk, sel):
            r_["site_reached"] = True
            r_["sinks"] = r_["sinks"] + [("control", row["line"] - 1)]
    if nondet:
        cause = "nondeterministic"
    elif no_output:
        cause = "no_output"
    elif full["site_block"] not in hits or not blk["site_reached"]:
        cause = "site_unreached"
    elif dead_arm:
        cause = "site_in_dead_arm"
        out["dead_arm"] = dead_arm
    elif not full["sinks"]:
        cause = "dead_value"
    elif not blk["sinks"]:
        cause = "consumer_unhit"
    elif not sel["sinks"]:
        cause = "masked_by_select"
    elif not data_sel:
        cause = "control_only"
    else:
        cause = "reaches_output"
    out["cause"] = cause
    return out


ctx_cache = {}
_shader = {}


def shader_ctx(name, nondet):
    if name not in _shader:
        cov = json.load(open(os.path.join(HERE, "cov", key(name) + ".json")))
        hits = set(cov["cov_hits"])
        arms = json.load(open(os.path.join(HERE, "probe", key(name), "arms.json")))
        eff, lines = effective_lines(os.path.join(PREP, name, "ref.metal"), arms, hits)
        no_output = not any(v.get("written") for v in cov.get("variants", []))
        _shader[name] = (hits, arms, eff, lines, nondet, no_output)
    return _shader[name]


def main(snapshot):
    run = {json.loads(l)["name"]: json.loads(l) for l in open(WORK + "/run_150_v3.jsonl")}
    nondet = {n for n, r in run.items() if r.get("deterministic") is False}
    seen = set()
    rows = []
    for l in open(snapshot):
        r = json.loads(l)
        k = (r["name"], r["line"], r["col"], r["rep"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    covered = {n for n in run if os.path.exists(os.path.join(HERE, "cov", key(n) + ".json"))}
    resolved_names = {r["name"] for r in rows}
    out = []
    for r in rows:
        if r["resolution"] not in ("unobservable", "unresolved", "killed_by_hunt", "equivalent"):
            continue
        c = classify(r, shader_ctx(r["name"], r["name"] in nondet))
        c["group"] = r["resolution"]
        c["stage"] = run[r["name"]]["stage"]
        out.append(c)
    # control group: mutants run_set killed, in covered shaders; plus survivors not yet resolved (forecast)
    for n, rr in run.items():
        if n not in covered:
            continue
        for m in rr.get("mutants", []):
            if m["verdict"] == "killed":
                g = "killed"
            elif m["verdict"] == "survived" and n not in resolved_names:
                g = "pending"
            else:
                continue
            m = dict(m, name=n, resolution=g)
            c = classify(m, shader_ctx(n, n in nondet))
            c["group"] = g
            c["stage"] = rr["stage"]
            out.append(c)
    for c in out:
        print(json.dumps(c))


if __name__ == "__main__":
    main(sys.argv[1])
