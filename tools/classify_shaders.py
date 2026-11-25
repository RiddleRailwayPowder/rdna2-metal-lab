"""Classify disassembled RDNA2 shaders by how much work a clean Metal translation needs.

Reads the .s files from shader_disasm.py (one per shader) plus the carve index for each shader's stage.
Static only: no GPU, no translation, no game data beyond the shaders themselves.

Tiers (a shader takes the hardest one it hits):
  A  straight-line: no branches, no cross-lane ops, no computed jumps. The deterministic lifting passes
     (registers -> variables, type recovery, descriptor and sampler recognition) should give clean MSL.
  B  branches and loops, no cross-lane ops: the same passes plus structured control flow.
  C  real lane shuffling (DPP row/dpp8, permlane, LDS permutes, writelane): needs a wave-aware mapping to
     simd_shuffle and friends, or 64-lane emulation if the shader is wave64. Lane ids, prefix counts
     and uniform broadcasts (v_mbcnt, v_readfirstlane, v_readlane) map to one Metal built-in each
     and don't raise the tier. Neither do quad ops (DPP quad_perm, QUAD_PERM swizzles): they are
     how AMD code computes derivatives and map to dfdx/dfdy or quad_shuffle.
  D  computed jumps (s_setpc) or dynamic register indexing (movrel): keep the
     emulator-style output, or rewrite by hand.

Usage:
    python3 -I classify_shaders.py <asm dir> <carve index.json> <out dir> [--jobs N]
Writes <out>/classify.jsonl (one record per shader), <out>/families.json and <out>/summary.md.
Logging: <out>/classify.jsonl.log. A shader whose file can't be read or parsed gets status "error" with the
exception and traceback in its record (and in the log); it is left out of the tiers.
"""
import collections
import hashlib
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

# Real lane shuffling: needs simd_shuffle-style mapping (or 64-lane emulation on wave64).
# ds_swizzle is decided per instruction: QUAD_PERM patterns are quad ops (see below).
XLANE_PREFIX = ("v_permlane", "ds_bpermute", "ds_permute", "v_writelane")
# Lane id (v_mbcnt with -1), prefix counts and uniform broadcasts: one Metal built-in each, tier unchanged.
TRIVIAL_LANE_PREFIX = ("v_readfirstlane", "v_mbcnt", "v_readlane")
# s_setpc jumps to a computed address: hard. s_swappc s[4:5] calls the driver's vertex-fetch routine and
# s_getpc + constant addresses data stored after the code: both are resolved statically (flags only).
INDIRECT = {"s_setpc_b64"}
STAGE_RE = re.compile(r"(?:^|[^A-Z])(VS|PS|MS|CS|GS|AS|HS|DS)(?=_|[A-Z])")


def family(name):
    """Shader family: the name without its hash, wrapper suffix, variant number and leading markers."""
    n = re.sub(r"~\d+$", "", name)
    n = re.sub(r"_Wrapped$", "", n)
    n = re.sub(r"_[0-9a-f]{8}$", "", n)
    return n.lstrip("!#'$%&()+,-.;=@[]^`{}0123456789") or n


def branch_offset(operands):
    """Signed simm16 branch offset from an s_branch / s_cbranch_* operand string, or None."""
    m = re.match(r"\s*(-?\d+)", operands)
    if not m:
        return None
    v = int(m.group(1))
    return v - 65536 if v > 32767 else v


def analyse(path):
    try:
        return _analyse(path)
    except Exception as e:                # runs in a worker process: hand the error back as a record
        return {"name": os.path.basename(path)[:-2], "status": "error", "tier": None,
                "exception": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


def _analyse(path):
    name = os.path.basename(path)[:-2]
    ins, mn, h = 0, collections.Counter(), hashlib.sha1()
    branches = loops = endpgm = 0
    xlane = dpp = trivial_lane = quad = 0
    wave64 = False
    invalid = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(";"):
                m = re.search(r"(\d+) invalid words", line)
                if m:
                    invalid = int(m.group(1))
                continue
            if not line.startswith("\t"):
                continue
            text = line.split(";", 1)[0].strip()
            if not text:
                continue
            parts = text.split(None, 1)
            op, rest = parts[0], (parts[1] if len(parts) > 1 else "")
            ins += 1
            mn[op] += 1
            h.update(text.encode())
            h.update(b"\n")
            if op == "s_endpgm":
                endpgm += 1
            elif op == "s_branch" or op.startswith("s_cbranch"):
                branches += 1
                off = branch_offset(rest)
                if off is not None and off < 0:
                    loops += 1
            if op.startswith(XLANE_PREFIX):
                xlane += 1
            elif op.startswith("ds_swizzle"):
                if "QUAD_PERM" in rest:
                    quad += 1
                else:
                    xlane += 1
            elif op.startswith(TRIVIAL_LANE_PREFIX):
                trivial_lane += 1
            if "dpp8:" in rest or "row_" in rest:
                dpp += 1
            elif "quad_perm" in rest:
                quad += 1
            if "exec_hi" in rest or ("exec" in rest and "_b64" in op and "exec_lo" not in rest):
                wave64 = True
    ops = set(mn)
    indirect = sum(mn[o] for o in INDIRECT)
    movrel = sum(c for o, c in mn.items() if "movrel" in o)
    rec = {
        "name": name,
        "family": family(name),
        "ins": ins,
        "code_sha1": h.hexdigest(),
        "programs": endpgm,
        "invalid_words": invalid,
        "branches": branches,
        "loops": loops,
        "xlane": xlane + dpp,
        "trivial_lane": trivial_lane,
        "quad": quad,
        "fetch_call": mn["s_swappc_b64"],
        "pc_data": mn["s_getpc_b64"],
        "wave64": wave64,
        "indirect": indirect,
        "movrel": movrel,
        "rt": sum(c for o, c in mn.items() if o.startswith("image_bvh")),
        "atomics": sum(c for o, c in mn.items() if "atomic" in o or o.startswith(("ds_add", "ds_or", "ds_and",
                                                                                    "ds_max", "ds_min", "ds_xor",
                                                                                    "ds_cmpst", "ds_wrxchg"))),
        "lds": sum(c for o, c in mn.items() if o.startswith("ds_")),
        "samples": sum(c for o, c in mn.items() if o.startswith(("image_sample", "image_gather"))),
        "image_rw": sum(c for o, c in mn.items() if o.startswith(("image_load", "image_store"))),
        "mem": sum(c for o, c in mn.items() if o.startswith(("buffer_", "global_", "flat_", "scratch_"))),
        "scalar_loads": sum(c for o, c in mn.items() if o.startswith(("s_load", "s_buffer_load"))),
        "barriers": mn["s_barrier"],
        "exports": mn["exp"],
        "f16": sum(c for o, c in mn.items() if "_f16" in o or "pk_" in o),
        "unique_ops": len(ops),
    }
    if indirect or movrel:
        rec["tier"] = "D"
    elif rec["xlane"]:
        rec["tier"] = "C"
    elif branches:
        rec["tier"] = "B"
    else:
        rec["tier"] = "A"
    rec["status"] = "ok" if ins else "empty"
    return rec


def size_band(n):
    for limit, label in ((100, "<100"), (500, "100-499"), (2000, "500-1999"), (10000, "2000-9999")):
        if n < limit:
            return label
    return "10000+"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    asm_dir, index_path, out = argv[0], argv[1], argv[2]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else os.cpu_count() or 4
    os.makedirs(out, exist_ok=True)
    log = runlog.setup(os.path.join(out, "classify.jsonl"), "classify", verbose="--verbose" in argv)
    stages = {}
    for e in json.load(open(index_path, encoding="utf-8")):
        stages[os.path.splitext(e["file"])[0]] = e["stage"]
    paths = sorted(os.path.join(asm_dir, n) for n in os.listdir(asm_dir) if n.endswith(".s"))
    recs = []
    with ProcessPoolExecutor(jobs) as ex, open(os.path.join(out, "classify.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(analyse, paths, chunksize=64), 1):
            r["stage"] = stages.get(r["name"]) or (STAGE_RE.search(r["name"]).group(1).lower()
                                                   if STAGE_RE.search(r["name"]) else "?")
            recs.append(r)
            f.write(json.dumps(r) + "\n")
            if r["status"] != "ok":
                log.warning("classify %s: %s %s\n%s", r["name"], r["status"], r.get("exception", "no instructions found"),
                            r.get("traceback", ""))
            if i % 5000 == 0:
                log.info(f"  {i}/{len(paths)}")
    bad = [r for r in recs if r["status"] != "ok"]
    runlog.summarise_failures(recs, lambda r: None if r["status"] == "ok" else f"{r['status']}: {r.get('exception', '')}"[:160], log)
    summarise([r for r in recs if r["status"] == "ok"], out)
    log.info("%d shaders classified, %d left out (see the log)", len(recs) - len(bad), len(bad))
    return 0


def summarise(recs, out):
    n = len(recs)
    total_ins = sum(r["ins"] for r in recs)
    tiers = "ABCD"
    tier_names = {"A": "straight-line", "B": "branches, no cross-lane", "C": "lane shuffling",
                  "D": "computed jumps / dynamic registers"}
    by_tier = collections.Counter(r["tier"] for r in recs)
    ins_by_tier = collections.Counter()
    for r in recs:
        ins_by_tier[r["tier"]] += r["ins"]
    unique_code = {}
    for r in recs:
        unique_code.setdefault(r["code_sha1"], r)
    uniq_tier = collections.Counter(r["tier"] for r in unique_code.values())

    fams = collections.defaultdict(list)
    for r in recs:
        fams[r["family"]].append(r)
    fam_rows = []
    for name, members in fams.items():
        fam_rows.append({
            "family": name,
            "members": len(members),
            "unique_code": len({m["code_sha1"] for m in members}),
            "stages": sorted({m["stage"] for m in members}),
            "tier": max(m["tier"] for m in members),
            "tiers": dict(collections.Counter(m["tier"] for m in members)),
            "ins_total": sum(m["ins"] for m in members),
            "ins_max": max(m["ins"] for m in members),
            "wave64": sum(m["wave64"] for m in members),
            "rt": sum(1 for m in members if m["rt"]),
        })
    fam_rows.sort(key=lambda x: -x["members"])
    json.dump(fam_rows, open(os.path.join(out, "families.json"), "w", encoding="utf-8"), indent=1)
    fam_tier = collections.Counter(x["tier"] for x in fam_rows)

    stage_tier = collections.defaultdict(collections.Counter)
    for r in recs:
        stage_tier[r["stage"]][r["tier"]] += 1
    bands = collections.defaultdict(collections.Counter)
    for r in recs:
        bands[size_band(r["ins"])][r["tier"]] += 1
    flags = {
        "wave64": sum(r["wave64"] for r in recs),
        "wave64 with lane shuffling (exact 64-lane emulation needed)": sum(1 for r in recs if r["wave64"] and r["xlane"]),
        "trivial lane ops only (lane id, broadcast)": sum(1 for r in recs if r["trivial_lane"] and not r["xlane"]),
        "quad ops (derivatives)": sum(1 for r in recs if r["quad"]),
        "quad ops and no real lane shuffling": sum(1 for r in recs if r["quad"] and not r["xlane"]),
        "vertex-fetch call (s_swappc)": sum(1 for r in recs if r["fetch_call"]),
        "PC-relative constant data (s_getpc)": sum(1 for r in recs if r["pc_data"]),
        "loops": sum(1 for r in recs if r["loops"]),
        "ray tracing (image_bvh)": sum(1 for r in recs if r["rt"]),
        "atomics": sum(1 for r in recs if r["atomics"]),
        "LDS (threadgroup memory)": sum(1 for r in recs if r["lds"]),
        "barriers": sum(1 for r in recs if r["barriers"]),
        "half-float maths": sum(1 for r in recs if r["f16"]),
        "several programs in one blob": sum(1 for r in recs if r["programs"] > 1),
        "invalid words in disassembly": sum(1 for r in recs if r["invalid_words"]),
    }

    L = [f"# Shader classification\n", f"{n} shaders, {total_ins:,} instructions, "
         f"{len(unique_code)} distinct programs, {len(fam_rows)} families.\n",
         "## Tiers\n", "| Tier | Shaders | % | Distinct programs | Families (hardest member) | % of instructions |",
         "|---|---|---|---|---|---|"]
    for t in tiers:
        L.append(f"| {t}: {tier_names[t]} | {by_tier[t]} | {100 * by_tier[t] / n:.1f}% | {uniq_tier[t]} | "
                 f"{fam_tier[t]} | {100 * ins_by_tier[t] / total_ins:.1f}% |")
    L += ["\n## Tiers by stage\n", "| Stage | " + " | ".join(tiers) + " | Total |", "|---" * 6 + "|"]
    for s in sorted(stage_tier, key=lambda s: -sum(stage_tier[s].values())):
        c = stage_tier[s]
        L.append(f"| {s} | " + " | ".join(str(c[t]) for t in tiers) + f" | {sum(c.values())} |")
    L += ["\n## Tiers by size (instructions)\n", "| Size | " + " | ".join(tiers) + " |", "|---" * 5 + "|"]
    for b in ("<100", "100-499", "500-1999", "2000-9999", "10000+"):
        L.append(f"| {b} | " + " | ".join(str(bands[b][t]) for t in tiers) + " |")
    L += ["\n## Features\n", "| Feature | Shaders |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in flags.items()]
    L += ["\n## Largest families\n", "| Family | Members | Distinct code | Stages | Tier | Max instructions |",
          "|---|---|---|---|---|---|"]
    for x in fam_rows[:25]:
        L.append(f"| {x['family']} | {x['members']} | {x['unique_code']} | {','.join(x['stages'])} | "
                 f"{x['tier']} {x['tiers']} | {x['ins_max']:,} |")
    cover = 0
    L.append("\n## Family coverage\n")
    for k, x in enumerate(fam_rows, 1):
        cover += x["members"]
        if k in (10, 50, 100, 250, 500, 1000):
            L.append(f"- top {k} families cover {cover} shaders ({100 * cover / n:.1f}%)")
    text = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
