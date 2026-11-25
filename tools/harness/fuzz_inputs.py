"""Coverage-guided input search for prepared shaders (no traces needed).

For each shader prepared by prepare.py, the reference runs alone (harness --b none) in batches of input variants.
A variant is a seed (the harness's seeded title memory, push constants and textures) plus patches: title-memory
words and push-constant dwords set to chosen values. The harness reports per variant which coverage ids fired
(blocks and lane guards) and which outputs were written and which 16-byte chunks of title memory the shader read.
Only the words the shader actually reads (and its non-pointer user SGPRs) are mutated. A variant is kept when it
fires a coverage id no kept variant fired, or writes an output none wrote. The search stops when a number of rounds
in a row find nothing new, or at the round/time budget. The kept set is reduced (greedy cover) and written to
<shader>/inputs.json, which run_set.py then uses instead of plain seeds.

Usage:
    python3 -I fuzz_inputs.py <prep dir> [--jobs N] [--rounds N] [--batch N] [--patience N] [--seconds S]
                              [--resume] [--skip-complete] [--only NAME ...] [--limit N]
--resume starts from each shader's existing inputs.json; --skip-complete skips shaders whose last search reached
every coverage id. Inputs marked "origin": "hunt" (mutant kills from resolve_survivors.py) are never dropped.
Writes <shader>/inputs.json and <shader>/fuzz.json per shader and <prep>/fuzz.jsonl (one summary row per shader).
Logging: <prep>/fuzz.jsonl.log has every harness call and, for every failure, the harness phase, stderr and exit code.
"""
import json
import os
import random
import re
import struct
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlog  # noqa: E402

HARNESS = os.environ.get("SBX_HARNESS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
LOG = None   # set in main()

# ---- the harness's seeded title memory, reproduced (harness.swift: SplitMix, value(), setupInputs) ----
M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
SPECIALS = [0, 0x3F800000, 0xBF800000, 0x3F000000, 0x40000000, 0x7FC00000, 0x7F800000, 0xFF800000,
            0x7F7FFFFF, 0x00000001, 0x80000000, 0xFFFFFFFF, 1, 2, 4, 16, 64, 256]


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def f2u(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _value(r):
    sel, x = r % 100, r >> 8
    if sel < 30:
        return x % 4096
    if sel < 45:
        return (x % (1 << 22)) & ~15
    if sel < 85:
        return f2u((x % 1_000_000) / 125_000.0 - 4.0)
    if sel < 95:
        return SPECIALS[x % len(SPECIALS)]
    return x & 0xFFFFFFFF


def seeded_word(seed, i):
    """Word i of the harness's seeded title memory for `seed` (before the FSSM blob is copied in)."""
    s0 = (seed * 0x1000193) & M64
    return _value(_mix((s0 + (i + 1) * GAMMA) & M64))


INTERESTING = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4095, 0xFFFF, 0x10000,
               0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE] + \
              [f2u(x) for x in (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 0.75, 0.999, 1e-3, 1e-6, 4.0, 10.0, 100.0,
                                1000.0, 1e6, 65504.0, 3.14159, -2.0)] + [0x7FC00000, 0x7F800000, 0xFF800000, 0x00800000]


def shader_dictionary(msl):
    """Values the shader compares against, as raw words: every unsigned constant in main0 (and +-1), every float
    constant written as as_type<float>(Nu) and each small constant shifted to every bit offset the shader extracts
    (extract_bits(x, off, n) or (x >> off) & mask), so a loaded word can carry it in the right field."""
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    body = msl[m.start():] if m else msl
    consts = {int(x) for x in re.findall(r"(?<![\w.\[])(\d+)u\b", body)}
    consts = {c for c in consts if c < (1 << 32)}
    offs = {int(o) for o in re.findall(r"extract_bits\([^,]+,\s*(\d+)u", body)}
    offs |= {int(o) for o in re.findall(r">>\s*\(?(\d+)u", body) if int(o) < 32}
    words = set()
    for c in consts:
        words.update((c, (c + 1) & 0xFFFFFFFF, (c - 1) & 0xFFFFFFFF))
        if c < 1 << 16:
            for o in offs:
                words.add((c << o) & 0xFFFFFFFF)
    return sorted(words)


def seeded_push(seed, i, meta):
    """Push dword i (< 32) as the harness's setupPush sets it before patches."""
    t = 0
    for r in meta.get("user_sgpr_reads") or []:
        k, sg = str(r.get("kind", "")), r.get("sgpr", 99)
        if sg >= 32:
            continue
        if k.startswith("ptr") and k.endswith("_lo"):
            if sg == i:
                return 0x00100000 + t * 0x10000
            t += 1
        elif k.startswith("ptr") and k.endswith("_hi") and sg == i:
            return 0
    s0 = (seed * 0x5bd1e995) & M64
    return _value(_mix((s0 + (i + 1) * GAMMA) & M64))


def field_specs(msl):
    """(offset, width) bit fields the shader extracts: extract_bits(x, off, width), (x >> off) & mask, x & mask."""
    specs = {(0, 32), (0, 16), (16, 16), (0, 8), (8, 8), (16, 8), (24, 8)}
    for o, w in re.findall(r"extract_bits\([^,]+,\s*(\d+)u,\s*(\d+)u\)", msl):
        specs.add((int(o), int(w)))
    for o, mk in re.findall(r">>\s*\(?(\d+)u(?:\s*&\s*31u\))?\)\s*&\s*(\d+)u", msl):
        mk = int(mk)
        if mk and mk & (mk + 1) == 0:
            specs.add((int(o), mk.bit_length()))
    for mk in re.findall(r"&\s*(\d+)u\b", msl):
        mk = int(mk)
        if mk and mk & (mk + 1) == 0 and mk < (1 << 32) - 1:
            specs.add((0, mk.bit_length()))
    return sorted(s for s in specs if 0 <= s[0] < 32 and 0 < s[1] <= 32 and s[0] + s[1] <= 32)


def i2s_variants(parent, cmp, words, specs, push_targets, meta, limit, rng, ordered=False):
    """Input-to-state: for each logged comparison (id, a, b), find title-memory words and push dwords that carry a
    (whole, or in a bit field the shader extracts) and set that field to b, b+1 or b-1."""
    seed = parent["seed"]
    mem, push = dict(map(tuple, parent["mem"])), dict(map(tuple, parent["push"]))
    cur = {("m", w): mem.get(w, seeded_word(seed, w)) for w in words}
    cur.update({("p", i): push.get(i, seeded_push(seed, i, meta)) for i in push_targets})
    pairs = [(a, b) for _, a, b in cmp if a != b] + [(b, a) for _, a, b in cmp if a != b]
    pairs += [(a, (a + 1) & 0xFFFFFFFF) for _, a, b in cmp if a == b]
    if ordered:      # keep the caller's order (closest comparisons first), both directions next to each other
        pairs = []
        for _, a, b in cmp:
            pairs += [(a, b), (b, a)] if a != b else [(a, (a + 1) & 0xFFFFFFFF)]
    else:
        rng.shuffle(pairs)
    out, seen = [], set()
    for a, b in pairs:
        for (kind, idx), v in cur.items():
            for off, width in specs:
                mask = (1 << width) - 1
                if (v >> off) & mask != a & mask or (width < 32 and a > mask):
                    continue
                for t in (b, (b + 1) & 0xFFFFFFFF, (b - 1) & 0xFFFFFFFF):
                    nv = (v & ~(mask << off) | ((t & mask) << off)) & 0xFFFFFFFF
                    if nv == v or (kind, idx, nv) in seen:
                        continue
                    seen.add((kind, idx, nv))
                    m2, p2 = dict(mem), dict(push)
                    (m2 if kind == "m" else p2)[idx] = nv
                    var = {"seed": seed, "mem": sorted(m2.items()), "push": sorted(p2.items())}
                    if parent.get("tex_seed") is not None:
                        var["tex_seed"] = parent["tex_seed"]
                    if parent.get("hash"):
                        var["hash"] = parent["hash"]
                    out.append(var)
                    if len(out) >= limit:
                        return out
    return out


def format_candidates(msl, rng, n=6):
    """GFX10 T# FORMAT values (word 1 bits 20-28) worth trying: set bits of the shader's 16-word constant format
    tables (indexed as table[f >> 5] bit f & 31), one per table where possible, plus random ones."""
    tables = []
    for body in re.findall(r"constant spvUnsafeArray<uint, 16> _\d+ = spvUnsafeArray<uint, 16>\(\{([^}]*)\}\)", msl):
        vals = [int(x) for x in re.findall(r"(\d+)u", body)]
        bits = [f for f in range(512) if (vals[f >> 5] >> (f & 31)) & 1]
        if bits:
            tables.append(bits)
    out = [rng.choice(t) for t in tables]
    # boundaries: next to a set bit (a wider or shifted field read differs there)
    out += [(rng.choice(t) + rng.choice((-1, 1))) % 512 for t in tables]
    out += [rng.randrange(512) for _ in range(max(0, n - len(out)))]
    return out


_FORMATS = None


def valid_formats():
    """GFX10 FORMAT codes that name a real format (Palmetto's tools/gfx10_formats.json), INVALID excluded."""
    global _FORMATS
    if _FORMATS is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "upstream", "palmetto", "tools",
                         "gfx10_formats.json")
        try:
            _FORMATS = sorted(int(k) for k, v in json.load(open(p)).items() if int(k) and "INVALID" not in v)
        except (OSError, ValueError):
            _FORMATS = []
    return _FORMATS


def declared_tsharp_types(msl):
    """T# TYPE codes for the image kinds the shader declares (1D 8, 2D 9, 3D 10, cube 11, 1D array 12, 2D array 13)."""
    out = set()
    for kind, code in (("texture1d_array", 12), ("texture2d_array", 13), ("texturecube", 11), ("texture3d", 10),
                       ("texture1d", 8), ("texture2d", 9), ("depth2d_array", 13), ("depth2d", 9)):
        if re.search(r"\b" + kind + r"<", msl):
            out.add(code)
    return sorted(out) or [9]


def valid_tsharp(key, rng, types, fmts):
    """Rewrite the image-descriptor fields of an 8-word T# key so a real runtime would register it (bit positions as
    harness_tex decodes them): FORMAT word1[28:20] from the format table, WIDTH word1[31:30]+word2[11:0], HEIGHT
    word2[27:14], DST_SEL word3[11:0] (no reserved 2/3), BASE/LAST_LEVEL word3[15:12]/[19:16], TYPE word3[31:28],
    DEPTH word4[12:0], BASE_ARRAY word4[28:16]. Address and other bits are kept."""
    t = list(key)
    ttype = rng.choice(types)
    w = rng.choice((1, 2, 4, 8, 16, 32, 64, 128, 256, rng.randrange(1, 257)))
    h = 1 if ttype in (8, 12) else rng.choice((w, 1, 2, 4, 16, 64, rng.randrange(1, 257)))
    levels = max(1, min(w, h).bit_length())
    last = rng.randrange(levels)
    sel = 0
    for c in range(4):
        sel |= (rng.choice((4, 5, 6, 7)) if rng.random() < 0.9 else rng.choice((0, 1))) << (3 * c)
    if rng.random() < 0.6:
        sel = 4 | (5 << 3) | (6 << 6) | (7 << 9)            # identity swizzle is the common case
    fmt = rng.choice(fmts) if fmts else (t[1] >> 20) & 0x1FF
    t[1] = (t[1] & ~((0x1FF << 20) | (3 << 30))) | (fmt << 20) | (((w - 1) & 3) << 30)
    t[2] = (t[2] & ~(0xFFF | (0x3FFF << 14))) | (((w - 1) >> 2) & 0xFFF) | (((h - 1) & 0x3FFF) << 14)
    t[3] = (t[3] & ~(0xFFF | (0xF << 12) | (0xF << 16) | (0xF << 28))) | sel | (last << 16) | (ttype << 28)
    if ttype == 10:
        depth = rng.choice((1, 2, 4, 8, 16, 32))
        t[4] = (t[4] & ~(0x1FFF | (0x1FFF << 16))) | (depth - 1)
    elif ttype in (11, 12, 13):
        layers = 6 * rng.randrange(1, 3) if ttype == 11 else rng.choice((1, 2, 4, 6, 8))
        t[4] = (t[4] & ~(0x1FFF | (0x1FFF << 16))) | (layers - 1)
    return [x & 0xFFFFFFFF for x in t]


def _key_start(key, cur, seed):
    """Title-memory word index where the 8 key words sit consecutively, or None."""
    for w0, v in cur.items():
        if v != key[0]:
            continue
        if all(cur.get(w0 + i, seeded_word(seed, w0 + i)) == key[i] for i in range(1, 8)):
            return w0
    return None


def hash_variants(parent, keys, words, msl, rng, limit=16):
    """Descriptor model: variants whose hash table holds every logged lookup key (so lookups hit) and variants
    where one key's T# FORMAT field is rewritten in the title-memory word that carries it (with a matching entry)."""
    if not keys:
        return []
    seed = parent["seed"]
    mem = dict(map(tuple, parent["mem"]))
    cur = {w: mem.get(w, seeded_word(seed, w)) for w in words}
    by_val = {}
    for w, v in cur.items():
        by_val.setdefault(v, []).append(w)
    entries = {tuple(k[1:9]): rng.randrange(1, 64) for k in keys}
    out = []

    def make(mem2, ent):
        v = {"seed": seed, "mem": sorted(mem2.items()), "push": [list(x) for x in parent["push"]],
             "hash": [list(k) + [i] for k, i in ent.items()]}
        if parent.get("tex_seed") is not None:
            v["tex_seed"] = parent["tex_seed"]
        return v
    out.append(make(dict(mem), entries))
    # valid descriptors: rewrite each T# key's words in title memory into an image a real runtime would register
    # (most generated keys are seeded random words: TYPE < 8, reserved swizzles, unknown formats); S# keys
    # (words 4..7 all 0xFFFFFFFF) are left as they are
    if VALID_DESC:
        types, vf = declared_tsharp_types(msl), valid_formats()
        tsharp = [k for k in keys if list(k[5:9]) != [0xFFFFFFFF] * 4]
        for _ in range(min(4, limit - len(out))):
            m2, ent = dict(mem), dict(entries)
            changed = False
            for k in tsharp:
                key = list(k[1:9])
                w0 = _key_start(key, cur, seed)
                if w0 is None:
                    continue
                nk = valid_tsharp(key, rng, types, vf)
                for i in range(8):
                    m2[w0 + i] = nk[i]
                ent[tuple(nk)] = ent.pop(tuple(key), rng.randrange(1, 64))
                changed = True
            if changed:
                out.append(make(m2, ent))
    fmts = format_candidates(msl, rng)
    for k in keys:
        key = list(k[1:9])
        for w in by_val.get(key[1], [])[:1]:
            for f in fmts:
                nk = list(key)
                nk[1] = (key[1] & ~(0x1FF << 20)) | (f << 20)
                ent = dict(entries)
                ent[tuple(nk)] = ent.pop(tuple(key), rng.randrange(1, 64))
                m2 = dict(mem)
                m2[w] = nk[1]
                out.append(make(m2, ent))
                if len(out) >= limit:
                    return out
    return out


def new_value(rng, cur, pool, dictionary=()):
    k = rng.random()
    if dictionary and k < 0.4:
        v = rng.choice(dictionary)
        return v if rng.random() < 0.7 else (cur & ~0xFFFF) | (v & 0xFFFF) if v < 1 << 16 else v
    k = rng.random()
    if k < 0.35:
        return rng.choice(INTERESTING)
    if k < 0.5:
        return cur ^ (1 << rng.randrange(32))
    if k < 0.6:
        return (cur + rng.choice((1, -1, 2, -2, 4, -4, 16, -16))) & 0xFFFFFFFF
    if k < 0.7 and pool:
        return rng.choice(pool)
    if k < 0.8:
        return rng.randrange(64)
    if k < 0.9:
        return f2u(rng.uniform(-8.0, 8.0))
    return rng.getrandbits(32)


# ---- harness calls ----

VALID_DESC = True   # hash_variants also builds valid T#s (--no-valid-desc turns it off)
DET_CHECK = set()   # shaders (names) whose search runs the plain reference as B to reject racing variants


def run_batch(d, variants, tmp, tag, reads=True):
    """Run the reference alone on `variants`; returns the harness JSON (with 'variants' rows) or None. For shaders in
    DET_CHECK the plain reference runs as B too and rows where A and B disagree (a race) are marked "racing"."""
    inp = os.path.join(tmp, f"in_{tag}.json")
    json.dump(variants, open(inp, "w"))
    ref = os.path.join(d, "ref.metal")
    cl = os.path.join(d, "cmplog.metal")
    det = os.path.basename(d) in DET_CHECK
    cmd = [HARNESS, "--meta", os.path.join(d, "meta.json"), "--a", cl if os.path.exists(cl) else ref,
           "--b", ref if det else "none", "--inputs", inp, "--all", "1", "--coverage", "1"]
    if os.path.exists(cl):
        cmd += ["--cmplog", "1"]
    if reads:
        cmd += ["--reads", "1"]
    vs = os.path.join(d, "vs.metal")
    if os.path.exists(vs):
        cmd += ["--vs", vs]
    r = runlog.run(cmd, 300, LOG, f"fuzz {os.path.basename(d)} batch {tag} ({len(variants)} variants)")
    lines = [l for l in (r.get("stdout") or "").splitlines() if l.startswith("{")]
    if not lines:
        return {"status": "process_timeout" if r["timed_out"] else "crash", "failure": runlog.failure(r)}
    try:
        res = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"status": "bad_json", "error": str(e), "failure": runlog.failure(r)}
    if res.get("status") != "ok":
        res["failure"] = runlog.failure(r)
        LOG.warning("fuzz %s batch %s: status %s in %s", os.path.basename(d), tag, res.get("status"), res.get("phase"))
    elif det:
        for row in res.get("variants", []):
            if row.get("equal") is False or row.get("loop_guard_b"):
                row["racing"] = True
                row["loop_guard"] = True        # callers already skip loop-guard rows: the same treatment
    return res


def cmp_features(row):
    """Comparison outcomes as coverage features: (id, a<b / a==b / a>b) of each logged comparison."""
    return {("c", i, (a > b) - (a < b)) for i, a, b in row.get("cmp", [])}


SINK = re.compile(r"\bout\.|reinterpret_cast<device|atomic_|\.write\(|\.atomic")
CL_ID = re.compile(r"sbx_cl\((\d+)u,")


def _indent(line):
    return len(line) - len(line.lstrip())


def block_targets(cl_text):
    """From cmplog.metal: for every coverage block, its parent block, the comparison ids of its immediate guard (its
    own header line, the matching if for an else and the lines assigning the variables that header tests, two
    levels back) and whether it holds a sink (output, store, atomic, texture write)."""
    lines = cl_text.split("\n")
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", cl_text)
    start = cl_text.count("\n", 0, m.start()) if m else 0
    assigns, block_at = {}, {}
    for k in range(start, len(lines)):
        a = re.match(r"\s*(?:\w+\s+)?(_\d+)\s*=[^=]", lines[k])
        if a:
            assigns.setdefault(a.group(1), []).append(k)
        h = re.match(r"\s*sbx_hit\((\d+)\);", lines[k])
        if h:
            block_at[k] = int(h.group(1))
    out = {}
    for i, bid in block_at.items():
        cur = _indent(lines[i])
        header, parent = None, None
        for k in range(i - 1, start, -1):
            t = lines[k]
            if t.strip() == "{" and _indent(t) < cur:
                if header is None:
                    j = k - 1
                    while j > start and not lines[j].strip():
                        j -= 1
                    header = j
                    cur = _indent(t)
                    continue
                # the enclosing block: its marker is the line after this brace
                if k + 1 in block_at:
                    parent = block_at[k + 1]
                    break
                cur = _indent(t)
        heads = [header] if header is not None else []
        if header is not None and lines[header].strip() == "else":
            for q in range(header - 1, start, -1):
                if _indent(lines[q]) == _indent(lines[header]) and lines[q].strip().startswith(("if (", "else if (")):
                    heads.append(q)
                    break
        ids, seen_vars, frontier = set(), set(), heads
        for _ in range(3):
            nxt = []
            for k in frontier:
                ids.update(int(x) for x in CL_ID.findall(lines[k]))
                for v in re.findall(r"\b_\d+\b", re.sub(r"sbx_cl\(\d+u,", "", lines[k])):
                    if v not in seen_vars:
                        seen_vars.add(v)
                        nxt += assigns.get(v, [])[:6]
            frontier = nxt
        sink = False
        for k in range(i + 1, min(i + 400, len(lines))):
            if re.match(r"\s*sbx_hit\(", lines[k]) or _indent(lines[k]) < _indent(lines[i]):
                break
            sink |= bool(SINK.search(lines[k]))
        out[bid] = {"guards": sorted(ids), "sink": sink, "parent": parent}
    return out


def _ordered(x, dom):
    """A logged operand (32-bit pattern) as a number whose order matches the comparison's domain."""
    if dom == "f":
        return x if x < 0x80000000 else -(x & 0x7FFFFFFF)      # float bits in total order (ulp distance)
    if dom == "i":
        return x - (1 << 32) if x >= 0x80000000 else x
    return x


NEG = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", "<=": ">", ">": "<="}


def branch_distance(op, dom, a, b, want):
    """0 when `a op b` has the wanted outcome, else how far the operands are from giving it."""
    if not want:
        op = NEG[op]
    x, y = _ordered(a, dom), _ordered(b, dom)
    if op == "==":
        return abs(x - y)
    if op == "!=":
        return 0 if x != y else 1
    if op == "<":
        return 0 if x < y else x - y + 1
    if op == "<=":
        return 0 if x <= y else x - y
    if op == ">":
        return 0 if x > y else y - x + 1
    return 0 if x >= y else y - x                              # >=


def taint_map(d, parent, words, interest, meta, rng, tmp, stats, group=8, refine_cap=256):
    """Which title-memory words (and push dwords) move which comparisons: perturb words in groups of `group`,
    then single words inside the groups that moved a comparison of interest. Returns {cmp id: [word keys]}, a key
    being ("m", word) or ("p", dword)."""
    base = run_batch(d, [parent], tmp, "taint_base")
    if base.get("status") != "ok":
        return {}
    ref = {c[0]: (c[1], c[2]) for c in base["variants"][0].get("cmp", [])}
    push_targets = [r["sgpr"] for r in meta.get("user_sgpr_reads") or []
                    if r.get("sgpr", 99) < 32 and not str(r.get("kind", "")).startswith("ptr")]
    keys = [("m", w) for w in words] + [("p", i) for i in push_targets]
    rng.shuffle(keys)

    def variant(ks):
        mem, push = dict(map(tuple, parent["mem"])), dict(map(tuple, parent["push"]))
        for kind, i in ks:
            if kind == "m":
                cur = mem.get(i, seeded_word(parent["seed"], i))
                mem[i] = cur ^ (0xFFFFFFFF if rng.random() < 0.5 else (1 << rng.randrange(32)) | 0x00400000)
            else:
                cur = push.get(i, seeded_push(parent["seed"], i, meta))
                push[i] = cur ^ (1 << rng.randrange(32)) ^ 0x3F800000
        v = {"seed": parent["seed"], "mem": sorted(mem.items()), "push": sorted(push.items())}
        for k in ("tex_seed", "hash"):
            if parent.get(k) is not None:
                v[k] = parent[k]
        return v

    def moved(rows):
        out = []
        for row in rows:
            got = {c[0]: (c[1], c[2]) for c in row.get("cmp", [])}
            out.append({i for i in interest if got.get(i) != ref.get(i)})
        return out
    groups = [keys[i:i + group] for i in range(0, len(keys), group)]
    hit_groups = []
    for i in range(0, len(groups), 32):
        chunk = groups[i:i + 32]
        res = run_batch(d, [variant(g) for g in chunk], tmp, f"taint_g{i}")
        stats["taint_batches"] += 1
        if res.get("status") != "ok":
            continue
        for g, mv in zip(chunk, moved(res["variants"])):
            if mv:
                hit_groups.append(g)
    singles = [k for g in hit_groups for k in g][:refine_cap]
    out = {}
    for i in range(0, len(singles), 32):
        chunk = singles[i:i + 32]
        res = run_batch(d, [variant([k]) for k in chunk], tmp, f"taint_s{i}")
        stats["taint_batches"] += 1
        if res.get("status") != "ok":
            continue
        for k, mv in zip(chunk, moved(res["variants"])):
            for cid in mv:
                out.setdefault(cid, []).append(k)
    return out


def directed(d, corpus, cover, words, specs, push_targets, meta, rng, args, tmp, stats):
    """Directed search on frontier blocks (unreached, parent reached; sinks first): for each, the guard comparisons
    whose needed outcome no input has produced yet are climbed by operator-aware branch distance, moving only the
    words a taint map ties to them, until the block's coverage id fires."""
    cl = os.path.join(d, "cmplog.metal")
    ops = meta.get("cmp_ops") or []
    if not os.path.exists(cl) or not ops:
        return []
    targets = block_targets(open(cl).read())
    nb = meta.get("cov_blocks") or 0
    todo = [b for b in targets if b not in cover and b <= nb and targets[b]["guards"]
            and (targets[b]["parent"] is None or targets[b]["parent"] in cover)]
    todo.sort(key=lambda b: (not targets[b]["sink"], b))
    stats["directed_frontier"] = len(todo)
    found = []
    taints = {}
    for b in todo[:args["directed_targets"]]:
        if b in cover:
            continue
        g = [i for i in targets[b]["guards"] if i < len(ops)]
        seen = {}
        for v in corpus:
            for i, a, c in v.get("cmp", []):
                if i in g:
                    op, dom = ops[i]
                    x, y = _ordered(a, dom), _ordered(c, dom)
                    res = {"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
                    seen.setdefault(i, set()).add(res)
        need = {i: (not next(iter(o))) for i, o in seen.items() if len(o) == 1}   # outcome never seen
        if not need:
            stats["directed_no_flip_guard"] += 1
            continue

        def fit(v):
            ds = [branch_distance(ops[i][0], ops[i][1], a, c, need[i]) for i, a, c in v.get("cmp", []) if i in need]
            return min(ds) if ds else float("inf")
        pool = sorted(corpus, key=fit)[:4]
        if fit(pool[0]) == float("inf"):
            continue
        key = (b, json.dumps(pool[0]["mem"])[:64])
        if key not in taints:
            taints[key] = taint_map(d, pool[0], words, set(need), meta, rng, tmp, stats)
        tmap = taints[key]
        tainted = sorted({k for i in need for k in tmap.get(i, [])})
        stats["directed_tainted_words"] += len(tainted)
        if not tainted:
            stats["directed_untainted"] += 1
        for rnd in range(args["directed_rounds"]):
            batch = []
            best = pool[0]
            near = [c for c in best.get("cmp", []) if c[0] in need]
            tw = [i for kind, i in tainted if kind == "m"]
            tp = [i for kind, i in tainted if kind == "p"]
            batch += i2s_variants(best, near, tw or words, specs, tp, meta, args["batch"] // 4, rng, ordered=True)
            dist = max(1, int(fit(best)) if fit(best) != float("inf") else 1)
            while len(batch) < args["batch"]:
                par = rng.choice(pool)
                mem, push = dict(map(tuple, par["mem"])), dict(map(tuple, par["push"]))
                for _ in range(rng.choice((1, 1, 2))):
                    kind, i = rng.choice(tainted) if tainted else ("m", rng.choice(words)) if words else ("p", 0)
                    cur = (mem.get(i, seeded_word(par["seed"], i)) if kind == "m"
                           else push.get(i, seeded_push(par["seed"], i, meta)))
                    r = rng.random()
                    if r < 0.5:      # a step scaled to the remaining distance, either way
                        step = max(1, int(dist * rng.choice((0.25, 0.5, 1, 2))))
                        nv = (cur + rng.choice((-1, 1)) * step) & 0xFFFFFFFF
                    elif r < 0.8:
                        nv = new_value(rng, cur, [])
                    else:
                        nv = (cur + rng.choice((-1, 1)) * (1 << rng.randrange(0, 24))) & 0xFFFFFFFF
                    (mem if kind == "m" else push)[i] = nv
                v = {"seed": par["seed"], "mem": sorted(mem.items()), "push": sorted(push.items())}
                for k in ("tex_seed", "hash"):
                    if par.get(k) is not None:
                        v[k] = par[k]
                if not tainted and rng.random() < 0.3:
                    v["tex_seed"] = rng.randrange(1, 1 << 31)
                batch.append(v)
            batch = batch[:args["batch"]]
            res = run_batch(d, batch, tmp, f"dir{b}_{rnd}")
            stats["directed_batches"] += 1
            if res.get("status") != "ok":
                break
            rows = [dict(v, hits=row["hits"], written=row["written"], cmp=row.get("cmp", []), keys=row.get("keys", []))
                    for v, row in zip(batch, res["variants"]) if not row.get("loop_guard")]
            hit = [r for r in rows if b in r["hits"]]
            if hit:
                found.append(hit[0])
                cover |= set(hit[0]["hits"])
                stats["directed_reached"] += 1
                break
            pool = sorted(pool + rows, key=fit)[:4]
    return found


def fuzz_one(d, args):
    try:
        return _fuzz_one(d, args)
    except Exception as e:                       # one shader must never stop the batch
        return {"name": os.path.basename(d), "status": "exception",
                "failure": runlog.exception(e, LOG, f"fuzz {os.path.basename(d)}")}


def _fuzz_one(d, args):
    name = os.path.basename(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    rng = random.Random(name)
    t0 = time.time()
    dictionary = shader_dictionary(open(os.path.join(d, "ref.metal")).read())
    read_shift = 4
    # push dwords that hold values (not table pointers): only those are mutated
    push_targets = [r["sgpr"] for r in meta.get("user_sgpr_reads") or []
                    if r.get("sgpr", 99) < 32 and not str(r.get("kind", "")).startswith("ptr")]
    corpus, cover, written = [], set(), set()
    reads = set()
    stats = Counter()
    inp_path = os.path.join(d, "inputs.json")
    previous = json.load(open(inp_path)) if args.get("resume") and os.path.exists(inp_path) else []
    total = (meta.get("cov_blocks") or 0) + (meta.get("cov_guards") or 0)
    if args.get("skip_complete") and previous and os.path.exists(os.path.join(d, "fuzz.json")):
        prev = json.load(open(os.path.join(d, "fuzz.json")))
        if prev.get("status") == "ok" and prev.get("ids_fuzzed", 0) >= total:
            LOG.debug("fuzz %s: skipped, all %d coverage ids already reached", name, total)
            return dict(prev, status="ok", skipped="complete")
    with tempfile.TemporaryDirectory() as tmp:
        base = [{"seed": s, "mem": [], "push": []} for s in (1, 2, 3)]
        seen = {json.dumps(v, sort_keys=True) for v in base}
        for v in previous:                       # resume: the kept inputs of earlier runs (and the resolver's kills)
            if json.dumps(v, sort_keys=True) not in seen:
                base.append(v)
                seen.add(json.dumps(v, sort_keys=True))
        res = run_batch(d, base, tmp, "seeds")
        if res.get("status") != "ok":
            return {"name": name, "status": res.get("status"), "phase": res.get("phase"),
                    "failure": res.get("failure")}
        read_shift = res.get("read_shift", 4)
        reads.update(res.get("reads", []))
        feats = set()
        if len(res.get("variants", [])) != len(base):
            return {"name": name, "status": "bad_rows", "phase": res.get("phase")}
        for v, row in zip(base, res["variants"]):
            if row.get("racing"):
                stats["racing_base_inputs"] += 1
                continue
            cover |= set(row["hits"])
            written |= set(row["written"])
            feats |= cmp_features(row)
            corpus.append(dict(v, hits=row["hits"], written=row["written"], cmp=row.get("cmp", []),
                               keys=row.get("keys", [])))
        if not corpus:
            return {"name": name, "status": "all_inputs_racing"}
        start_cover, start_written, start_feats = len(cover), sorted(written), len(feats)
        msl = open(os.path.join(d, "ref.metal")).read()
        specs = field_specs(msl)
        i2s_queue = []
        idle, rounds = 0, 0
        while rounds < args["rounds"] and idle < args["patience"] and time.time() - t0 < args["seconds"]:
            rounds += 1
            words = sorted({(c << read_shift) // 4 + k for c in reads for k in range((1 << read_shift) // 4)})
            if not words and not push_targets:
                break
            if len(i2s_queue) < args["batch"]:
                for par in rng.sample(corpus, min(4, len(corpus))):
                    i2s_queue += hash_variants(par, par.get("keys", []), words, msl, rng)
                    i2s_queue += i2s_variants(par, par.get("cmp", []), words, specs, push_targets, meta,
                                              args["batch"], rng)
            batch = i2s_queue[:args["batch"] // 2]
            i2s_queue = i2s_queue[args["batch"] // 2:]
            stats["i2s_tried"] += len(batch)
            while len(batch) < args["batch"]:
                parent = rng.choice(corpus[-8:] if rng.random() < 0.5 else corpus)
                mem = dict(parent["mem"])
                push = dict(parent["push"])
                for _ in range(rng.choice((1, 1, 2, 3, 4, 8))):
                    if push_targets and (not words or rng.random() < 0.2):
                        i = rng.choice(push_targets)
                        push[i] = new_value(rng, push.get(i, 0), [], dictionary)
                    else:
                        i = rng.choice(words)
                        cur = mem.get(i, seeded_word(parent["seed"], i))
                        pool = [mem.get(j, seeded_word(parent["seed"], j)) for j in rng.sample(words, min(4, len(words)))]
                        mem[i] = new_value(rng, cur, pool, dictionary)
                v = {"seed": parent["seed"], "mem": sorted(mem.items()), "push": sorted(push.items())}
                if parent.get("hash"):
                    v["hash"] = parent["hash"]
                tex = parent.get("tex_seed")
                if rng.random() < 0.15:
                    tex = rng.randrange(1, 1 << 31)   # new texture contents
                if tex is not None:
                    v["tex_seed"] = tex
                batch.append(v)
            res = run_batch(d, batch, tmp, f"r{rounds}")
            if res.get("status") != "ok":
                stats[f"batch_{res.get('status')}"] += 1
                idle += 1
                continue
            reads.update(res.get("reads", []))
            found = False
            for v, row in zip(batch, res["variants"]):
                if row.get("loop_guard"):
                    stats["loop_guard_variants"] += 1
                    continue                     # a cut-short loop: not a usable input
                new_ids = set(row["hits"]) - cover
                new_out = set(row["written"]) - written
                new_f = cmp_features(row) - feats
                if new_ids or new_out or new_f:
                    cover |= new_ids
                    written |= new_out
                    feats |= new_f
                    corpus.append(dict(v, hits=row["hits"], written=row["written"], cmp=row.get("cmp", []),
                                       keys=row.get("keys", [])))
                    found = True
            idle = 0 if found else idle + 1
        if args.get("directed_targets"):
            words = sorted({(c << read_shift) // 4 + k for c in reads for k in range((1 << read_shift) // 4)})
            for v in directed(d, corpus, cover, words, specs, push_targets, meta, rng, args, tmp, stats):
                corpus.append(v)
                written |= set(v["written"])
                feats |= cmp_features(v)
        # reduce: greedy cover of coverage ids and written outputs, written-output variants first
        # inputs that killed a mutant (resolve_survivors marks them "origin": "hunt") are always kept
        keep = [v for v in corpus if v.get("origin") == "hunt"]
        left_ids, left_out, left_f = set(cover), set(written), set(feats)
        for v in keep:
            left_ids -= set(v["hits"])
            left_out -= set(v["written"])
            left_f -= cmp_features(v)
        pool = [v for v in corpus if v.get("origin") != "hunt"]
        n_hunt = len(keep)
        while (left_ids or left_out or left_f) and pool and len(keep) < args["keep"] + n_hunt:
            best = max(pool, key=lambda v: (len(set(v["written"]) & left_out), len(set(v["hits"]) & left_ids),
                                            len(cmp_features(v) & left_f)))
            if not (set(best["written"]) & left_out or set(best["hits"]) & left_ids or cmp_features(best) & left_f):
                break
            keep.append(best)
            pool.remove(best)
            left_ids -= set(best["hits"])
            left_out -= set(best["written"])
            left_f -= cmp_features(best)
        for s in (1, 2, 3):                       # keep the plain seeds too: comparable with earlier runs
            if not any(v["seed"] == s and not v["mem"] and not v["push"] and v.get("tex_seed") is None for v in keep):
                plain = next((v for v in corpus if v["seed"] == s and not v["mem"] and not v["push"]
                              and v.get("tex_seed") is None), None)
                if plain is not None:           # absent when that seed races (det-check) or loops
                    keep.append(plain)
    inputs = [dict({"seed": v["seed"], "mem": v["mem"], "push": v["push"]},
                   **({"tex_seed": v["tex_seed"]} if v.get("tex_seed") is not None else {}),
                   **({"hash": v["hash"]} if v.get("hash") else {}),
                   **({"origin": v["origin"]} if v.get("origin") else {})) for v in keep]
    json.dump(inputs, open(os.path.join(d, "inputs.json"), "w"))
    rec = {"name": name, "status": "ok", "stage": meta.get("stage"), "rounds": rounds,
           "seconds": round(time.time() - t0, 1), "corpus": len(corpus), "kept": len(keep),
           "resumed_from": len(previous), "kept_hunt": n_hunt,
           "ids_total": (meta.get("cov_blocks") or 0) + (meta.get("cov_guards") or 0),
           "ids_seeds": start_cover, "ids_fuzzed": len(cover),
           "written_seeds": start_written, "written_fuzzed": sorted(written),
           "cmp_outcomes_seeds": start_feats, "cmp_outcomes_fuzzed": len(feats),
           "read_chunks": len(reads), "dictionary": len(dictionary), **stats}
    json.dump(rec, open(os.path.join(d, "fuzz.json"), "w"), indent=1)
    LOG.debug("fuzz %s: %s", name, rec)
    return rec


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    prep = argv[0]

    def opt(k, default):
        return type(default)(argv[argv.index(k) + 1]) if k in argv else default
    args = {"rounds": opt("--rounds", 40), "batch": opt("--batch", 32), "patience": opt("--patience", 6),
            "seconds": opt("--seconds", 240.0), "keep": opt("--keep", 32),
            "resume": "--resume" in argv, "skip_complete": "--skip-complete" in argv,
            "directed_targets": opt("--directed-targets", 12), "directed_rounds": opt("--directed-rounds", 6)}
    jobs = opt("--jobs", 6)
    global VALID_DESC
    if "--no-valid-desc" in argv:
        VALID_DESC = False
    if "--det-check" in argv:   # --det-check NAME,NAME,... (or "all")
        val = argv[argv.index("--det-check") + 1]
        DET_CHECK.update(n for n in os.listdir(prep) if val == "all" or n in val.split(","))
    only = set(argv[argv.index("--only") + 1:]) if "--only" in argv else None
    global LOG
    out = os.path.join(prep, "fuzz.jsonl")
    LOG = runlog.setup(out, "fuzz_inputs", verbose="--verbose" in argv)
    LOG.info("fuzz_inputs: prep %s, %s, jobs %d, harness %s", prep, args, jobs, HARNESS)
    dirs = sorted(os.path.join(prep, n) for n in os.listdir(prep)
                  if os.path.isfile(os.path.join(prep, n, "meta.json")) and (not only or n in only))
    if "--limit" in argv:
        dirs = dirs[:opt("--limit", 0)]
    recs = []
    with ThreadPoolExecutor(jobs) as ex, open(out, "w") as f:
        for i, r in enumerate(ex.map(lambda d: fuzz_one(d, args), dirs), 1):
            recs.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0:
                LOG.info("  %d/%d", i, len(dirs))
    ok = [r for r in recs if r["status"] == "ok"]
    LOG.info("%d shaders: %s", len(recs), dict(Counter(r["status"] for r in recs)))
    if ok:
        tot = sum(r["ids_total"] for r in ok)
        LOG.info("coverage ids: seeds %d -> fuzzed %d of %d (%.1f%% -> %.1f%%)", sum(r["ids_seeds"] for r in ok),
                 sum(r["ids_fuzzed"] for r in ok), tot, 100 * sum(r["ids_seeds"] for r in ok) / max(tot, 1),
                 100 * sum(r["ids_fuzzed"] for r in ok) / max(tot, 1))
        LOG.info("shaders writing output: seeds %d -> fuzzed %d of %d", sum(1 for r in ok if r["written_seeds"]),
                 sum(1 for r in ok if r["written_fuzzed"]), len(ok))
    runlog.summarise_failures(recs, lambda r: None if r["status"] == "ok" else
                              f"{r['status']} in {r.get('phase')}"[:160], LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
