"""Prepare translated shaders for the Metal harness.

For each shader in <spirv dir>/index.json (rdna2spirv.py output):
  SPIR-V -> spirv_metal_set0.py (Palmetto: fixed argument-buffer IDs) -> SPIRV-Cross MSL
  -> sandboxed MSL: every device-pointer cast goes through sbx(), which wraps the address into a
     window of title memory. Random memory contents then can't send the GPU outside its buffer.
     Reference and candidate get the same rewrite, so comparing them stays valid.
  -> for pixel shaders, a generated vertex shader that feeds every fragment input with varied values.

Usage:
    python3 -I prepare.py <spirv dir> <fssm dir> <out dir> [--jobs N] [--only NAME ...]
Writes <out>/<name>/ref.metal, vs.metal (pixel shaders), meta.json and <out>/prepare.jsonl.
Logging: <out>/prepare.jsonl.log has every subprocess call and, for each failure, the command, exit code and
stderr; failed records in prepare.jsonl carry the same context under "failure". An exception in one shader is
caught, logged with its traceback and recorded as status "exception"; the batch goes on.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlog  # noqa: E402

LOG = None   # set in main()

SET0 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "upstream", "palmetto", "tools", "spirv_metal_set0.py")

SBX_PRELUDE = """
// ---- harness sandbox: device addresses wrap inside the title-memory window ----
#ifndef SBX_BASE
#define SBX_BASE 0ul
#endif
#ifndef SBX_MASK
#define SBX_MASK 0x3FFFFFFul
#endif
static inline ulong sbx(ulong a) { return SBX_BASE + ((a - SBX_BASE) & SBX_MASK); }
template<typename T> static inline device T* sbx(device T* p)
{ return reinterpret_cast<device T*>(sbx(reinterpret_cast<ulong>(p))); }
template<typename T> static inline const device T* sbx(const device T* p)
{ return reinterpret_cast<const device T*>(sbx(reinterpret_cast<ulong>(p))); }
// Coverage and loop guard. sbx_cov is an extra buffer argument of main0 ([[buffer(29)]]):
//   sbx_cov[0]  set when the loop guard broke out of a loop (the run is then inconclusive)
//   sbx_cov[i]  set when block i ran (only with SBX_COV defined)
#ifndef SBX_LOOP_CAP
#define SBX_LOOP_CAP 4096u
#endif
#ifdef SBX_COV
#define sbx_hit(i) atomic_store_explicit(&sbx_cov[(i)], 1u, memory_order_relaxed)
#else
#define sbx_hit(i) ((void)0)
#endif
// Lane guards: `X = c ? new : X;` becomes `X = sbx_sel(id, c) ? new : X;` (id set when the write took effect);
// `X = c ? X : new;` becomes `X = sbx_nsel(id, c) ? X : new;`. c is evaluated once either way.
#define sbx_sel(i, c) ((c) ? (sbx_hit(i), true) : false)
#define sbx_nsel(i, c) ((c) ? true : (sbx_hit(i), false))
#define SBX_LOOP_GUARD if (++sbx_iter > SBX_LOOP_CAP) { atomic_store_explicit(&sbx_cov[0], 1u, memory_order_relaxed); break; }
// ---- end harness sandbox ----
"""
COV_BUFFER_INDEX = 29
LOOP_HEAD = re.compile(r"^\s*(for \(.*\)|while \(.*\)|do)\s*$")


READS_HOOK = """
// ---- harness read tracking (SBX_READS, set by the harness with --reads): every sandboxed address marks the
// 64 bytes from it in a bitmap at sbx_cov[32768...], one bit per (1 << SBX_RD_SHIFT) bytes of the window.
#ifdef SBX_READS
static inline ulong sbx_rd(ulong a, device atomic_uint* cov)
{
    ulong o = (a - SBX_BASE) & SBX_MASK;
    for (uint k = 0u; k < 64u; k += (1u << SBX_RD_SHIFT))
    {
        ulong c = ((o + k) & SBX_MASK) >> SBX_RD_SHIFT;
        atomic_fetch_or_explicit(&cov[32768u + uint(c >> 5)], 1u << uint(c & 31ul), memory_order_relaxed);
    }
    return sbx(a);
}
template<typename T> static inline device T* sbx_rd(device T* p, device atomic_uint* cov)
{ sbx_rd(reinterpret_cast<ulong>(p), cov); return sbx(p); }
template<typename T> static inline const device T* sbx_rd(const device T* p, device atomic_uint* cov)
{ sbx_rd(reinterpret_cast<ulong>(p), cov); return sbx(p); }
#define sbx(a) sbx_rd((a), sbx_cov)
#endif
"""


CMPLOG_HOOK = """
// ---- harness comparison log (SBX_CMPLOG, only in cmplog.metal for the input search): the operands of
// comparison i are kept at sbx_cov[65536 + 3i] (flag), [+1] left bits, [+2] right bits, first writer wins.
#ifdef SBX_CMPLOG
static inline uint sbx_bits(uint x) { return x; }
static inline uint sbx_bits(int x) { return as_type<uint>(x); }
static inline uint sbx_bits(float x) { return as_type<uint>(x); }
static inline uint sbx_bits(ulong x) { return uint(x); }
static inline uint sbx_bits(long x) { return uint(ulong(x)); }
static inline uint sbx_bits(half x) { return uint(as_type<ushort>(x)); }
static inline uint sbx_bits(ushort x) { return uint(x); }
static inline uint sbx_bits(short x) { return uint(ushort(x)); }
static inline uint sbx_bits(bool x) { return uint(x); }
// keeps the operand pair closest to equal across all threads (|bits(a) - bits(b)|; ulp distance for floats of one
// sign): [base] = 0xFFFFFFFF - distance (atomic max, 0 = never ran), then the winner writes its operands
template<typename T, typename U> static inline T sbx_cl_(uint id, T a, U b, device atomic_uint* cov)
{
    uint base = 65536u + id * 3u;
    uint x = sbx_bits(a), y = sbx_bits(b);
    uint key = 0xFFFFFFFFu - (x > y ? x - y : y - x);
    key = max(key, 1u);
    if (atomic_fetch_max_explicit(&cov[base], key, memory_order_relaxed) < key)
    {
        atomic_store_explicit(&cov[base + 1u], x, memory_order_relaxed);
        atomic_store_explicit(&cov[base + 2u], y, memory_order_relaxed);
    }
    return a;
}
#define sbx_cl(id, a, b) sbx_cl_((id), (a), (b), sbx_cov)
// descriptor hash lookups: the 8 key words of call i at sbx_cov[200000 + 9i] (flag, then the keys)
static inline uint sbx_kl_(uint id, uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, device atomic_uint* cov)
{
    uint base = 200000u + id * 9u;
    if (atomic_load_explicit(&cov[base], memory_order_relaxed) == 0u)
    {
        uint k[8] = { a, b, c, d, e, f, g, h };
        for (uint i = 0u; i < 8u; i++) atomic_store_explicit(&cov[base + 1u + i], k[i], memory_order_relaxed);
        atomic_store_explicit(&cov[base], 1u, memory_order_relaxed);
    }
    return a;
}
#define sbx_kl(id, a, b, c, d, e, f, g, h) sbx_kl_((id), (a), (b), (c), (d), (e), (f), (g), (h), sbx_cov)
#else
#define sbx_cl(id, a, b) (a)
#define sbx_kl(id, a, b, c, d, e, f, g, h) (a)
#endif
"""
# a call to the descriptor hash lookup: _N(8 key words, (*spvDescriptorSet0.m_K))
HASH_CALL_RE = re.compile(r"(_\d+)\(((?:[^(),]+, ){8})\(\*spvDescriptorSet0\.(m_\d+)\)\)")
_ATOM = r"(?:_\d+(?:\[\d+u?\])?(?:\.[xyzw])?|\d+(?:\.\d+)?(?:ul|u|f)?)"
_OPND = (r"(?:_\d+(?:\[\d+u?\])?(?:\.[xyzw])?|abs\(_\d+\)|\d+(?:\.\d+)?(?:ul|u|f)?|as_type<float>\(\d+u\)"
         r"|\(" + _ATOM + r" [-+*&|^] " + _ATOM + r"\))")
CMP_RE = re.compile(r"\((" + _OPND + r") (==|!=|>=|<=|>|<) (" + _OPND + r")\)")


def cmplog(msl):
    """cmplog.metal: every simple comparison `(a op b)` in main0 becomes `(sbx_cl(id, a, b) op b)`, which logs
    the operands under SBX_CMPLOG and is `a` otherwise. Returns (text, number of comparison ids)."""
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    if not m:
        return msl, 0, 0
    line_start = msl.rfind("\n", 0, m.start()) + 1
    head, body = msl[:line_start], msl[line_start:]
    n = [0]
    float_vars = set(re.findall(r"\b(?:float|half)\s+(_\d+)\s*=", body))
    int_vars = set(re.findall(r"\b(?:int|long|short)\s+(_\d+)\s*=", body))
    ops = []

    def domain(x):
        if "as_type<float>" in x or re.fullmatch(r"\d+\.\d+f?", x) or any(v in float_vars for v in re.findall(r"_\d+", x)):
            return "f"
        if any(v in int_vars for v in re.findall(r"_\d+", x)):
            return "i"
        return "u"

    def sub(g):
        i = n[0]
        n[0] += 1
        da, db = domain(g.group(1)), domain(g.group(3))
        ops.append([g.group(2), "f" if "f" in (da, db) else ("i" if "i" in (da, db) else "u")])
        return f"(sbx_cl({i}u, {g.group(1)}, {g.group(3)}) {g.group(2)} {g.group(3)})"
    body = CMP_RE.sub(sub, body)
    cmplog.ops = ops
    k = [0]

    def key(g):
        args = [a.strip() for a in g.group(2).split(",")[:8]]
        i = k[0]
        k[0] += 1
        return f"{g.group(1)}(sbx_kl({i}u, {', '.join(args)}), {', '.join(args[1:])}, (*spvDescriptorSet0.{g.group(3)}))"
    body = HASH_CALL_RE.sub(key, body)
    return head + CMPLOG_HOOK + body, n[0], k[0]


def add_cmplog(d):
    """Write <d>/cmplog.metal from <d>/ref.metal and record cmp_ids in meta.json."""
    text, n, nk = cmplog(open(os.path.join(d, "ref.metal")).read())
    open(os.path.join(d, "cmplog.metal"), "w").write(text)
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    meta["cmp_ids"] = n
    meta["key_ids"] = nk
    meta["cmp_ops"] = getattr(cmplog, "ops", [])      # per comparison id: [operator, domain f/i/u]
    json.dump(meta, open(mp, "w"), indent=1)
    return n


def instrument(msl):
    """Add the coverage buffer to main0, a block marker after every block-opening brace line in its body
    and a loop guard at the top of every loop body. Returns (text, blocks, loops).

    Block ids start at 1 (id 0 is the loop-guard flag). A brace line directly after `switch (...)` is skipped:
    code there would sit before the first case label and never run."""
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    if not m:
        return msl, 0, 0
    close = matching_paren(msl, m.end() - 1)
    msl = msl[:close] + f", device atomic_uint* sbx_cov [[buffer({COV_BUFFER_INDEX})]]" + msl[close:]
    line_start = msl.rfind("\n", 0, m.start()) + 1
    msl = msl[:line_start] + READS_HOOK + msl[line_start:]
    close += len(READS_HOOK)
    body_start = msl.index("{", close)
    head, body = msl[:body_start], msl[body_start:]
    lines = body.split("\n")
    closing, stack = {}, []                  # brace line -> its closing line
    for k, line in enumerate(lines):
        s = line.strip()
        if s.startswith("}") and stack:
            closing[stack.pop()] = s
        if s == "{":
            stack.append(k)
    out, block, loops, prev = [], 0, 0, ""
    for k, line in enumerate(lines):
        out.append(line)
        if line.strip() != "{":
            if line.strip():
                prev = line
            continue
        indent = line[:len(line) - len(line.lstrip())] + "    "
        if k == 0:
            out.append(indent + "uint sbx_iter = 0u;")
        p = prev.strip()
        if p.startswith("switch"):
            prev = line
            continue
        block += 1
        out.append(f"{indent}sbx_hit({block});")
        is_loop = p.startswith(("for (", "while (")) or (p == "do" and closing.get(k, "") != "} while (false);")
        if is_loop:
            out.append(indent + "SBX_LOOP_GUARD")
            loops += 1
        prev = line
    return head + "\n".join(out), block, loops


GUARD_LINE = re.compile(r"^(\s*)([\w.]+(?:\[\w+\])*) = (.*);\s*$")


def split_ternary(expr):
    """(cond, a, b) when expr is `cond ? a : b` at top level (nested ternaries in a or b allowed), else None."""
    depth, q, cond_end = 0, 0, None
    for i, c in enumerate(expr):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif depth == 0 and c == "?":
            if cond_end is None:
                cond_end = i
            q += 1
        elif depth == 0 and c == ":" and cond_end is not None:
            q -= 1
            if q == 0:
                return expr[:cond_end].strip(), expr[cond_end + 1:i].strip(), expr[i + 1:].strip()
    return None


def instrument_guards(msl, first_id, max_id=(1 << 15) - 1):   # ids above 32767: read bitmap
    """Rewrite lane-guarded writes in main0's body so coverage records whether each write took effect.
    Only plain assignments (no declaration) whose kept-value side is the target itself:
        X = c ? new : X;   ->  X = sbx_sel(id, c) ? new : X;
        X = c ? X : new;   ->  X = sbx_nsel(id, c) ? X : new;
    Ids start at first_id (after the block ids). Returns (text, guards, inverted, skipped_for_id_space)."""
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    if not m:
        return msl, 0, 0, 0
    start = msl.index("{", matching_paren(msl, m.end() - 1))
    head, lines = msl[:start], msl[start:].split("\n")
    gid, inverted, skipped = first_id, 0, 0
    for k, line in enumerate(lines):
        g = GUARD_LINE.match(line)
        if not g or "?" not in g.group(3):
            continue
        ind, lhs, rhs = g.groups()
        t = split_ternary(rhs)
        if not t:
            continue
        cond, a, b = t
        if b == lhs:
            form = f"{ind}{lhs} = sbx_sel({{}}, {cond}) ? {a} : {b};"
        elif a == lhs:
            form = f"{ind}{lhs} = sbx_nsel({{}}, {cond}) ? {a} : {b};"
        else:
            continue
        if gid > max_id:
            skipped += 1
            continue
        lines[k] = form.format(gid)
        inverted += a == lhs
        gid += 1
    return head + "\n".join(lines), gid - first_id, inverted, skipped


def block_of_offsets(msl):
    """For instrumented text: list of (offset, block id) at each line start of main0's body, by brace lines.
    Lines inside block k (innermost) map to k; used to place a mutation site in its block."""
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    if not m:
        return []
    start = msl.index("{", matching_paren(msl, m.end() - 1))
    marks, stack, pos = [], [], start
    for line in msl[start:].split("\n"):
        s = line.strip()
        if s.startswith("}"):
            if stack:
                stack.pop()
        hit = re.match(r"sbx_hit\((\d+)\);", s)
        if hit and stack and stack[-1] is None:
            stack[-1] = int(hit.group(1))
        marks.append((pos, next((b for b in reversed(stack) if b is not None), 0)))
        if s == "{":
            stack.append(None)
        pos += len(line) + 1
    return marks


def matching_paren(text, open_at):
    """Index of the ')' matching the '(' at open_at."""
    depth = 0
    for i in range(open_at, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses")


def sandbox(msl):
    """Wrap the argument of every reinterpret_cast to a device pointer in sbx(...); add the prelude."""
    text, n = _wrap_casts(msl)
    anchor = "using namespace metal;\n"
    if anchor in text:
        text = text.replace(anchor, anchor + SBX_PRELUDE, 1)
    else:
        text = SBX_PRELUDE + text
    return text, n


def _wrap_casts(msl):
    out, pos, n = [], 0, 0
    for m in re.finditer(r"reinterpret_cast<", msl):
        if m.start() < pos:
            continue
        # the template argument, with nested <>
        depth, i = 1, m.end()
        while depth:
            if msl[i] == "<":
                depth += 1
            elif msl[i] == ">":
                depth -= 1
            i += 1
        target = msl[m.end():i - 1]
        if msl[i] != "(" or "device" not in target or not target.rstrip().endswith("*"):
            continue
        close = matching_paren(msl, i)
        inner, k = _wrap_casts(msl[i + 1:close])
        out.append(msl[pos:i + 1])
        out.append("sbx(" + inner + ")")
        pos = close
        n += 1 + k
    out.append(msl[pos:])
    return "".join(out), n


MEMBER_RE = re.compile(r"^\s*([\w:<>, ]+?)\s+(\w+)\s*\[\[([^\]]*)\]\];", re.M)


def struct_members(msl, name):
    m = re.search(r"struct " + name + r"\s*\{(.*?)\};", msl, re.S)
    if not m:
        return []
    return [(t.strip(), n, a.strip()) for t, n, a in MEMBER_RE.findall(m.group(1))]


def base_type(t):
    t = t.replace("packed_", "")
    m = re.match(r"(float|half|uint|int|ushort|short|bool)(\d?)", t)
    return (m.group(1), int(m.group(2) or 1)) if m else (t, 1)


def gen_vs(inputs):
    """A vertex shader drawing one full-screen triangle that feeds every fragment input."""
    lines = ["#include <metal_stdlib>", "using namespace metal;", "struct vs_out {",
             "    float4 pos [[position]];"]
    body = []
    for k, (t, n, attrs) in enumerate(inputs):
        loc = re.search(r"user\(\w+\)", attrs)
        if not loc:
            continue
        lines.append(f"    {t} {n} [[{loc.group(0)}]];")
        bt, width = base_type(t)
        if bt in ("float", "half"):
            comps = [f"(uv.x * 3.0 - 1.0 + {0.25 * k:.2f})", f"(uv.y * 2.0 - 0.5 - {0.1 * k:.2f})",
                     f"((uv.x + uv.y) * 0.7 + {0.05 * k:.2f})", f"(1.0 - uv.x * 0.5 + {0.03 * k:.2f})"]
        else:
            comps = [f"{3 + 7 * k}", f"{1 + 5 * k}", f"{2 + 3 * k}", f"{11 + k}"]
        body.append(f"    o.{n} = {t}({', '.join(comps[:width])});")
    lines.append("};")
    lines += ["vertex vs_out vs_main(uint vid [[vertex_id]])", "{",
              "    float2 p = float2(float((vid << 1) & 2), float(vid & 2));",
              "    float2 uv = p;", "    vs_out o;",
              "    o.pos = float4(p * 2.0 - 1.0, 0.2 + 0.3 * p.x, 1.0);"] + body + ["    return o;", "}"]
    return "\n".join(lines) + "\n"


def outputs(msl):
    outs = []
    for t, n, attrs in struct_members(msl, "main0_out"):
        c = re.search(r"color\((\d+)\)", attrs)
        bt, width = base_type(t)
        if c:
            outs.append({"name": n, "kind": "color", "slot": int(c.group(1)), "type": bt, "width": width})
        elif "depth" in attrs:
            outs.append({"name": n, "kind": "depth", "slot": 0, "type": bt, "width": width})
        elif "sample_mask" in attrs:
            outs.append({"name": n, "kind": "sample_mask", "slot": 0, "type": bt, "width": width})
    return outs


def prepare(name, rec, spirv_dir, fssm_dir, out_dir):
    try:
        return _prepare(name, rec, spirv_dir, fssm_dir, out_dir)
    except Exception as e:                # one shader must never stop the batch
        return {"name": name, "stage": rec.get("stage"), "status": "exception",
                "failure": runlog.exception(e, LOG, f"prepare {name}")}


def _prepare(name, rec, spirv_dir, fssm_dir, out_dir):
    res = {"name": name, "stage": rec.get("stage")}
    spv = os.path.join(spirv_dir, name + ".spv")
    if not os.path.exists(spv):
        res["status"] = "no_spv"
        LOG.warning("prepare %s: no SPIR-V at %s (translation failed or not run)", name, spv)
        return res
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fixed, msl_path = os.path.join(tmp, "m.spv"), os.path.join(tmp, "m.metal")
        r = runlog.run([sys.executable, "-I", SET0, spv, fixed], 600, LOG, f"set0 {name}")
        if not r["ok"]:
            res.update(status="set0_failed", failure=runlog.failure(r))
            return res
        r = runlog.run(["spirv-cross", "--msl", "--msl-version", "30100", "--msl-argument-buffers",
                        "--msl-argument-buffer-tier", "1", "--msl-decoration-binding", fixed, "--output", msl_path],
                       600, LOG, f"spirv-cross {name}")
        if not r["ok"]:
            res.update(status="spirv_cross_failed", failure=runlog.failure(r))
            return res
        msl = open(msl_path).read()
    text, casts = sandbox(msl)
    text, blocks, loops = instrument(text)
    text, guards, inverted, no_room = instrument_guards(text, blocks + 1)
    if no_room:
        LOG.warning("prepare %s: %d lane guards left uninstrumented (coverage id space full)", name, no_room)
    open(os.path.join(d, "ref.metal"), "w").write(text)
    stage = rec.get("stage")
    meta = {
        "name": name, "stage": stage,
        "local_size": rec.get("local_size"),
        "user_sgprs": rec.get("user_sgprs"),
        "user_sgpr_reads": rec.get("user_sgpr_reads") or [],
        "wave64": rec.get("wave64"),
        "lds_bytes": rec.get("lds_bytes"),
        "push_constant_bytes": rec.get("push_constant_bytes", 144),
        "fssm": os.path.join(fssm_dir, name + ".fssm"),
        "sandboxed_casts": casts,
        "cov_blocks": blocks,
        "loop_guards": loops,
        "cov_guards": guards,
        "cov_guards_inverted": inverted,
        "cov_buffer_index": COV_BUFFER_INDEX,
    }
    if stage == "ps":
        meta["outputs"] = outputs(msl)
        open(os.path.join(d, "vs.metal"), "w").write(gen_vs(struct_members(msl, "main0_in")))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    res["cmp_ids"] = add_cmplog(d)
    res.update(status="ok", casts=casts, msl_lines=text.count("\n"), blocks=blocks, loops=loops, guards=guards,
               guards_inverted=inverted)
    if casts == 0 and "reinterpret_cast<" in msl:
        LOG.warning("prepare %s: reinterpret_casts present but none sandboxed: check the cast forms", name)
    LOG.debug("prepare %s: ok, %d casts sandboxed, %d blocks, %d loops, %d lane guards (%d inverted)", name, casts,
              blocks, loops, guards, inverted)
    return res


def main(argv):
    if len(argv) >= 2 and argv[0] == "--add-cmplog":            # update an existing prepared set: prepare.py --add-cmplog <prep dir>
        tot = 0
        for n in sorted(os.listdir(argv[1])):
            if os.path.isfile(os.path.join(argv[1], n, "ref.metal")):
                tot += add_cmplog(os.path.join(argv[1], n))
        print(f"cmplog.metal written, {tot} comparison ids")
        return 0
    if len(argv) < 3:
        print(__doc__)
        return 2
    spirv_dir, fssm_dir, out_dir = argv[:3]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 6
    only = set(argv[argv.index("--only") + 1:]) if "--only" in argv else None
    os.makedirs(out_dir, exist_ok=True)
    global LOG
    LOG = runlog.setup(os.path.join(out_dir, "prepare.jsonl"), "prepare", verbose="--verbose" in argv)
    index_path = os.path.join(spirv_dir, "index.json")
    if not os.path.exists(index_path):
        LOG.error("no %s: the translator writes it at the end of a run (a crashed run leaves none)", index_path)
        return 1
    index = json.load(open(index_path))
    items = index.get("shaders", index) if isinstance(index, dict) else {r["name"]: r for r in index}
    names = [n for n in items if not only or n in only]
    os.makedirs(out_dir, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex, open(os.path.join(out_dir, "prepare.jsonl"), "w") as f:
        results = list(ex.map(lambda n: prepare(n, items[n], spirv_dir, fssm_dir, out_dir), names))
        for r in results:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    LOG.info("%d shaders: %s", len(results), dict(Counter(r["status"] for r in results)))
    def reason(r):
        if r["status"] == "ok":
            return None
        f = r.get("failure") or {}
        msg = f.get("exception") or next((l for l in (f.get("stderr") or "").splitlines() if l.strip()), "")
        return f"{r['status']}: {msg}"[:160]
    runlog.summarise_failures(results, reason, LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
