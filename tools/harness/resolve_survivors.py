"""Resolve the mutants that survived a run_set.py run: prove them equivalent or find an input that kills them.

For every surviving mutant (run_set records where it was: line, col, original text, replacement):
  1. equivalence: reference and mutant are compiled to optimised LLVM IR by the Metal front end (-O2, coverage off).
     Identical IR (ignoring the module header) means no input can tell them apart: the mutant is equivalent and
     it leaves the denominator of the kill rate.
  2. hunt: batches of input variants derived from the shader's inputs.json (fuzz_inputs.py) run on reference vs
     mutant until one differs (the harness stops at the first difference). Only title-memory words the shader reads
     and its non-pointer user SGPRs are changed. The killing variant is appended to the shader's inputs.json, so
     later runs keep it.
  3. whatever is left is reported as unresolved with its source line.

Usage:
    python3 -I resolve_survivors.py <prep dir> <run_set results.jsonl> [--jobs N] [--rounds N] [--batch N]
Writes <results>.resolved.jsonl (one row per survivor) and logs to <results>.resolved.jsonl.log.
"""
import json
import os
import random
import re
import struct
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runlog  # noqa: E402
import run_set  # noqa: E402
import fuzz_inputs  # noqa: E402
import z3_equiv  # noqa: E402

LOG = None
INPUT_LOCKS = {}
LOCK = threading.Lock()


def mutant_text(msl, m):
    """Rebuild a run_set mutant from its record."""
    starts = [0] + [i + 1 for i, c in enumerate(msl) if c == "\n"]
    off = starts[m["line"] - 1] + m["col"]
    if msl[off:off + len(m["orig"])] != m["orig"]:
        raise ValueError(f"site text mismatch at line {m['line']} col {m['col']}: "
                         f"{msl[off:off + len(m['orig'])]!r} != {m['orig']!r}")
    text = msl[:off] + m["rep"] + msl[off + len(m["orig"]):]
    return text.replace("using namespace metal;\n", "using namespace metal;\n" + run_set.MUT_HELPERS, 1)


LOUD_HELPERS = """
template<typename T, typename U> static inline T spvLoud(T a, U b) { return T(12345.0); }
template<typename T, typename U, typename V> static inline T spvLoud3(T a, U b, V c) { return T(12345.0); }
"""


def loud_text(msl, m):
    """The mutant's site made loud: the float op returns 12345, the integer constant gets far-off bits. An input
    that tells this apart from the reference is one where the site is visible at an output."""
    loud = dict(m)
    if m["kind"] == "float_op":
        loud["rep"] = "spvLoud3(" if m["orig"] == "fma(" else "spvLoud("
    else:
        loud["rep"] = str((int(m["orig"]) ^ 0x5A5A5A5A) & 0xFFFFFFFF)
    text = mutant_text(msl, loud)
    return text.replace("using namespace metal;\n", "using namespace metal;\n" + LOUD_HELPERS, 1)


def ir(path, tmp, tag):
    out = os.path.join(tmp, tag + ".ll")
    # -fno-fast-math: the harness compiles with mathMode = .safe; the CLI default is fast math, which folds x*0,
    # NaN checks etc. and would let two shaders that differ under .safe produce identical IR (unsound proof).
    r = runlog.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", "-fno-fast-math", "-S", "-emit-llvm", "-w",
                    "-DSBX_BASE=0x100000000ul", "-DSBX_MASK=0xFFFFFFul", path, "-o", out], 300, LOG, f"ir {tag}")
    if not r["ok"]:
        return None
    return _norm(open(out).read())


def _norm(text):
    # drop everything that names the source file: module header lines and metadata strings with the path
    out = []
    for l in text.splitlines():
        if l.startswith(("; ModuleID", "source_filename")):
            continue
        out.append(re.sub(r'!\{!"[^"]*\.(?:metal|ll)"\}', '!{!"<source>"}', l))
    return "\n".join(out)


AIR_OPT = None


def ir_o3(ll_text, tmp, tag):
    """Second stage: Apple's air-opt -O3 over the -O2 IR (it reads AIR directly)."""
    global AIR_OPT
    if AIR_OPT is None:
        r = runlog.run(["xcrun", "-sdk", "macosx", "-f", "air-opt"], 30, LOG, "find air-opt")
        AIR_OPT = (r.get("stdout") or "").strip() if r["ok"] else ""
    if not AIR_OPT:
        return None
    src, dst = os.path.join(tmp, tag + ".in.ll"), os.path.join(tmp, tag + ".o3.ll")
    open(src, "w").write(ll_text)
    r = runlog.run([AIR_OPT, "-O3", "-S", src, "-o", dst], 300, LOG, f"air-opt {tag}")
    return _norm(open(dst).read()) if r["ok"] else None


def read_set(d, inputs, tmp):
    """Title-memory chunks read across the shader's input variants and the comparison log of each variant
    (A only: cmplog.metal with coverage, reads and comparison logging)."""
    res = fuzz_inputs.run_batch(d, inputs, tmp, "reads")
    if res.get("status") != "ok":
        return 4, set(), [[] for _ in inputs], [[] for _ in inputs]
    return (res.get("read_shift", 4), set(res.get("reads", [])), [row.get("cmp", []) for row in res["variants"]],
            [row.get("keys", []) for row in res["variants"]])


def cmp_lines(d):
    """Comparison id -> line number in ref.metal (cmplog.metal is ref.metal plus the hook lines before main0)."""
    cl = os.path.join(d, "cmplog.metal")
    if not os.path.exists(cl):
        return {}
    import prepare
    shift = prepare.CMPLOG_HOOK.count("\n")
    out = {}
    for k, line in enumerate(open(cl).read().split("\n"), 1):
        for i in re.findall(r"sbx_cl\((\d+)u,", line):
            out[int(i)] = k - shift
    return out


def random_variant(rng, parents, words, push_targets, dictionary, small=False):
    """A parent with a few read words / push dwords changed and sometimes new texture contents."""
    parent = rng.choice(parents)
    mem, push = dict(map(tuple, parent["mem"])), dict(map(tuple, parent["push"]))
    for _ in range(rng.choice((1, 1, 2, 3) if small else (1, 2, 3, 4, 8, 16))):
        if push_targets and (not words or rng.random() < 0.25):
            i = rng.choice(push_targets)
            push[i] = fuzz_inputs.new_value(rng, push.get(i, 0), [], dictionary)
        elif words:
            i = rng.choice(words)
            mem[i] = fuzz_inputs.new_value(rng, mem.get(i, fuzz_inputs.seeded_word(parent["seed"], i)), [], dictionary)
    v = {"seed": parent["seed"], "mem": sorted(mem.items()), "push": sorted(push.items())}
    if parent.get("hash"):
        v["hash"] = parent["hash"]
    tex = parent.get("tex_seed")
    if rng.random() < (0.1 if small else 0.25):
        tex = rng.randrange(1, 1 << 31)
    if tex is not None:
        v["tex_seed"] = tex
    return v


def _cl_maps(d, path, variants, tmp, tag):
    """Comparison logs of `path` (a cmplog build, run alone) per variant: list of {id: (a, b)}."""
    bp = os.path.join(tmp, f"cl_{tag}.json")
    json.dump(variants, open(bp, "w"))
    cmd = [fuzz_inputs.HARNESS, "--meta", os.path.join(d, "meta.json"), "--a", path, "--b", "none", "--inputs", bp,
           "--all", "1", "--coverage", "1", "--cmplog", "1"]
    vs = os.path.join(d, "vs.metal")
    if os.path.exists(vs):
        cmd += ["--vs", vs]
    r = runlog.run(cmd, 300, LOG, f"cmplog {os.path.basename(d)} {tag}")
    lines = [l for l in (r.get("stdout") or "").splitlines() if l.startswith("{")]
    if not lines:
        return None
    res = json.loads(lines[-1])
    if res.get("status") != "ok":
        return None
    return [{c[0]: (c[1], c[2]) for c in row.get("cmp", [])} for row in res["variants"]]


def _gate_fitness(R, M):
    """Over comparisons the mutation reached (operands differ between reference and mutant): 0 when the two sit on
    different sides of the right operand (the outcome can flip), else the smallest distance to it."""
    best = None
    for i, (ar, br) in R.items():
        if i not in M:
            continue
        am, bm = M[i]
        if (ar, br) == (am, bm):
            continue
        if br == bm and ((ar > br) != (am > bm) or (ar == br) != (am == bm)):
            return 0
        d = min(abs(ar - br), abs(am - bm))
        best = d if best is None else min(best, d)
    return best


def _nudge(rng, v):
    """A small move of a word: float ulps, a relative float step, an integer step."""
    k = rng.random()
    if k < 0.4:
        return (v + rng.choice((-1, 1)) * (1 << rng.randrange(0, 22))) & 0xFFFFFFFF
    if k < 0.7:
        f = struct.unpack("<f", struct.pack("<I", v))[0]
        if f != f or f in (float("inf"), float("-inf")):
            f = 1.0
        f *= 1 + rng.choice((-1, 1)) * 2.0 ** -rng.randrange(1, 20)
        try:
            return struct.unpack("<I", struct.pack("<f", f))[0]
        except OverflowError:
            return v
    return (v + rng.choice((-1, 1)) * rng.randrange(1, 64)) & 0xFFFFFFFF


def gate_hunt(d, m, msl, parents, words, push_targets, rng, args, tmp, k):
    """Branch distance: push the inputs until a comparison the mutation reached flips for the mutant and not for
    the reference (or the other way round), then confirm on the outputs. Returns the killing variant or None."""
    import prepare
    mut_cl = os.path.join(tmp, f"mcl{k}.metal")
    open(mut_cl, "w").write(prepare.cmplog(mutant_text(msl, m))[0])
    ref_cl = os.path.join(d, "cmplog.metal")
    mp = os.path.join(tmp, f"m{k}.metal")
    pool = list(parents)[:args["batch"]]
    R, M = _cl_maps(d, ref_cl, pool, tmp, f"r{k}_0"), _cl_maps(d, mut_cl, pool, tmp, f"m{k}_0")
    if R is None or M is None:
        return None, "cmplog failed"
    scored = [(_gate_fitness(r, mm), v) for r, mm, v in zip(R, M, pool)]
    scored = [x for x in scored if x[0] is not None]
    if not scored:
        return None, "no comparison reached by the mutation"
    for rnd in range(args["gate_rounds"]):
        scored.sort(key=lambda x: x[0])
        flips = [v for f, v in scored if f == 0]
        if flips:
            bp = os.path.join(tmp, f"gate{k}_{rnd}.json")
            json.dump(flips[:args["batch"]], open(bp, "w"))
            r = run_set.harness(d, os.path.join(d, "ref.metal"), mp, inputs=bp)
            if r.get("status") == "ok" and not r.get("equal"):
                cand = flips[r.get("first_diff_variant", 0)]
                if confirm_kill(d, mp, cand, tmp, f"g{k}_{rnd}"):
                    return cand, f"gate flipped in round {rnd}"
            scored = [x for x in scored if x[0] != 0] or scored
        best = [v for _, v in scored[:4]]
        batch = []
        for _ in range(args["batch"]):
            par = rng.choice(best)
            mem, push = dict(map(tuple, par["mem"])), dict(map(tuple, par["push"]))
            for _ in range(rng.choice((1, 1, 2, 3))):
                if push_targets and rng.random() < 0.2:
                    i = rng.choice(push_targets)
                    push[i] = _nudge(rng, push.get(i, fuzz_inputs.seeded_push(par["seed"], i, {})))
                elif words:
                    i = rng.choice(words)
                    mem[i] = _nudge(rng, mem.get(i, fuzz_inputs.seeded_word(par["seed"], i)))
            v = {"seed": par["seed"], "mem": sorted(mem.items()), "push": sorted(push.items())}
            for key in ("tex_seed", "hash"):
                if par.get(key) is not None:
                    v[key] = par[key]
            if rng.random() < 0.1:
                v["tex_seed"] = rng.randrange(1, 1 << 31)
            batch.append(v)
        R, M = _cl_maps(d, ref_cl, batch, tmp, f"r{k}_{rnd + 1}"), _cl_maps(d, mut_cl, batch, tmp, f"m{k}_{rnd + 1}")
        if R is None or M is None:
            return None, "cmplog failed"
        new = [(_gate_fitness(r, mm), v) for r, mm, v in zip(R, M, batch)]
        scored = sorted(scored + [x for x in new if x[0] is not None], key=lambda x: x[0])[:args["batch"]]
    return None, f"no flip after {args['gate_rounds']} rounds (closest {scored[0][0] if scored else None})"


def confirm_kill(d, mp, variant, tmp, tag):
    """A kill counts only if the input is race-free (reference vs reference equal 3 times) and the difference
    repeats (reference vs mutant unequal 2 more times)."""
    ref = os.path.join(d, "ref.metal")
    bp = os.path.join(tmp, f"confirm_{tag}.json")
    json.dump([variant], open(bp, "w"))
    for _ in range(3):
        r = run_set.harness(d, ref, ref, inputs=bp)
        if r.get("status") != "ok" or not r.get("equal") or r.get("loop_guard_a") or r.get("loop_guard_b"):
            return False
    for _ in range(2):
        r = run_set.harness(d, ref, mp, inputs=bp)
        if r.get("status") != "ok" or r.get("equal") is not False:
            return False
    return True


def resolve_shader(d, survivors, args):
    try:
        return _resolve_shader(d, survivors, args)
    except Exception as e:
        fail = runlog.exception(e, LOG, f"resolve {os.path.basename(d)}")
        return [dict(m, resolution="error", failure=fail) for m in survivors]


def _resolve_shader(d, survivors, args):
    name = os.path.basename(d)
    ref = os.path.join(d, "ref.metal")
    msl = open(ref).read()
    meta = json.load(open(os.path.join(d, "meta.json")))
    inp_path = os.path.join(d, "inputs.json")
    inputs = json.load(open(inp_path)) if os.path.exists(inp_path) else [{"seed": s, "mem": [], "push": []} for s in (1, 2, 3)]
    rng = random.Random(name + ":resolve")
    push_targets = [r["sgpr"] for r in meta.get("user_sgpr_reads") or []
                    if r.get("sgpr", 99) < 32 and not str(r.get("kind", "")).startswith("ptr")]
    dictionary = fuzz_inputs.shader_dictionary(msl)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        ref_ir = ir(ref, tmp, "ref")
        ref_o3 = None
        shift, chunks, cmps, keyss = read_set(d, inputs, tmp)
        id_line = cmp_lines(d)
        specs = fuzz_inputs.field_specs(msl)
        words = sorted({(c << shift) // 4 + k for c in chunks for k in range((1 << shift) // 4)})
        for k, m in enumerate(survivors):
            row = dict(m, name=name)
            try:
                text = mutant_text(msl, m)
            except ValueError as e:
                row.update(resolution="error", error=str(e))
                out.append(row)
                continue
            mp = os.path.join(tmp, f"m{k}.metal")
            open(mp, "w").write(text)
            mir = ir(mp, tmp, f"m{k}") if ref_ir is not None else None
            if mir is not None and mir == ref_ir:
                row.update(resolution="equivalent", proof="metal -O2 IR identical")
                out.append(row)
                continue
            if mir is not None:
                if ref_o3 is None:
                    ref_o3 = ir_o3(ref_ir, tmp, "ref") or ""
                mo3 = ir_o3(mir, tmp, f"m{k}")
                if ref_o3 and mo3 is not None and mo3 == ref_o3:
                    row.update(resolution="equivalent", proof="air-opt -O3 IR identical")
                    out.append(row)
                    continue
            row["ir_differs"] = mir is not None
            ok, why = z3_equiv.prove_equivalent(msl, m)
            row["z3"] = why
            if ok:
                row.update(resolution="equivalent", proof=why)
                out.append(row)
                continue
            killed = None
            # stage 1, observability: inputs where the loud site changes an output
            lp = os.path.join(tmp, f"l{k}.metal")
            open(lp, "w").write(loud_text(msl, m))
            observe = []
            oq, hq = [], []
            for par, keys in zip(inputs, keyss):
                hq += fuzz_inputs.hash_variants(par, keys, words, msl, rng)
            for par, cmp in zip(inputs, cmps):
                near = sorted(cmp, key=lambda c: abs(id_line.get(c[0], 10 ** 9) - m["line"]))
                oq += fuzz_inputs.i2s_variants(par, near, words, specs, push_targets, meta,
                                               args["batch"] * args["rounds"] // max(len(inputs), 1) + 8, rng,
                                               ordered=True)
            rng.shuffle(oq)
            oq = hq + oq
            row["observe_i2s"] = len(oq)
            for rnd in range(args["rounds"] * 2):
                if rnd == 0:
                    batch = list(inputs)
                else:
                    batch = oq[:args["batch"] // 2]
                    oq = oq[args["batch"] // 2:]
                    while len(batch) < args["batch"]:
                        batch.append(random_variant(rng, inputs, words, push_targets, dictionary))
                bp = os.path.join(tmp, f"obs{k}_{rnd}.json")
                json.dump(batch, open(bp, "w"))
                r = run_set.harness(d, ref, lp, inputs=bp, extra=["--all", "1"])
                if r.get("status") != "ok":
                    row.update(resolution="observe_error", status=r.get("status"), phase=r.get("phase"),
                               reason=run_set.first_error(r))
                    break
                observe += [batch[v["i"]] for v in r.get("variants", []) if v.get("equal") is False]
                if len(observe) >= 4:
                    break
            row["observable_inputs"] = len(observe)
            if row.get("resolution") == "observe_error":
                out.append(row)
                continue
            # the loud stand-in is not a superset of every mutant (e.g. a bit-field width): hunt anyway,
            # from the plain inputs when no observable input was found
            row["unobservable_loud"] = not observe
            parents = observe[:16] if observe else list(inputs)
            # input-to-state first: comparisons logged under each input, closest to the mutated line first
            queue = []
            _, _, par_cmps, par_keys = read_set(d, parents, tmp)
            for par, keys in zip(parents, par_keys):
                queue += fuzz_inputs.hash_variants(par, keys, words, msl, rng, limit=8)
            for par, cmp in zip(parents, par_cmps):
                near = sorted(cmp, key=lambda c: abs(id_line.get(c[0], 10 ** 9) - m["line"]))
                queue += fuzz_inputs.i2s_variants(par, near[:24], words, specs, push_targets, meta,
                                                  args["batch"] * 2, rng, ordered=True)
            row["i2s_candidates"] = len(queue)
            q_rounds = (len(queue) + args["batch"] - 1) // args["batch"]
            for rnd in range(args["rounds"] + min(q_rounds, args["rounds"])):
                batch = queue[:args["batch"]]
                queue = queue[args["batch"]:]
                if rnd == 0:
                    batch = (parents + batch)[:args["batch"]]
                while len(batch) < args["batch"]:
                    batch.append(random_variant(rng, parents, words, push_targets, dictionary, small=True))
                bp = os.path.join(tmp, f"hunt{k}_{rnd}.json")
                json.dump(batch, open(bp, "w"))
                r = run_set.harness(d, ref, mp, inputs=bp)
                if r.get("status") == "ok" and not r.get("equal"):
                    cand = batch[r.get("first_diff_variant", 0)]
                    if confirm_kill(d, mp, cand, tmp, f"{k}_{rnd}"):
                        killed = cand
                        row.update(resolution="killed_by_hunt", hunt_round=rnd)
                        break
                    row["unconfirmed_kills"] = row.get("unconfirmed_kills", 0) + 1
                if r.get("status") != "ok":
                    row.update(resolution="hunt_error", status=r.get("status"), phase=r.get("phase"),
                               reason=run_set.first_error(r))
                    break
            if killed is not None:
                killed = dict(killed, origin="hunt")
                with LOCK:
                    cur = json.load(open(inp_path)) if os.path.exists(inp_path) else list(inputs)
                    cur.append(killed)
                    json.dump(cur, open(inp_path, "w"))
                inputs.append(killed)
            if row.get("resolution") is None and args.get("gate_rounds"):
                gk, why = gate_hunt(d, m, msl, parents, words, push_targets, rng, args, tmp, k)
                row["gate"] = why
                if gk is not None:
                    killed = dict(gk, origin="hunt")
                    with LOCK:
                        cur = json.load(open(inp_path)) if os.path.exists(inp_path) else list(inputs)
                        cur.append(killed)
                        json.dump(cur, open(inp_path, "w"))
                    inputs.append(killed)
                    row["resolution"] = "killed_by_gate"
            if row.get("resolution") is None and row.get("unobservable_loud"):
                row["resolution"] = "unobservable"
            row.setdefault("resolution", "unresolved")
            out.append(row)
    return out


def recheck_one(d, rows, args):
    """Stage 3 only, for rows already marked unobservable/unresolved."""
    try:
        name = os.path.basename(d)
        msl = open(os.path.join(d, "ref.metal")).read()
        meta = json.load(open(os.path.join(d, "meta.json")))
        inp_path = os.path.join(d, "inputs.json")
        inputs = json.load(open(inp_path)) if os.path.exists(inp_path) else [{"seed": s, "mem": [], "push": []} for s in (1, 2, 3)]
        push_targets = [r["sgpr"] for r in meta.get("user_sgpr_reads") or []
                        if r.get("sgpr", 99) < 32 and not str(r.get("kind", "")).startswith("ptr")]
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            shift, chunks, _, _ = read_set(d, inputs, tmp)
            words = sorted({(c << shift) // 4 + k for c in chunks for k in range((1 << shift) // 4)})
            for k, m in enumerate(rows):
                rng = random.Random(f"{name}:{m['line']}:{m['col']}:gate")
                mp = os.path.join(tmp, f"m{k}.metal")
                open(mp, "w").write(mutant_text(msl, m))
                gk, why = gate_hunt(d, m, msl, inputs, words, push_targets, rng, args, tmp, k)
                row = dict(m, gate=why)
                if gk is not None:
                    killed = dict(gk, origin="hunt")
                    with LOCK:
                        cur = json.load(open(inp_path)) if os.path.exists(inp_path) else list(inputs)
                        cur.append(killed)
                        json.dump(cur, open(inp_path, "w"))
                    inputs.append(killed)
                    row["resolution"] = "killed_by_gate"
                out.append(row)
        return out
    except Exception as e:
        fail = runlog.exception(e, LOG, f"recheck {os.path.basename(d)}")
        return [dict(m, gate="error", failure=fail) for m in rows]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    prep, results = argv[0], argv[1]

    def opt(k, default):
        return type(default)(argv[argv.index(k) + 1]) if k in argv else default
    args = {"rounds": opt("--rounds", 8), "batch": opt("--batch", 32), "gate_rounds": opt("--gate-rounds", 10)}
    jobs = opt("--jobs", 4)
    global LOG
    if "--recheck" in argv:          # stage 3 only over an existing .resolved.jsonl: writes <that>.gate.jsonl
        out_path = results + ".gate.jsonl"
        LOG = runlog.setup(out_path, "resolve_survivors_gate", verbose="--verbose" in argv)
        run_set.LOG = LOG
        fuzz_inputs.LOG = LOG
        rows = [json.loads(l) for l in open(results)]
        todo = [r for r in rows if r.get("resolution") in ("unobservable", "unresolved")]
        LOG.info("gate recheck: %d of %d rows, %s, jobs %d", len(todo), len(rows), args, jobs)
        from concurrent.futures import as_completed
        done = []
        with ThreadPoolExecutor(jobs) as ex, open(out_path, "w") as f:
            futs = [ex.submit(recheck_one, os.path.join(prep, r["name"]), [r], args) for r in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                for row in fut.result():
                    done.append(row)
                    f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 25 == 0:
                    LOG.info("  %d/%d: %s", i, len(todo), dict(Counter(r["resolution"] for r in done)))
        LOG.info("gate recheck: %s", dict(Counter(r["resolution"] for r in done)))
        return 0
    out_path = results + ".resolved.jsonl"
    LOG = runlog.setup(out_path, "resolve_survivors", verbose="--verbose" in argv)
    run_set.LOG = LOG
    fuzz_inputs.LOG = LOG
    by_shader = {}
    for l in open(results):
        r = json.loads(l)
        for m in r.get("mutants", []):
            if m.get("verdict") == "survived":
                by_shader.setdefault(r["name"], []).append(m)
    n = sum(len(v) for v in by_shader.values())
    LOG.info("resolve: %d survivors in %d shaders, %s, jobs %d", n, len(by_shader), args, jobs)
    rows = []
    # one work unit per survivor (a shader with many survivors doesn't hold up the rest); rows written as they finish
    units = [(name, m) for name, ms in by_shader.items() for m in ms]
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(jobs) as ex, open(out_path, "w") as f:
        futs = [ex.submit(resolve_shader, os.path.join(prep, name), [m], args) for name, m in units]
        for i, fut in enumerate(as_completed(futs), 1):
            for row in fut.result():
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0:
                LOG.info("  %d/%d survivors: %s", i, len(units), dict(Counter(r["resolution"] for r in rows)))
    c = Counter(r["resolution"] for r in rows)
    LOG.info("survivors: %s", dict(c))
    tot = Counter()
    for l in open(results):
        r = json.loads(l)
        for m in r.get("mutants", []):
            tot[m.get("verdict")] += 1
    killed = tot["killed"] + c["killed_by_hunt"] + c["killed_by_gate"]
    denom = tot["killed"] + tot["survived"] - c["equivalent"]
    LOG.info("after resolving: killed %d of %d non-equivalent mutants (%.1f%%); equivalent %d; unresolved %d; "
             "unobservable %d; errors %d", killed, denom, 100 * killed / max(denom, 1), c["equivalent"], c["unresolved"],
             c["unobservable"],
             c["error"] + c["hunt_error"] + c["observe_error"])
    for r in rows:
        if r["resolution"] == "unresolved":
            LOG.debug("unresolved %s line %d (%s, %s): %s", r["name"], r["line"], r["kind"], r.get("why"), r.get("src"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
