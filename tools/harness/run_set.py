"""Run the Metal harness over a prepared set: determinism, then mutation testing.

For each shader prepared by prepare.py:
  1. determinism: the reference against itself (seed 1). Unequal = nondeterministic; its comparisons can't be trusted.
  2. exercised: did the shader write anything (render target, writable texture, title memory)?
  3. mutation: up to --mutants copies of the reference, each with one small change in main0 (a float + / * / -
     swapped, an integer constant off by one). A mutant is killed when any output differs, survives when every
     output matches, invalid when it doesn't compile. Kill rate = how well the generated inputs exercise the code:
     a candidate that is wrong in the same way would slip through when a mutant survives.

Usage:
    python3 -I run_set.py <prep dir> <out.jsonl> [--jobs N] [--mutants N] [--limit N] [--seeds N] [--no-inputs] [--verbose]
Shaders with an inputs.json (fuzz_inputs.py) run on those variants instead of plain seeds.

Logging: <out.jsonl>.log has every harness call (command, duration) and, for every failure, the harness's phase log
(the last phase it reached), its stderr and exit code. Failed records in <out.jsonl> carry the same context under
"failure"; failed mutants carry status, phase and the first error line.
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HARNESS = os.environ.get("SBX_HARNESS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import block_of_offsets  # noqa: E402
import runlog  # noqa: E402

LOG = None   # set in main()
USE_INPUTS = True   # use <shader>/inputs.json (fuzz_inputs.py) when present; --no-inputs turns it off
PINNED = False      # --pinned: test exactly <shader>/mutants_pinned.json + mutants_holdout.json (see --pin)


def harness(d, a, b, seed=1, timeout=180, coverage=False, seeds=None, inputs=None, extra=None):
    """One harness process. With `seeds`, every seed runs in that process (A and B compiled once). With `inputs`
    (a fuzz_inputs.py inputs.json), its variants run instead of plain seeds."""
    seed_arg = ",".join(str(x) for x in (seeds or (seed,)))
    cmd = [HARNESS, "--meta", os.path.join(d, "meta.json"), "--a", a, "--b", b]
    if inputs:
        cmd += ["--inputs", inputs]
        seed_arg = "inputs.json"
    else:
        cmd += ["--seeds", seed_arg]
    if coverage:
        cmd += ["--coverage", "1"]
    cmd += list(extra or [])
    vs = os.path.join(d, "vs.metal")
    if os.path.exists(vs):
        cmd += ["--vs", vs]
    what = f"harness {os.path.basename(d)} a={os.path.basename(a)} b={os.path.basename(b)} seeds={seed_arg}"
    r = runlog.run(cmd, timeout, LOG, what)
    lines = [l for l in (r.get("stdout") or "").strip().splitlines() if l.startswith("{")]
    if r["timed_out"]:
        res = {"status": "process_timeout"}
    elif not lines:
        res = {"status": "crash"}
    else:
        try:
            res = json.loads(lines[-1])
        except json.JSONDecodeError as e:
            res = {"status": "bad_json", "error": str(e)}
    if res.get("status") != "ok":
        # the harness's phase log (stderr) names the last phase reached: keep it with the result
        res["failure"] = runlog.failure(r)
        phase_lines = [l for l in (r.get("stderr") or "").splitlines() if l.startswith("[harness")]
        res.setdefault("phase", phase_lines[-1].split("] ", 1)[-1][:120] if phase_lines else "no phase log")
        LOG.warning("%s: status %s, phase %s, %s", what, res["status"], res.get("phase"),
                    first_error(res) or "no error message")
    return res


def first_error(res):
    """The first error message a harness result carries, shortened."""
    for k in ("compile_error", "pipeline_error_a", "pipeline_error_b", "gpu_error_a", "gpu_error_b", "reason",
              "unsupported", "error"):
        if res.get(k):
            return str(res[k]).replace("\n", " ")[:200]
    f = res.get("failure") or {}
    err = [l for l in (f.get("stderr") or "").splitlines() if "error" in l.lower() or "fatal" in l.lower()]
    return err[0][:200] if err else None


def main_body(msl):
    m = re.search(r"\b(fragment|kernel)\s+\w+\s+main0\s*\(", msl)
    if not m:
        return None
    start = msl.index("{", m.end())
    return start, len(msl)


def mutants(msl, n, rng, reached=None, stats=None, hits=None, build=True):
    """Up to n mutants as (kind, text, tag). With `reached` (block ids that ran), only sites inside those blocks are
    used; `stats` gets the number of candidate sites, how many are in reached blocks and (with `hits`, every coverage
    id that fired) how many of those are effective. `tag` says whether the mutated site was effective and why."""
    live = liveness(msl, hits) if hits is not None else None
    span = main_body(msl)
    if not span:
        return []
    marks = block_of_offsets(msl)
    starts = [m[0] for m in marks]
    lo, hi = span
    body = msl[lo:hi]
    sites = []
    # SPIRV-Cross writes exact float maths as helper calls (precise: no contraction).
    swap = {"spvFMul(": "spvMutFAdd(", "spvFAdd(": "spvMutFSub(", "spvFSub(": "spvMutFAdd("}
    for m in re.finditer(r"spvF(Mul|Add|Sub)\(", body):
        sites.append(("float_op", m.start(), m.end(), swap[m.group(0)]))
    for m in re.finditer(r"\bfma\(", body):
        sites.append(("float_op", m.start(), m.end(), "spvFmaSwap("))
    for m in re.finditer(r"(?<![\w.\[])(\d+)u\b(?!\])", body):
        line = body[body.rfind("\n", 0, m.start()) + 1: body.find("\n", m.end())]
        if "spvUnsafeArray<" in line:        # register-array initialisers: overwritten before use
            continue
        v = int(m.group(1))
        sites.append(("int_const", m.start(1), m.end(1), str(v + 1)))
    if reached is not None:
        import bisect
        def block(off):
            i = bisect.bisect_right(starts, lo + off) - 1
            return marks[i][1] if i >= 0 else 0
        in_reached = [x for x in sites if block(x[1]) in reached]
        if stats is not None:
            stats["sites"] = len(sites)
            stats["reached_sites"] = len(in_reached)
            if live:
                why = Counter(live(lo + x[1])["why"] for x in in_reached)
                stats["effective_sites"] = sum(v for k, v in why.items()
                                               if k in ("guard_condition", "guard_fired", "ssa_live", "statement"))
                stats["site_why"] = dict(why)
        sites = in_reached
    rng.shuffle(sites)
    picked, kinds = [], Counter()
    for s in sites:                       # at most half of the mutants of each kind
        if kinds[s[0]] < (n + 1) // 2 and len(picked) < n:
            picked.append(s)
            kinds[s[0]] += 1
    out = []
    for kind, a, b, rep in picked:
        if build:
            text = msl[:lo + a] + rep + msl[lo + b:]
            text = text.replace("using namespace metal;\n", "using namespace metal;\n" + MUT_HELPERS, 1)
        else:
            text = None
        tag = dict(live(lo + a)) if live else {}
        ls = msl.rfind("\n", 0, lo + a) + 1
        le = msl.find("\n", lo + a)
        tag.update(line=msl.count("\n", 0, lo + a) + 1, col=lo + a - ls, orig=msl[lo + a:lo + b], rep=rep,
                   src=msl[ls:le].strip()[:300])
        out.append((kind, text, tag))
    return out


# Same exact (no contraction) forms SPIRV-Cross uses for spvFAdd/spvFSub, under names that always exist.
MUT_HELPERS = """
template<typename T> [[clang::optnone]] T spvMutFAdd(T l, T r) { return fma(T(1), l, r); }
template<typename T> [[clang::optnone]] T spvMutFSub(T l, T r) { return fma(T(-1), r, l); }
template<typename T> static inline T spvFmaSwap(T a, T b, T c) { return fma(a, c, b); }
"""

SSA_DEF = re.compile(r"^\s*(?:const\s+)?[A-Za-z_][\w:<>, ]*?\s+(_\d+)\s*=[^=]")
GUARD = re.compile(r"sbx_n?sel\((\d+), ")
VAR = re.compile(r"\b_\d+\b")


def liveness(msl, hits):
    """Per-line effectiveness of main0's body for one run (`hits` = coverage ids that fired).

    A line is effective when its block ran and, for a lane-guarded write, the guard fired (the written value took
    effect). A guard's condition counts as effective whenever its block ran. Statements that are not SSA definitions
    (register/output/memory stores, control flow, returns) are effective in blocks that ran. An SSA definition
    `T _N = ...;` is effective only if a variable it defines is read by an effective line (fixpoint).
    Returns a function: absolute offset -> {"effective": bool, "why": str}."""
    import bisect
    marks = block_of_offsets(msl)
    if not marks:
        return None
    starts = [m[0] for m in marks]
    lines = [msl[a:(starts[k + 1] - 1 if k + 1 < len(starts) else len(msl))] for k, a in enumerate(starts)]
    info = []
    for k, (line, (_, blk)) in enumerate(zip(lines, marks)):
        d = SSA_DEF.match(line)
        g = GUARD.search(line)
        rec = {"reached": blk in hits, "def": d.group(1) if d else None, "guard": None}
        if g:
            close = _close_paren(line, g.end() - len(", ") - len(g.group(1)) - 1)
            rec["guard"] = (int(g.group(1)) in hits, g.end(), close)
            rec["cond_uses"] = set(VAR.findall(line[g.end():close]))
            rec["uses"] = set(VAR.findall(line)) - {d.group(1) if d else None}
        else:
            rec["uses"] = set(VAR.findall(line[d.end() - 1:] if d else line))
        info.append(rec)
    live_vars, work = set(), []

    def add(vs):
        for v in vs:
            if v not in live_vars:
                live_vars.add(v)
                work.append(v)
    defs = {}
    for k, r in enumerate(info):
        if r["def"]:
            defs.setdefault(r["def"], []).append(k)
        if not r["reached"]:
            continue
        if r["guard"]:
            add(r["uses"] if r["guard"][0] else r["cond_uses"])
        elif not r["def"]:
            add(r["uses"])
    while work:
        v = work.pop()
        for k in defs.get(v, []):
            if info[k]["reached"]:
                add(info[k]["uses"])

    def tag(off):
        k = bisect.bisect_right(starts, off) - 1
        if k < 0:
            return {"effective": False, "why": "outside_main"}
        r, col = info[k], off - starts[k]
        if not r["reached"]:
            return {"effective": False, "why": "block_not_run"}
        if r["guard"]:
            fired, c0, c1 = r["guard"]
            if c0 <= col < c1:
                return {"effective": True, "why": "guard_condition"}
            return {"effective": fired, "why": "guard_fired" if fired else "guard_not_fired"}
        if r["def"]:
            ok = r["def"] in live_vars
            return {"effective": ok, "why": "ssa_live" if ok else "ssa_dead"}
        return {"effective": True, "why": "statement"}
    return tag


def _close_paren(line, open_at):
    depth = 0
    for i in range(open_at, len(line)):
        if line[i] == "(":
            depth += 1
        elif line[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(line)


def apply_site(msl, rec):
    """Rebuild a mutant from its record (line, col, orig, rep): same text mutants() produces."""
    starts = [0] + [i + 1 for i, c in enumerate(msl) if c == "\n"]
    off = starts[rec["line"] - 1] + rec["col"]
    if msl[off:off + len(rec["orig"])] != rec["orig"]:
        raise ValueError(f"site text mismatch at line {rec['line']} col {rec['col']}")
    text = msl[:off] + rec["rep"] + msl[off + len(rec["orig"]):]
    return text.replace("using namespace metal;\n", "using namespace metal;\n" + MUT_HELPERS, 1)


def pin_sets(prep, results, holdout_n=8):
    """Write <shader>/mutants_pinned.json (the mutants of a run_set results file) and <shader>/mutants_holdout.json
    (a fresh sample from all of main0's sites with another seed, excluding pinned sites; never given to the
    resolver), so later runs re-test exactly the same mutants."""
    n_pin = n_hold = 0
    for l in open(results):
        r = json.loads(l)
        d = os.path.join(prep, r["name"])
        if not os.path.isdir(d):
            continue
        keys = ("kind", "line", "col", "orig", "rep")
        pinned = [{k: m[k] for k in keys} for m in r.get("mutants", []) if "line" in m]
        json.dump(pinned, open(os.path.join(d, "mutants_pinned.json"), "w"), indent=0)
        msl = open(os.path.join(d, "ref.metal")).read()
        taken = {(m["line"], m["col"]) for m in pinned}
        cand = [dict(t, kind=k) for k, _, t in mutants(msl, 10 ** 6, random.Random(f"{r['name']}:holdout"), build=False)]
        hold = []
        for t in cand:
            if (t["line"], t["col"]) not in taken and len(hold) < holdout_n:
                hold.append({k: t[k] for k in keys})
                taken.add((t["line"], t["col"]))
        json.dump(hold, open(os.path.join(d, "mutants_holdout.json"), "w"), indent=0)
        n_pin += len(pinned)
        n_hold += len(hold)
    return n_pin, n_hold


def pinned_work(d, msl, hits, rec):
    """(kind, text, tag) for the pinned and held-out mutants; text None when the site's block was not reached."""
    import bisect
    marks = block_of_offsets(msl)
    starts_l = [0] + [i + 1 for i, c in enumerate(msl) if c == "\n"]
    mstarts = [m[0] for m in marks]
    live = liveness(msl, hits)
    out = []
    for sample in ("pinned", "holdout"):
        p = os.path.join(d, f"mutants_{sample}.json")
        for m in (json.load(open(p)) if os.path.exists(p) else []):
            off = starts_l[m["line"] - 1] + m["col"]
            i = bisect.bisect_right(mstarts, off) - 1
            blk = marks[i][1] if i >= 0 else 0
            tag = dict(m, sample=sample, block=blk)
            if live:
                tag.update(live(off))
            if blk not in hits:
                out.append((m["kind"], None, tag))
                continue
            out.append((m["kind"], apply_site(msl, m), tag))
    rec["pinned"] = True
    return out


def written(r):
    return any(o.get("written_by_a") for o in r.get("outputs", []))


def one(d, n_mut, seed, seeds=(1,)):
    try:
        return _one(d, n_mut, seed, seeds)
    except Exception as e:                # one shader must never stop the batch
        return {"name": os.path.basename(d), "status": "exception",
                "failure": runlog.exception(e, LOG, f"run_set shader {os.path.basename(d)}")}


def _one(d, n_mut, seed, seeds=(1,)):
    name = os.path.basename(d)
    LOG.debug("shader %s: start", name)
    ref = os.path.join(d, "ref.metal")
    rec = {"name": name}
    inputs = os.path.join(d, "inputs.json") if USE_INPUTS and os.path.exists(os.path.join(d, "inputs.json")) else None
    rec["inputs"] = "inputs.json" if inputs else f"seeds {list(seeds)}"
    det = harness(d, ref, ref, coverage=True, seeds=seeds, inputs=inputs)
    rec["status"] = det.get("status")
    rec["stage"] = det.get("stage")
    if det.get("status") != "ok":
        rec["detail"] = {k: det[k] for k in det if k.endswith("error") or k.startswith("pipeline_error")
                         or k.startswith("gpu_error") or k in ("unsupported", "reason", "max_threads_a")}
        rec["phase"] = det.get("phase")
        rec["reason"] = first_error(det)
        rec["failure"] = det.get("failure")
        return rec
    if inputs and det.get("status") == "ok":
        # determinism per input variant, three times over: drop the variants where the reference races (shared
        # writes, atomic append order) or trips the loop guard and record how many were dropped
        variants = json.load(open(inputs))
        bad = set()
        for _ in range(3):
            full = harness(d, ref, ref, coverage=True, inputs=inputs, extra=["--all", "1"])
            if full.get("status") != "ok":
                break
            for v in full.get("variants", []):
                if not v.get("equal") or v.get("loop_guard") or v.get("loop_guard_b"):
                    bad.add(v["i"])
        good = [v for i, v in enumerate(variants) if i not in bad]
        rec["variants_dropped"] = len(variants) - len(good)
        if bad and good:
            fd, kept = tempfile.mkstemp(suffix=".json", prefix="det_inputs_")
            os.close(fd)
            json.dump(good, open(kept, "w"))
            inputs = kept
            det = harness(d, ref, ref, coverage=True, seeds=seeds, inputs=inputs)
            LOG.info("shader %s: %d input variants dropped (race or loop guard in the reference), %d kept", name,
                     len(bad), len(good))
    rec["deterministic"] = bool(det.get("equal"))
    rec["cov_blocks"] = det.get("cov_blocks", 0)
    rec["cov_guards"] = det.get("cov_guards", 0)
    rec["cov_hit"] = sum(1 for i in det.get("cov_hits", []) if i <= rec["cov_blocks"])
    rec["guard_hit"] = sum(1 for i in det.get("cov_hits", []) if i > rec["cov_blocks"])
    rec["hits"] = sorted(det.get("cov_hits", []))      # for target (b): which blocks (incl. sink blocks) ran
    rec["loop_guard"] = bool(det.get("loop_guard_a") or det.get("loop_guard_b"))
    rec["exercised"] = written(det)
    rec["gpu_ms"] = det.get("gpu_ms_a")
    rec["nonfinite"] = sum(o.get("nonfinite_a", 0) for o in det.get("outputs", []))
    if not rec["deterministic"]:
        rec["nondet_outputs"] = [o["output"] for o in det.get("outputs", []) if "diff_words" in o]
        return rec
    rng = random.Random(f"{name}:{seed}")
    msl = open(ref).read()
    res = []
    stats = {}
    hits = set(det.get("cov_hits", []))
    reached = hits if det.get("cov_blocks") else None
    if PINNED and os.path.exists(os.path.join(d, "mutants_pinned.json")):
        work = pinned_work(d, msl, hits, rec)
    else:
        work = mutants(msl, n_mut, rng, reached, stats, hits)
    with tempfile.TemporaryDirectory() as tmp:
        for i, (kind, text, tag) in enumerate(work):
            if text is None:                       # pinned mutant in a block no input reaches: target (b)
                res.append(dict(tag, kind=kind, verdict="not_reached"))
                continue
            path = os.path.join(tmp, f"mut{i}.metal")
            open(path, "w").write(text)
            r = harness(d, ref, path, seeds=seeds, inputs=inputs)   # one process; stops at the first difference
            st = r.get("status")
            killed_by = r.get("first_diff_variant", r.get("first_diff_seed"))
            if st == "ok" and not r.get("equal"):
                verdict = "killed"
            elif st == "ok" and r.get("loop_guard_a"):
                verdict = "inconclusive"           # the reference's loop was cut short: outputs aren't comparable
            elif st == "ok" and r.get("loop_guard_b"):
                verdict = "killed"                 # the mutant loops where the reference doesn't: different behaviour
            elif st == "ok":
                verdict = "survived"
            elif st in ("compile_b", "pipeline_b"):
                verdict = "invalid"
            elif st in ("compile_timeout_b", "pipeline_timeout_b", "process_timeout") or st.endswith("_a"):
                verdict = "inconclusive"           # compiler stall, or the reference itself failed
            else:
                verdict = "killed"                 # the mutant faulted or hung on the GPU: different behaviour
            m = {"kind": kind, "verdict": verdict, "seed": killed_by, **tag}
            if st != "ok":
                m.update(status=st, phase=r.get("phase"), reason=first_error(r))
            res.append(m)
    rec["mutants"] = res
    rec.update(stats)
    return rec


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    prep, out = argv[0], argv[1]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 8
    n_mut = int(argv[argv.index("--mutants") + 1]) if "--mutants" in argv else 4
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    seeds = tuple(range(1, 1 + (int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 1)))
    global LOG, USE_INPUTS, PINNED
    USE_INPUTS = "--no-inputs" not in argv
    PINNED = "--pinned" in argv
    if "--pin" in argv:                 # run_set.py <prep> <results.jsonl> --pin : write the pinned/holdout lists
        n_pin, n_hold = pin_sets(prep, out)
        print(f"pinned {n_pin} mutants, held out {n_hold}")
        return 0
    LOG = runlog.setup(out, "run_set", verbose="--verbose" in argv)
    LOG.info("run_set: prep %s, jobs %d, mutants %d, seeds %s, harness %s", prep, jobs, n_mut, seeds, HARNESS)
    dirs = sorted(os.path.join(prep, n) for n in os.listdir(prep)
                  if os.path.isfile(os.path.join(prep, n, "meta.json")))[:limit]
    recs = []
    with ThreadPoolExecutor(jobs) as ex, open(out, "w") as f:
        for i, r in enumerate(ex.map(lambda d: one(d, n_mut, 1, seeds), dirs), 1):
            recs.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            LOG.debug("shader %s: %s", r["name"], r["status"])
            if i % 25 == 0:
                LOG.info(f"  {i}/{len(dirs)}")
    summarise(recs)
    runlog.summarise_failures(recs, lambda r: None if r["status"] == "ok" else
                              f"{r['status']} in {r.get('phase')}: {r.get('reason') or ''}"[:160], LOG)
    runlog.summarise_failures([dict(m, name=r["name"]) for r in recs for m in r.get("mutants", [])],
                              lambda m: None if "status" not in m else
                              f"mutant {m['verdict']}: {m['status']} in {m.get('phase')}: {m.get('reason') or ''}"[:160], LOG)
    return 0


def summarise(recs):
    LOG.info(f"{len(recs)} shaders")
    by_stage = {}
    for r in recs:
        by_stage.setdefault(r.get("stage") or "?", []).append(r)
    LOG.info("status: %s", dict(Counter(r["status"] for r in recs)))
    ok = [r for r in recs if r["status"] == "ok"]
    for st, rs in sorted(by_stage.items()):
        okr = [r for r in rs if r["status"] == "ok"]
        if not okr:
            continue
        det = sum(r["deterministic"] for r in okr)
        ex = sum(r["exercised"] for r in okr)
        LOG.info(f"  {st}: ran {len(okr)}/{len(rs)}, deterministic {det}, wrote output {ex}, "
              f"non-finite outputs {sum(1 for r in okr if r['nonfinite'])}")
    cov = [r for r in ok if r.get("cov_blocks")]
    if cov:
        tb, hb = sum(r["cov_blocks"] for r in cov), sum(r["cov_hit"] for r in cov)
        ts, rs_ = sum(r.get("sites", 0) for r in cov), sum(r.get("reached_sites", 0) for r in cov)
        LOG.info(f"coverage: {hb}/{tb} blocks ran ({100 * hb / tb:.1f}%), {rs_}/{ts} mutation sites in blocks that ran"
              f" ({100 * rs_ / max(ts, 1):.1f}%), loop guard tripped in {sum(r['loop_guard'] for r in ok)} shaders")
    tg, hg = sum(r.get("cov_guards", 0) for r in cov), sum(r.get("guard_hit", 0) for r in cov)
    es = sum(r.get("effective_sites", 0) for r in cov)
    if tg:
        LOG.info(f"lane guards: {hg}/{tg} fired ({100 * hg / tg:.1f}%); effective sites {es}/{rs_} of the sites in blocks"
                 f" that ran ({100 * es / max(rs_, 1):.1f}%)")
        why = Counter()
        for r in cov:
            why.update(r.get("site_why", {}))
        LOG.info("  sites in blocks that ran, by reason: %s", dict(why.most_common()))
    muts = [(r, m) for r in ok for m in r.get("mutants", [])]
    for label, sel in (("effective", True), ("not effective", False)):
        g = Counter(m["verdict"] for _, m in muts if m.get("effective") is sel)
        n = g["killed"] + g["survived"]
        if n:
            LOG.info(f"mutants on {label} sites: {dict(g)} kill rate {100 * g['killed'] / n:.1f}%")
    for label, sel in (("wrote output", True), ("wrote nothing", False)):
        g = Counter(m["verdict"] for r, m in muts if bool(r.get("exercised")) is sel and m.get("effective"))
        n = g["killed"] + g["survived"]
        if n:
            LOG.info(f"  effective, shaders that {label}: {g['killed']}/{n} killed ({100 * g['killed'] / n:.1f}%)")
    w = Counter(m.get("why") for _, m in muts if m["verdict"] in ("killed", "survived"))
    for k in sorted(w):
        kk = sum(1 for _, m in muts if m.get("why") == k and m["verdict"] == "killed")
        LOG.info(f"  {k}: {kk}/{w[k]} killed")
    v = Counter(m["verdict"] for r in ok for m in r.get("mutants", []))
    k = Counter((m["kind"], m["verdict"]) for r in ok for m in r.get("mutants", []))
    valid = v["killed"] + v["survived"]
    LOG.info("mutants (only in blocks that ran): %s %s", dict(v), f"kill rate {100 * v['killed'] / valid:.1f}%" if valid else "")
    for kind in ("float_op", "int_const"):
        kv = k[(kind, "killed")] + k[(kind, "survived")]
        if kv:
            LOG.info(f"  {kind}: {k[(kind, 'killed')]}/{kv} killed")



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
