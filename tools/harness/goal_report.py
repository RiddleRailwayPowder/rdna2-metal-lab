"""Report the two-part goal from a pinned run_set run (run_set.py --pinned).

(a) mutants: for each sample (pinned = the mutants the resolver works on, holdout = never hunted), the kill rate
    killed / (killed + survived - proved equivalent) over mutants in blocks our inputs reach. Equivalence proofs
    come from resolve_survivors.py output files (they don't depend on inputs); kills count only when the pinned
    run itself kills the mutant, so every kill is reproduced on the final inputs with one harness build.
(b) sink coverage: blocks holding an output, store, atomic or texture write (fuzz_inputs.block_targets on
    cmplog.metal) that some input reaches, out of all sink blocks; "proved infeasible" is a separate bucket (empty
    until a proof tool fills it).
Excluded shaders (cannot run on this Mac's Metal) are listed by name with the reason.

Usage: python3 -I goal_report.py <prep dir> <pinned results.jsonl> [--equiv FILE.resolved.jsonl ...] [--out REPORT.md]
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuzz_inputs  # noqa: E402


def key(m):
    return (m.get("name"), m["line"], m["col"], m["orig"], m["rep"])


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    prep, results = argv[0], argv[1]
    equiv_files = []
    if "--equiv" in argv:
        i = argv.index("--equiv") + 1
        while i < len(argv) and not argv[i].startswith("--"):
            equiv_files.append(argv[i])
            i += 1
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else None
    proofs = {}
    for f in equiv_files:
        for l in open(f):
            r = json.loads(l)
            if r.get("resolution") == "equivalent":
                proofs[key(r)] = r.get("proof", "")
    recs = [json.loads(l) for l in open(results)]
    lines = []
    excluded = [(r["name"], r["status"], r.get("reason") or r.get("phase")) for r in recs if r.get("status") != "ok"]
    per = {"pinned": Counter(), "holdout": Counter()}
    conflicts = []                     # proved equivalent AND killed: an unsound proof or a false kill
    for r in recs:
        if r.get("status") != "ok":
            continue
        for m in r.get("mutants", []):
            s = m.get("sample", "pinned")
            v = m["verdict"]
            k = key(dict(m, name=r["name"]))
            if v == "killed" and k in proofs:
                conflicts.append((r["name"], m["line"], m["orig"], m["rep"], proofs[k]))
            if v == "survived" and k in proofs:
                v = "equivalent"
            per[s][v] += 1
    lines.append("# Goal report\n")
    lines.append(f"Results: `{os.path.basename(results)}`; equivalence proofs from: "
                 f"{', '.join(os.path.basename(f) for f in equiv_files) or 'none'}\n")
    lines.append("## (a) Mutants in reached code\n")
    lines.append("| Sample | Killed | Survived | Proved equivalent | Not reached | Invalid | Inconclusive | Kill rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s, c in per.items():
        denom = c["killed"] + c["survived"]
        rate = f"{100 * c['killed'] / denom:.1f}%" if denom else "n/a"
        lines.append(f"| {s} | {c['killed']} | {c['survived']} | {c['equivalent']} | {c['not_reached']} | "
                     f"{c['invalid']} | {c['inconclusive']} | {rate} |")
    lines.append("\nKill rate = killed / (killed + survived); proved-equivalent mutants are out of the denominator.\n")
    lines.append(f"**Proof/kill conflicts: {len(conflicts)}** (must be 0; each is an unsound proof or a false kill)\n")
    for c in conflicts[:20]:
        lines.append(f"- `{c[0]}` line {c[1]}: `{c[2]}` -> `{c[3]}` killed, but proof: {c[4]}")
    if conflicts:
        lines.append("")
    # (b)
    tot = cov = 0
    worst = []
    for r in recs:
        if r.get("status") != "ok" or "hits" not in r:
            continue
        d = os.path.join(prep, r["name"])
        cl = os.path.join(d, "cmplog.metal")
        if not os.path.exists(cl):
            continue
        nb = json.load(open(os.path.join(d, "meta.json"))).get("cov_blocks") or 0
        t = fuzz_inputs.block_targets(open(cl).read())
        sinks = [b for b, v in t.items() if v["sink"] and b <= nb]
        hit = set(r["hits"])
        c = sum(1 for b in sinks if b in hit)
        tot += len(sinks)
        cov += c
        if sinks and c < len(sinks):
            worst.append((len(sinks) - c, r["name"], c, len(sinks)))
    lines.append("## (b) Sink blocks reached\n")
    lines.append(f"{cov} of {tot} sink blocks reached ({100 * cov / max(tot, 1):.1f}%); proved infeasible: 0.\n")
    worst.sort(reverse=True)
    if worst:
        lines.append("Most unreached sink blocks:\n")
        for n, name, c, t in worst[:10]:
            lines.append(f"- `{name}`: {c} of {t}")
        lines.append("")
    lines.append("## Excluded shaders\n")
    for name, st, why in excluded:
        lines.append(f"- `{name}`: {st} ({why})")
    text = "\n".join(lines) + "\n"
    print(text)
    if out_path:
        open(out_path, "w").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
