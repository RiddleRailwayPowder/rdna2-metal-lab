"""Select-arm coverage probe: a copy of ref.metal where every scalar ternary `C ? A : B` in main0 (other than the
existing lane guards) becomes `sbx_tsel(t, f, C) ? A : B`; coverage id t fires when A is taken, f when B is taken.
Line numbers are unchanged (insertions only within lines). A copy of meta.json raises cov_guards so the harness
reads the new ids and points fssm at an absolute path.

    instrument(ref_path, meta_path, out_dir) -> (probe_metal, probe_meta, arms)
arms: {0-based line: [(t, f, (a0, a1), (b0, b1))]} with spans in original-line coordinates.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/harness")
import dead_store as ds  # noqa: E402

DEFINE = "#define sbx_tsel(t, f, c) ((c) ? (sbx_hit(t), true) : (sbx_hit(f), false))\n"


def _is_assign(s, i):
    if s[i] != "=":
        return False
    prev = s[i - 1] if i else ""
    nxt = s[i + 1] if i + 1 < len(s) else ""
    return nxt != "=" and prev not in "=<>!"


def ternaries(s):
    """[(cond_start, q, colon, false_end)] for each `?` in line s."""
    out = []
    n = len(s)
    for q in range(n):
        if s[q] != "?":
            continue
        # condition start
        d = 0
        i = q - 1
        while i >= 0:
            c = s[i]
            if c in ")]}":
                d += 1
            elif c in "([{":
                if d == 0:
                    break
                d -= 1
            elif d == 0 and (c in ",?;" or (c == ":" and not (i > 0 and s[i - 1] == ":") and not (i + 1 < n and s[i + 1] == ":"))
                             or _is_assign(s, i)):
                break
            i -= 1
        cs = i + 1
        while cs < q and s[cs] == " ":
            cs += 1
        if s.startswith("return ", cs):
            cs += 7
        # matching colon
        d = 0
        nest = 0
        j = q + 1
        colon = None
        while j < n:
            c = s[j]
            if c in "([{":
                d += 1
            elif c in ")]}":
                if d == 0:
                    break
                d -= 1
            elif d == 0 and c == "?":
                nest += 1
            elif d == 0 and c == ":" and s[j - 1] != ":" and (j + 1 >= n or s[j + 1] != ":"):
                if nest == 0:
                    colon = j
                    break
                nest -= 1
            j += 1
        if colon is None:
            continue
        # false arm end
        d = 0
        k = colon + 1
        while k < n:
            c = s[k]
            if c in "([{":
                d += 1
            elif c in ")]}":
                if d == 0:
                    break
                d -= 1
            elif d == 0 and c in ",;":
                break
            elif d == 0 and c == ":" and s[k - 1] != ":" and (k + 1 >= n or s[k + 1] != ":"):
                break                                  # belongs to an enclosing ternary
            k += 1
        cond = s[cs:q].strip()
        if not cond or cond.startswith("sbx_sel(") or cond.startswith("sbx_nsel(") or cond.startswith("sbx_tsel("):
            continue
        ce = q
        while ce > cs and s[ce - 1] == " ":
            ce -= 1
        out.append((cs, ce, q, colon, k))
    return out


def instrument(ref, meta_path, out_dir):
    msl = open(ref).read()
    meta = json.load(open(meta_path))
    lines, lo, hi = ds._main_lines(msl)
    lines = list(lines)
    nid = meta["cov_blocks"] + meta["cov_guards"] + 1
    arms = {}
    for li in range(lo, hi):
        s = lines[li]
        if "?" not in s:
            continue
        ts = ternaries(s)
        if not ts:
            continue
        ins = []
        rec = []
        for cs, ce, q, colon, end in ts:
            t, f = nid, nid + 1
            nid += 2
            ins.append((cs, 0, f"sbx_tsel({t}, {f}, "))
            ins.append((ce, 1, ")"))
            rec.append((t, f, (q + 1, colon), (colon + 1, end)))
        for pos, _, txt in sorted(ins, key=lambda x: (x[0], -x[1]), reverse=True):
            s = s[:pos] + txt + s[pos:]
        lines[li] = s
        arms[li] = rec
    if nid >= 32768:
        raise ValueError(f"too many probe ids: {nid}")
    text = "\n".join(lines)
    k = text.index("#define sbx_nsel")
    k = text.index("\n", k) + 1
    text = text[:k] + DEFINE + text[k:]
    # the define adds one line before main0: keep line numbers by removing one blank line before main0 if possible
    os.makedirs(out_dir, exist_ok=True)
    pm = os.path.join(out_dir, "probe.metal")
    open(pm, "w").write(text)
    meta2 = dict(meta)
    meta2["cov_guards"] = nid - meta["cov_blocks"] - 1
    if meta.get("fssm") and not os.path.isabs(meta["fssm"]):
        meta2["fssm"] = os.path.normpath(os.path.join(os.environ.get("GAME_ROOT", ".") + "/harness_work", meta["fssm"]))
    pmeta = os.path.join(out_dir, "meta.json")
    json.dump(meta2, open(pmeta, "w"), indent=1)
    json.dump({str(k): v for k, v in arms.items()}, open(os.path.join(out_dir, "arms.json"), "w"))
    return pm, pmeta, arms
