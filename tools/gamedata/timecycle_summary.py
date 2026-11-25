"""Read-only: summarise timecycle_keyframe_data / timecycle_modifier_data XML in an RPF9 archive.

python3 -I timecycle_summary.py ARCHIVE CARVE_INDEX.json [--show VAR ...] [--json OUT]
Joins 64 KiB continuation streams, parses each keyframe document, prints cycle names, regions, variable counts
and the per-hour values of the variables named with --show (first cycle, first region).
"""
import os  # noqa: E401
import argparse, json, re, sys, zlib, collections, xml.etree.ElementTree as ET
sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/gamedata")
from rpf_peek import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("index"); ap.add_argument("--show", nargs="*", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    files = json.load(open(a.index))["files"]; arc = open(a.archive, "rb")
    out = {"keyframe_docs": [], "modifier_docs": 0, "modifiers": 0, "modifier_vars": collections.Counter()}
    i = 0
    while i < len(files):
        if i and files[i - 1]["size"] == 65536:
            i += 1; continue
        d = load(arc, files, i)
        i += 1
        head = d[:200]
        if b"timecycle_keyframe_data" in head:
            root = ET.fromstring(d.decode("utf-8-sig", "replace").lstrip("﻿"))
            hl = {c.tag: c.text.strip() for c in root.find("High_Level_Vars")} if root.find("High_Level_Vars") is not None else {}
            for cyc in root.iter("cycle"):
                regions = cyc.findall("region")
                r0 = regions[0] if regions else cyc
                vars_ = {c.tag: [float(x) for x in (c.text or "").split()] for c in r0}
                doc = {"offset": hex(files[i - 1]["off"]), "cycle": cyc.get("name"), "regions": [r.get("name") for r in regions],
                       "vars": len(vars_), "samples_per_var": collections.Counter(len(v) for v in vars_.values()).most_common(2),
                       "high_level": hl}
                doc["show"] = {k: vars_[k] for k in a.show if k in vars_}
                doc["var_names_sample"] = list(vars_)[:12]
                out["keyframe_docs"].append(doc)
        elif b"timecycle_modifier_data" in head:
            root = ET.fromstring(d.decode("utf-8-sig", "replace").lstrip("﻿"))
            out["modifier_docs"] += 1
            for m in root.iter("modifier"):
                out["modifiers"] += 1
                for c in m:
                    out["modifier_vars"][c.tag] += 1
    out["modifier_vars_distinct"] = len(out["modifier_vars"])
    out["modifier_vars"] = out["modifier_vars"].most_common(25)
    s = json.dumps(out, indent=1)
    if a.json: open(a.json, "w").write(s)
    print(s[:20000])


main()
