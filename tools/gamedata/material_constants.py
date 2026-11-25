"""Read-only: find material shader records ("atem" at +4) in resource system pages of an RPF9 archive and print
their constant-buffer values with parameter names (joaat dictionary). Layout: Asset-Info notes formats/07-models.md 7.6.

python3 -I material_constants.py ARCHIVE CARVE_INDEX.json --dict WORDS.txt [--max-resources N] [--show K]
"""
import os  # noqa: E401
import argparse, json, struct, sys, zlib, collections
sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/upstream/palmetto/tools")
from hashdict import joaat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("index"); ap.add_argument("--dict")
    ap.add_argument("--max-resources", type=int, default=400); ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()
    names = {}
    if a.dict:
        for w in open(a.dict, encoding="latin-1").read().split():
            names.setdefault(joaat(w), w)
    files = json.load(open(a.index))["files"]; arc = open(a.archive, "rb")
    seen = 0; shown = 0; mats = 0; consts = 0; named = 0; shaders = collections.Counter()
    for f in files:
        h = bytes.fromhex(f["head"])
        if f["off"] & 0xFFF != 0x10 or len(h) < 12 or h[11] >> 4 != 5:
            continue
        seen += 1
        if seen > a.max_resources: break
        arc.seek(f["off"]); v = zlib.decompress(arc.read(f["stored"]))
        def vp(o):
            x = struct.unpack_from("<Q", v, o)[0]
            return x & 0x0FFFFFFF if (x >> 28) == 5 and (x >> 32) == 0 and (x & 0x0FFFFFFF) < len(v) else None
        p = v.find(b"atem")
        while p >= 4:
            s = p - 4
            try:
                cb, pt = vp(s + 8), vp(s + 0x20)
                if cb is None or pt is None: raise ValueError
                cnt = v[pt + 4]; ntex = v[pt + 1]
                rows = [struct.unpack_from("<II", v, pt + 8 + 8 * i) for i in range(cnt)]
                bufs = {}
                k = 0
                while vp(cb + 8 * k) is not None and k < 4:
                    bufs[k] = vp(cb + 8 * k); k += 1
                bufidx = sorted({d & 0xFF for _, d in rows[ntex:]})
                mats += 1; shaders[struct.unpack_from("<I", v, s)[0]] += 1
                out = []
                for nh, d in rows[ntex:]:
                    comps, off, buf = d >> 22, (d >> 8) & 0x3FFF, d & 0xFF
                    bi = bufidx.index(buf) if buf in bufidx else None
                    if bi is None or bi not in bufs: continue
                    vals = struct.unpack_from("<%df" % comps, v, bufs[bi] + off)
                    consts += 1; named += nh in names
                    out.append(f"{names.get(nh, '%08x' % nh)}[b{buf}+{off}]={[round(x, 4) for x in vals]}")
                if shown < a.show:
                    sh = struct.unpack_from("<I", v, s)[0]
                    print(f"{f['off']:#x}+{s:#x} shader {names.get(sh, '%08x' % sh)} textures {ntex} constants {len(out)}: " + "; ".join(out[:14]))
                    shown += 1
            except (ValueError, struct.error, IndexError):
                pass
            p = v.find(b"atem", p + 4)
    print(f"resources scanned {min(seen, a.max_resources)}, materials {mats}, constants {consts}, named {named}")
    print("top shaders", [(names.get(h, '%08x' % h), c) for h, c in shaders.most_common(10)])


main()
