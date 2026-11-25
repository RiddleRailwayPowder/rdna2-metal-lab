"""Read-only helpers over an rpf9_carve.py --index-only index: inflate one stream (plus its 64 KiB
continuations) straight from the archive, find PRD0 resources and their name strings.

python3 -I rpf_peek.py ARCHIVE INDEX.json --census           # PRD0 resources, names with .#xx suffixes
python3 -I rpf_peek.py ARCHIVE INDEX.json --dump OFFSET [N]  # hexdump first N bytes of the joined stream
python3 -I rpf_peek.py ARCHIVE INDEX.json --grep REGEX [--max K]  # streams whose joined text matches
"""
import argparse, json, re, struct, sys, zlib, collections


def load(arc, files, i, join=True, cap=1 << 26):
    out = bytearray()
    while i < len(files):
        f = files[i]
        arc.seek(f["off"])
        out += zlib.decompress(arc.read(f["stored"]))
        if not join or f["size"] != 65536 or len(out) >= cap:
            break
        i += 1
    return bytes(out)


DXGI = {2: "R32G32B32A32_FLOAT", 10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM", 24: "R10G10B10A2_UNORM",
        26: "R11G11B10_FLOAT", 28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB", 34: "R16G16_FLOAT", 41: "R32_FLOAT",
        49: "R8G8_UNORM", 54: "R16_FLOAT", 56: "R16_UNORM", 61: "R8_UNORM", 65: "A8_UNORM", 71: "BC1", 72: "BC1_SRGB",
        74: "BC2", 77: "BC3", 78: "BC3_SRGB", 80: "BC4", 81: "BC4_SNORM", 83: "BC5", 84: "BC5_SNORM", 87: "B8G8R8A8",
        91: "B8G8R8A8_SRGB", 95: "BC6H_UF16", 96: "BC6H_SF16", 98: "BC7", 99: "BC7_SRGB"}


def txd(arc, files):
    """Texture dictionaries: pgDictionary<grcTexture> as documented by Asset-Info notes section 3."""
    fmts = collections.Counter(); total = 0; dicts = 0
    for i, f in enumerate(files):
        h = bytes.fromhex(f["head"])
        if len(h) < 16 or h[11] >> 4 != 5 or f["size"] > 1 << 20:
            continue
        v = load(arc, files, i, join=False)
        def vp(x): return x & 0x0FFFFFFF if (x >> 28) == 5 else None
        def u64(o): return struct.unpack_from("<Q", v, o)[0]
        try:
            ha, hn, ta, tn = vp(u64(0x20)), struct.unpack_from("<H", v, 0x28)[0], vp(u64(0x30)), struct.unpack_from("<H", v, 0x38)[0]
            if ha is None or ta is None or hn != tn or not 0 < tn < 5000:
                continue
            dicts += 1; rows = []
            for k in range(tn):
                t = vp(u64(ta + 8 * k)); np_ = vp(u64(t + 0x30))
                e = v.index(b"\0", np_ + 8); name = v[np_ + 8:e].decode("latin-1") or v[np_:v.index(b"\0", np_)].decode("latin-1")
                w, hh, dep = struct.unpack_from("<HHH", v, t + 0x38)
                fmt = v[t + 0x3F]; mips = v[t + 0x42]
                fmts[DXGI.get(fmt, fmt)] += 1; total += 1
                rows.append(f"{name} {w}x{hh}x{dep} {DXGI.get(fmt, fmt)} mips {mips}")
            print(f"{f['off']:#x} dict of {tn}: " + "; ".join(rows[:8]) + (" ..." if tn > 8 else ""))
        except (struct.error, ValueError, TypeError):
            continue
    print("dictionaries", dicts, "textures", total, "formats", fmts.most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("index")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--dump", type=lambda s: int(s, 0)); ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--txd", action="store_true", help="list texture dictionaries (system pages with a 0x5... pointer at +8)")
    ap.add_argument("--grep"); ap.add_argument("--max", type=int, default=10)
    a = ap.parse_args()
    files = json.load(open(a.index))["files"]
    arc = open(a.archive, "rb")
    if a.census:
        names = collections.Counter(); prd = 0; ex = []
        for i, f in enumerate(files):
            h = bytes.fromhex(f["head"])
            if h[0x0:0x4] == b"\0\0\0\0" and len(h) >= 16:
                pass
            arc.seek(f["off"])
            d = zlib.decompressobj().decompress(arc.read(min(f["stored"], 1 << 16)), 4096)
            if d[0x10:0x14] == b"0DRP":
                prd += 1
                full = load(arc, files, i, join=False)
                for m in re.finditer(rb"[A-Za-z0-9_\-/:]{2,80}\.#[a-z]{2}", full):
                    names[m.group().decode()[-3:]] += 1
                    if len(ex) < 25: ex.append((hex(f["off"]), m.group().decode()))
        print("PRD0 resources", prd, "name suffixes", dict(names)); print("examples", ex)
        return
    if a.txd:
        txd(arc, files)
        return
    if a.dump is not None:
        i = next(k for k, f in enumerate(files) if f["off"] == a.dump)
        d = load(arc, files, i)
        print("joined size", len(d))
        for o in range(0, min(a.n, len(d)), 16):
            r = d[o:o + 16]
            print("%08x  %-48s %s" % (o, r.hex(" "), "".join(chr(c) if 32 <= c < 127 else "." for c in r)))
        return
    if a.grep:
        rx = re.compile(a.grep.encode()); shown = 0; i = 0
        while i < len(files) and shown < a.max:
            f = files[i]
            prev = files[i - 1]["size"] if i else 0
            if prev == 65536:  # continuation, skip
                i += 1; continue
            d = load(arc, files, i)
            m = rx.search(d)
            if m:
                print(f"{f['off']:#x} joined {len(d)} match@{m.start():#x}: {d[max(0, m.start()-80):m.end()+160]!r}")
                shown += 1
            i += 1


if __name__ == "__main__":
    main()
