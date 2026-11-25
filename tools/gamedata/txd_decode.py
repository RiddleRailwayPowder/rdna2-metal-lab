"""Read-only: decode named textures from a texture dictionary in textures.rpf to PNG (mip 0, slice 0).

python3 -I txd_decode.py ARCHIVE CARVE_INDEX.json DICT_OFFSET NAME [NAME...] --out DIR [--raw]
DICT_OFFSET is the system-page stream offset printed by rpf_peek.py --txd. Physical (graphics) data is the run
of streams after the system page up to the next resource start. Layout: 6stars TextureLayout / Asset-Info notes
section 4 (4 KiB tiles, per-element-size bit orders; mode 0 = 64 KiB macro tiles). BC decode: palmetto ttpf_mips.
--raw also writes the linear element bytes (.bin) for use as a harness texture.
"""
import os  # noqa: E401
import argparse, json, os, struct, sys, zlib
sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/upstream/palmetto/tools")
import ttpf_mips as T  # noqa: E402

EB = {2: 16, 10: 8, 11: 8, 24: 4, 26: 4, 28: 4, 29: 4, 34: 4, 41: 4, 49: 2, 54: 2, 56: 2, 61: 1, 65: 1, 87: 4, 91: 4,
      71: 8, 72: 8, 80: 8, 81: 8, 74: 16, 77: 16, 78: 16, 83: 16, 84: 16, 95: 16, 96: 16, 98: 16, 99: 16}
BC = {71, 72, 74, 77, 78, 80, 81, 83, 84, 95, 96, 98, 99}


class Layout:  # port of 6stars Sources/RPFKit/TextureDB.swift TextureLayout (non-macro and macro paths)
    def __init__(self, eb, mode):
        self.eb, self.macro = eb, mode == 0
        block = {16: "xyxy", 8: "xyyxx", 4: "xxyyyx", 2: "xxxyyyx"}.get(eb, "xxxxyyyy")
        bits = "yyxxyxyx" if (eb == 16 and mode != 0) else ("xxyyxyyxyx" if (eb == 4 and mode == 0) else block + "yxyx")
        self.pat = [c == "x" for c in bits]
        self.tw = 1 << sum(self.pat); self.th = 1 << (len(self.pat) - sum(self.pat))

    def in_tile(self, x, y):
        idx = xb = yb = 0
        for bit, isx in enumerate(self.pat):
            if isx: idx |= ((x >> xb) & 1) << bit; xb += 1
            else: idx |= ((y >> yb) & 1) << bit; yb += 1
        return idx

    def slot_bytes(self, ew, eh):
        if ew > self.tw or eh > self.th:
            return ((ew + self.tw - 1) // self.tw) * ((eh + self.th - 1) // self.th) * 4096
        top = xb = yb = 0
        for bit, isx in enumerate(self.pat):
            if isx:
                if ew > 1 << xb: top = bit + 1
                xb += 1
            else:
                if eh > 1 << yb: top = bit + 1
                yb += 1
        return max(256, (1 << top) * self.eb)

    def linear(self, src, base, ew, eh):
        eb = self.eb; out = bytearray(ew * eh * eb)
        tx_n = (ew + self.tw - 1) // self.tw; ty_n = (eh + self.th - 1) // self.th
        tiled = ew > self.tw or eh > self.th
        macro = self.macro and tx_n % 4 == 0 and ty_n % 4 == 0
        lut = [[self.in_tile(x, y) for x in range(self.tw)] for y in range(self.th)]
        for y in range(eh):
            for x in range(ew):
                if tiled:
                    tx, ty = x // self.tw, y // self.th
                    if macro:
                        lx, ly = tx & 3, ty & 3
                        t = ((ty >> 2) * (tx_n >> 2) + (tx >> 2)) * 16 + ((ly & 1) | (lx & 1) << 1 | (ly >> 1) << 2 | (lx >> 1) << 3)
                    else:
                        t = ty * tx_n + tx
                    e = t * (4096 // eb) + lut[y % self.th][x % self.tw]
                else:
                    e = self.in_tile(x, y)
                so = base + e * eb
                if so + eb <= len(src):
                    d = (y * ew + x) * eb; out[d:d + eb] = src[so:so + eb]
        return bytes(out)


def png(path, w, h, rgb):
    raw = b"".join(b"\0" + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                          + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def inflate(arc, f):
    arc.seek(f["off"]); return zlib.decompress(arc.read(f["stored"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("index"); ap.add_argument("dict", type=lambda s: int(s, 0))
    ap.add_argument("names", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    files = json.load(open(a.index))["files"]
    arc = open(a.archive, "rb")
    i = next(k for k, f in enumerate(files) if f["off"] == a.dict)
    v = inflate(arc, files[i])
    phys = bytearray(); k = i + 1
    while k < len(files):
        h = bytes.fromhex(files[k]["head"])
        if files[k]["off"] & 0xFFF == 0x10 and len(h) >= 12 and h[11] >> 4 == 5:
            break
        phys += inflate(arc, files[k]); k += 1
    def vp(o):
        x = struct.unpack_from("<Q", v, o)[0]; return x & 0x0FFFFFFF if x >> 28 == 5 else None
    def pp(o):
        x = struct.unpack_from("<Q", v, o)[0]; return x & 0x0FFFFFFF if x >> 28 == 6 else None
    ta, n = vp(0x30), struct.unpack_from("<H", v, 0x38)[0]
    os.makedirs(a.out, exist_ok=True)
    for t in (vp(ta + 8 * j) for j in range(n)):
        np_ = vp(t + 0x30); s = np_
        while v[s] == 0 and s < np_ + 8: s += 1
        name = v[s:v.index(b"\0", s)].decode("latin-1")
        if name not in a.names: continue
        w, h, d = struct.unpack_from("<HHH", v, t + 0x38); fmt = v[t + 0x3F]; flags = struct.unpack_from("<H", v, t + 0x10)[0]
        data = pp(t + 0x18); eb = EB.get(fmt)
        mode = (flags >> 8) & 3
        bs = 4 if fmt in BC else 1
        ew, eh = (w + bs - 1) // bs, (h + bs - 1) // bs
        lay = Layout(eb, mode)
        mips = v[t + 0x42]
        # Asset-Info notes section 4: mips are stored smallest first; mip m starts at
        # align_up(sum of the slot sizes of mips m+1..N-1, 4096) from the data pointer.
        tail = 0
        for m in range(mips - 1, 0, -1):
            mw, mh = max(1, w >> m), max(1, h >> m)
            tail += lay.slot_bytes((mw + bs - 1) // bs, (mh + bs - 1) // bs)
        base = data + (tail + 4095) // 4096 * 4096
        lin = lay.linear(phys, base, ew, eh)
        info = dict(mips=mips, name=name, w=w, h=h, depth=d, dxgi=fmt, flags=hex(flags), mode=mode, phys_off=data, phys_bytes=len(phys))
        if a.raw: open(os.path.join(a.out, name + ".bin"), "wb").write(lin)
        px = T.decode_pixels(fmt, lin, ew, eh)
        if px is None:
            if fmt == 2: px = [struct.unpack_from("<3f", lin, p * 16) for p in range(ew * eh)]
            elif fmt == 41: px = [(q,) * 3 for q in struct.unpack_from(f"<{ew*eh}f", lin)]
            elif fmt == 24:
                px = []
                for (q,) in struct.iter_unpack("<I", lin):
                    px.append(((q & 1023) / 4.0115, ((q >> 10) & 1023) / 4.0115, ((q >> 20) & 1023) / 4.0115))
        if px is None:
            info["png"] = "no decoder"; print(json.dumps(info)); continue
        W, H = ew * bs, eh * bs
        flt = fmt in (2, 10, 41, 54, 95, 96)
        vals = [c for p in px for c in p[:3]]
        lo, hi = min(vals), max(vals)
        scale = 255.0 / (hi - lo) if (flt and hi > lo) else 1.0
        rgb = bytearray(max(0, min(255, int(round((c - lo) * scale if flt else c)))) for c in vals)
        png(os.path.join(a.out, name + ".png"), W, H, rgb)
        info.update(png=f"{W}x{H}", value_range=[round(lo, 4), round(hi, 4)], mean=round(sum(vals) / len(vals), 4))
        print(json.dumps(info))


main()
