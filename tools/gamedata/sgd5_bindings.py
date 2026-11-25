"""Read-only: dump the per-program binding trailer of SGD5 program records in shaders.rpf
and try to name its hashes with joaat over a candidate-string file.

python3 -I sgd5_bindings.py SHADERS_RPF --name SUBSTR [--dict STRINGS.txt] [--max 3]
python3 -I sgd5_bindings.py SHADERS_RPF --census [--dict STRINGS.txt]
python3 -I sgd5_bindings.py SHADERS_RPF --census --db 1   (second database)
Layout: see parse() and notes/gamedata_recon.md section 2a.
"""
import os  # noqa: E401
import argparse, struct, sys, re, collections
PAL = os.environ.get("GAME_ROOT", ".") + "/upstream/palmetto/tools"
sys.path.insert(0, PAL)
import sgd5_effects as S  # noqa: E402
from hashdict import joaat  # noqa: E402


def records(st, groups, want=None, limit=None):
    for g, offs in enumerate(groups):
        for start in offs:
            st.seek(start)
            ln = st.read(1)[0]
            label = st.read(ln)[:-1].decode("latin-1")
            if want and want not in label:
                continue
            st.read(1)
            size = S.u32(st.read(4))
            st.seek(size, 1)
            extra = st.read(152)
            tl = S.u32(st.read(4))
            yield g, start, label, extra, st.read(tl)


def parse(tr):
    """Header: u16 resource rows, u16 total member rows, u16 0, u16 table size. Resource rows are 24 B:
    u8 kind, u8 ?, u16 register, u32 flags (low byte = member count, high u16 = byte offset from this row to its
    first member row), u32 0, u32 size, u32 hash, u32 hash. Member rows are 24 B: u16 type, u16 array count,
    u32 byte offset, 8 zero bytes, u32 hash, u32 0. Kind 0x30 = cbuffer, 0xb0 = push-constant block."""
    if len(tr) < 8: return [], []
    cnt, total = struct.unpack_from("<HH", tr, 0)
    rows = []; mrows = []
    for i in range(cnt):
        o = 8 + 24 * i
        if o + 24 > len(tr): return None, None
        kind, b1, slot, flags, z, size, h1, h2 = struct.unpack_from("<BBHIIIII", tr, o)
        rows.append(dict(kind=kind, b1=b1, slot=slot, flags=flags, size=size, hash=h1))
        n = flags & 0xFF
        if n:
            m0 = o + (flags >> 16)
            for k in range(n):
                q = m0 + 24 * k
                if q + 24 > len(tr): return None, None
                t, arr, off, z0, z1, h, z2 = struct.unpack_from("<HHIIIII", tr, q)
                mrows.append(dict(type=t, array=arr, offset=off, hash=h, cb=h1, kind=kind))
    if len(mrows) != total: return None, None
    return rows, mrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rpf"); ap.add_argument("--name"); ap.add_argument("--dict"); ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--census", action="store_true"); ap.add_argument("--db", type=int, default=0, help="0 = main SGD5 database, 1 = second")
    a = ap.parse_args()
    names = {}
    if a.dict:
        for w in open(a.dict, encoding="latin-1").read().split():
            names.setdefault(joaat(w), w)
    st = open(a.rpf, "rb")
    db, first, end = S.DATABASES[a.db]
    st.seek(first)
    groups = S.program_groups(st, end)
    if a.census:
        n = 0; res = collections.Counter(); mem = collections.Counter(); kinds = collections.Counter()
        cbs = 0; members = 0; bad = 0; with_cb = 0; mtypes = collections.Counter(); pc_prog = pc_mem = inb = outb = 0
        for g, start, label, extra, tr in records(st, groups):
            n += 1
            rows, mrows = parse(tr)
            if rows is None: bad += 1; continue
            for r in rows:
                kinds[r["kind"]] += 1; res[r["hash"]] += 1
            cbs += sum(1 for r in rows if r["kind"] == 0x30); with_cb += any(r["kind"] == 0x30 for r in rows)
            pcs = [r for r in rows if r["kind"] == 0xB0]; pc_prog += bool(pcs)
            sz = {r["hash"]: r["size"] for r in rows}
            for m in mrows:
                inb += m["offset"] < sz.get(m["cb"], 0); outb += m["offset"] >= sz.get(m["cb"], 0)
                if m["kind"] == 0xB0: pc_mem += 1
            for m in mrows:
                members += 1; mem[m["hash"]] += 1; mtypes[m["type"]] += 1
        rr = sum(1 for h in res if h in names); mr = sum(1 for h in mem if h in names)
        ru = sum(c for h, c in res.items() if h in names); mu = sum(c for h, c in mem.items() if h in names)
        print(f"programs {n} (unparsed {bad}); programs with a cbuffer {with_cb}; cbuffers {cbs}; member rows {members} (push-constant members {pc_mem} in {pc_prog} programs); offset inside block size {inb}, outside {outb}")
        print(f"resource names: {len(res)} distinct, {rr} named by dict ({ru}/{sum(res.values())} uses)")
        print(f"member names: {len(mem)} distinct, {mr} named by dict ({mu}/{sum(mem.values())} uses)")
        print("resource kinds", [(hex(k), c) for k, c in kinds.most_common()])
        print("member type codes", [(hex(k), c) for k, c in mtypes.most_common()])
        print("top resources", [(f"{h:08x}", c, names.get(h)) for h, c in res.most_common(25)])
        print("top members", [(f"{h:08x}", c, names.get(h)) for h, c in mem.most_common(40)])
        return
    shown = 0
    for g, start, label, extra, tr in records(st, groups, a.name):
        print(f"group {g} @{start:#x} {label} trailer {len(tr)}")
        rows, mrows = parse(tr)
        if rows is None:
            print("   unparsed"); continue
        for i, r in enumerate(rows):
            print(f"   row {i}: kind {r['kind']:#04x} reg {r['slot']} members {r['flags'] & 0xFF} size {r['size']:#x} hash {r['hash']:08x} {names.get(r['hash'], '')}")
        for m in mrows:
            print(f"     member of {m['cb']:08x}: type {m['type']:#x} array {m['array']} offset {m['offset']} hash {m['hash']:08x} {names.get(m['hash'], '')}")
        shown += 1
        if shown >= a.max: break


main()
