"""Read-only peek at the SGD5 effect databases in shaders.rpf (Xbox 1.0.2.1).

python3 -I sgd5_peek.py SHADERS_RPF [--effect NAME] [--program-sample N]
Prints sizes of the effect region, per-effect field sizes, the string pool and
(for --program-sample) the 152-byte block plus trailer after N program records.
Reads program headers by seeking; reads only the effect region in full.
"""
import os  # noqa: E401
import argparse, os, struct, sys, json, re
PAL = os.environ.get("GAME_ROOT", ".") + "/upstream/palmetto/tools"
sys.path.insert(0, PAL)
import sgd5_effects as S  # noqa: E402


def hexdump(b, base=0, n=None):
    b = b[:n] if n else b
    for i in range(0, len(b), 16):
        row = b[i:i + 16]
        print("  %08x  %-48s %s" % (base + i, row.hex(" "), "".join(chr(c) if 32 <= c < 127 else "." for c in row)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rpf")
    ap.add_argument("--effect")
    ap.add_argument("--program-sample", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    with open(a.rpf, "rb") as st:
        for db, first, end in S.DATABASES:
            st.seek(first)
            groups = S.program_groups(st, end)
            count = S.u32(st.read(4))
            base = st.tell()
            print(f"DB {db:#x}: groups {[len(g) for g in groups]} effects {count} effect region {base:#x}..{end:#x} ({end-base} bytes)")
            if a.program_sample and db == S.DATABASES[0][0]:
                for g, offs in enumerate(groups):
                    for start in offs[:a.program_sample]:
                        st.seek(start)
                        ln = st.read(1)[0]
                        label = st.read(ln)
                        tag = st.read(1)[0]
                        size = S.u32(st.read(4))
                        prog = st.read(size)
                        extra = st.read(152)
                        trailer = S.u32(st.read(4))
                        tr = st.read(trailer)
                        print(f" group {g} @{start:#x} label {label[:-1]!r} tag {tag} size {size} prog-head {prog[:16].hex()} trailer {trailer}")
                        print("  152-byte block:"); hexdump(extra, 0)
                        print("  trailer:"); hexdump(tr, 0, 512)
            st.seek(base)
            data = st.read(end - base)
            sizes = []
            for e in S.effect_records(data, groups, count):
                fields = {"name": e.name, "bytes": e.end - e.start, "pool": e.end - e.pool,
                          "post_pass": e.pool - e.pass_end}
                sizes.append(fields)
                if a.effect and e.name == a.effect:
                    print(f"== effect {e.name} @{base + e.start:#x} total {e.end - e.start}")
                    hexdump(data[e.start:e.end], base + e.start, 6000)
                    pool = data[e.pool + 4:e.end]
                    print("  pool strings:", [s.decode() for s in pool.split(b"\0") if s][:400])
            if a.summary:
                big = sorted(sizes, key=lambda x: -x["bytes"])[:5]
                print("  largest effects", big)
                print("  total pool bytes", sum(x["pool"] for x in sizes), "post-pass bytes", sum(x["post_pass"] for x in sizes))


main()
