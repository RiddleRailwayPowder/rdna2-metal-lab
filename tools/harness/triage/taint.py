"""Forward def-use taint over main0 of a prepared shader (ref.metal), with and without the coverage union.

Uses tools/harness/dead_store.py read-only for the statement tree and the def/use regexes.

    taint(msl, site_line, hits=None) -> dict(sinks=[(kind, line)], site_block=id, site_guard=id or None,
                                            site_reached=bool)
hits=None: every block runs (static def-use). hits=set: blocks whose sbx_hit(N) id is not in hits are skipped and
lane-guarded writes (`X = sbx_sel(id, c) ? new : X;`) whose guard id never fired keep X unchanged.

Taint: the locations the mutated statement defines are tainted right after it. A statement reading a tainted
location taints what it defines (strong update for scalars and constant-index slots, weak for computed indexes); a
statement with no definition (store, texture write, atomic, call, discard) reading taint is a sink; an if/loop/switch
condition reading taint is a "control" sink. Loops run to a fixpoint (at most 6 passes); break/continue are ignored
(over-approximation: state after a loop = before it | after any number of passes).
"""
import os  # noqa: E401
import re
import sys

sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/harness")
import dead_store as ds  # noqa: E402  (read-only reuse of the parser)

HIT = re.compile(r"^sbx_hit\((\d+)\);$")
SEL = re.compile(r"=\s*sbx_sel\((\d+),")
NSEL = re.compile(r"=\s*sbx_nsel\((\d+),")
DYNSTORE = re.compile(r"^(_\d+)\[(?!\d+u?\]\s*=)(.+?)\]\s*=[^=]")
GUARD_ANY = re.compile(r"sbx_n?sel\((\d+),")


def sink_kind(text):
    if "discard_fragment" in text:
        return "discard"
    if text.startswith("out.") or re.search(r"\bout\.\w+\s*=", text):
        return "out"
    if ".write(" in text:
        return "texwrite"
    if "atomic" in text:
        return "atomic"
    if "reinterpret_cast<device" in text or text.startswith("(*"):
        return "store"
    if text.startswith("return"):
        return "return"
    if re.match(r"^_\d+\(", text) or re.search(r"\b_\d+\(", text):
        return "call"
    return "other"


class Tree:
    def __init__(self, msl):
        lines, lo, hi = ds._main_lines(msl)
        self.tg = set(re.findall(r"threadgroup\s+[\w<>, :]+?\s*&?\s*(_\d+)", msl))
        p = ds.Parser(list(lines), lo, hi)
        self.nodes = []
        while p.peek() is not None:
            self.nodes.append(p.stmt())
        self.lo, self.hi = lo, hi
        self.block_of = {}          # 0-based line -> innermost block id (sbx_hit)
        self.guard_of = {}          # 0-based line -> lane guard id
        self.discard_ifs = set()     # lines of if-nodes whose branches discard
        self._index(self.nodes, 1)

    def _bid(self, nodes, parent):
        if nodes and nodes[0].kind == "stmt":
            m = HIT.match(nodes[0].text)
            if m:
                return int(m.group(1))
        return parent

    def _index(self, nodes, bid):
        bid = self._bid(nodes, bid)
        for n in nodes or []:
            self.block_of[n.line] = bid
            if n.kind == "stmt":
                g = GUARD_ANY.search(n.text)
                if g:
                    self.guard_of[n.line] = int(g.group(1))
            if n.kind == "if":
                for br in n.children:
                    if br and any(c.kind == "stmt" and "discard_fragment" in c.text for c in br):
                        self.discard_ifs.add(n.line)
                self._index(n.children[0], bid)
                if n.children[1] is not None:
                    self._index(n.children[1], bid)
            elif n.kind in ("loop", "do", "scope"):
                self._index(n.children[0], bid)
            elif n.kind == "switch":
                for _, body in n.extra["cases"]:
                    self._index(body, bid)


def tainted_use(uses, T):
    for u in uses:
        if u in T:
            return True
        if u[0] == "a" and ("A", u[1]) in T:
            return True
        if u[0] == "A" and any(t[0] == "a" and t[1] == u[1] for t in T):
            return True
    return False


class Run:
    def __init__(self, tree, site, hits, eff=None):
        self.t, self.site, self.hits = tree, site, hits
        self.eff = eff or {}                               # 0-based line -> text with never-taken ternary arms blanked
        self.sinks = []
        self.site_reached = False

    def runs(self, nodes):
        if self.hits is None or not nodes:
            return True
        b = self.t._bid(nodes, None)
        return b is None or b in self.hits

    def seq(self, nodes, T):
        for n in nodes or []:
            T = self.node(n, T)
        return T

    def cond(self, text, line, T):
        if line == self.site:
            self.site_reached = True
            self.sinks.append(("control", line))
        elif tainted_use(ds.uses_of(text), T):
            self.sinks.append(("control", line))

    def stmt(self, n, T):
        text = self.eff.get(n.line, n.text)
        if HIT.match(text) or re.match(r"^[\w:<>, ]+\s+_\d+;$", text):
            return T
        if self.hits is not None:
            g = SEL.search(text) or NSEL.search(text)
            if g and int(g.group(1)) not in self.hits:
                return T                                   # guard never fired: X = X
        defs, _ = ds.defs_of(text, self.t.tg)
        is_site = n.line == self.site
        if is_site:
            self.site_reached = True
        rhs = text.split("=", 1)[1] if defs else text
        dirty = is_site or tainted_use(ds.uses_of(rhs if defs else text), T)
        if not defs:
            m = DYNSTORE.match(text)
            if m and m.group(1) not in self.t.tg:
                # computed-index store into a register array: weak update
                if dirty or tainted_use(ds.uses_of(text.split("=", 1)[1]), T):
                    T = T | {("A", m.group(1))}
                return T
            if dirty:
                self.sinks.append((sink_kind(text) if not (m and m.group(1) in self.t.tg) else "threadgroup", n.line))
            return T
        if dirty and re.search(r"atomic|\.write\(", rhs):
            self.sinks.append((sink_kind(rhs), n.line))      # loads through pointers and helper calls are reads
        if dirty:
            return T | defs
        return T - defs

    def node(self, n, T):
        k = n.kind
        if k == "stmt":
            return self.stmt(n, T)
        if k == "return":
            if n.line == self.site or tainted_use(ds.uses_of(n.text), T):
                self.sinks.append(("return", n.line))
            return T
        if k in ("break", "continue"):
            return T
        if k == "scope":
            return self.seq(n.children[0], T) if self.runs(n.children[0]) else T
        if k == "if":
            c = n.text
            e = self.eff.get(n.line)
            if e is not None:
                e = e[5:] if e.startswith("else ") else e
                if e.startswith("if (") and e.endswith(")"):
                    c = e[4:-1]
            self.cond(c, n.line, T)
            then, els = n.children
            outs = []
            if self.runs(then):
                outs.append(self.seq(then, T))
            if els is None:
                outs.append(T)
            elif self.runs(els):
                outs.append(self.seq(els, T))
            if not outs:
                return T
            return set().union(*outs)
        if k in ("loop", "do"):
            head = n.extra.get("head", n.text)
            body = n.children[0]
            if not self.runs(body):
                self.cond(head, n.line, T)
                return T
            acc = set(T)
            for _ in range(6):
                self.cond(head, n.line, acc)
                new = acc | self.seq(body, acc)
                if new == acc:
                    break
                acc = new
            return acc
        if k == "switch":
            self.cond(n.text, n.line, T)
            outs = [T]
            fall = None
            for _, body in n.extra["cases"]:
                if not self.runs(body):
                    continue
                start = T | (fall or set())
                fall = self.seq(body, start)
                outs.append(fall)
            return set().union(*outs)
        return T


_cache = {}


def tree_for(path):
    if path not in _cache:
        _cache[path] = Tree(open(path).read())
    return _cache[path]


def taint(path, line1, hits=None, eff=None):
    t = tree_for(path)
    r = Run(t, line1 - 1, hits, eff)
    T = r.seq(t.nodes, set())
    # one more pass so loop-carried / whole-body effects are not missed is unnecessary for straight-line main0
    return {"sinks": r.sinks, "site_reached": r.site_reached, "discard_ifs": t.discard_ifs,
            "site_block": t.block_of.get(line1 - 1), "site_guard": t.guard_of.get(line1 - 1)}
