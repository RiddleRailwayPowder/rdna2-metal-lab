"""Prove a mutant equivalent because the value it changes is dead (never reaches an output, memory or control flow).

main0's body (SPIRV-Cross layout: one statement per line, braces on their own lines) is parsed into a tree of
statements, if/else, loops (for, while, do-while) and switch, then a backward strong-liveness analysis runs over it:
  - locations: scalar variables `_N` and register-array slots `_N[ku]` with a constant index;
  - a read with a computed index (`_N[_M]`) or a bare array name keeps every slot of that array live;
  - a statement's reads count only if the statement matters: it defines a live location, or it has a side effect
    (writes to `out`, device memory, textures or threadgroup arrays, atomics, barriers, discard, calls to helper
    functions, which may write through references); conditions of if/loop/switch always count;
  - loops iterate to a fixpoint; break/continue/return/fall-through are followed.
A mutant whose statement only defines locations that are dead right after it can't change anything observable:
equivalent. Everything uncertain counts as live, so the proof is conservative.

    ok, why = prove_dead(msl, mutant_record)    # True (dead), False (live), None (out of scope)
"""
import re

SLOT = re.compile(r"\b(_\d+)\[(\d+)u?\]")
DYN = re.compile(r"\b(_\d+)\[(?!\d+u?\])")
VAR = re.compile(r"\b_\d+\b")
SIDE = re.compile(r"\bout\.|reinterpret_cast<device|atomic_|\.write\(|\.atomic|threadgroup_barrier|simdgroup_barrier|"
                  r"discard_fragment|\b_\d+\(|spvImage|\.fence\(")


class OutOfScope(Exception):
    pass


class Node:
    def __init__(self, kind, text="", line=-1, children=None, extra=None):
        self.kind, self.text, self.line = kind, text, line
        self.children = children or []
        self.extra = extra or {}


def _main_lines(msl):
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    if not m:
        raise OutOfScope("no main0")
    start = msl.index("{", m.end())
    first_line = msl.count("\n", 0, start)            # 0-based line index of the opening brace
    lines = msl.split("\n")
    depth, end = 0, None
    for k in range(first_line, len(lines)):
        s = lines[k].strip()
        if s == "{":
            depth += 1
        elif s.startswith("}"):
            depth -= 1
            if depth == 0:
                end = k
                break
    if end is None:
        raise OutOfScope("unbalanced main0")
    return lines, first_line + 1, end                 # body lines [first_line + 1, end)


class Parser:
    def __init__(self, lines, lo, hi):
        self.lines, self.i, self.hi = lines, lo, hi

    def peek(self):
        while self.i < self.hi and not self.lines[self.i].strip():
            self.i += 1
        return self.lines[self.i].strip() if self.i < self.hi else None

    def block(self):
        """`{` ... `}` -> list of nodes; the caller has consumed the head line."""
        if self.peek() != "{":
            raise OutOfScope(f"expected '{{' at line {self.i + 1}")
        self.i += 1
        out = []
        while True:
            s = self.peek()
            if s is None:
                raise OutOfScope("unterminated block")
            if s.startswith("}"):
                return out
            out.append(self.stmt())

    def close(self):
        s = self.peek()
        if s is None or not s.startswith("}"):
            raise OutOfScope("expected '}'")
        self.i += 1
        return s

    def stmt(self):
        s = self.peek()
        k = self.i
        if s == "else" or s.startswith("else if ("):
            raise OutOfScope("dangling else")
        if re.match(r"if \(.*\)$", s):
            self.i += 1
            cond = s[4:-1]
            then = self.block()
            self.close()
            els = None
            nxt = self.peek()
            if nxt == "else":
                self.i += 1
                els = self.block()
                self.close()
            elif nxt is not None and nxt.startswith("else if ("):
                self.lines[self.i] = self.lines[self.i].replace("else if (", "if (", 1)
                els = [self.stmt()]
            return Node("if", cond, k, [then, els])
        if re.match(r"(for|while) \(.*\)$", s):
            self.i += 1
            body = self.block()
            self.close()
            return Node("loop", s, k, [body], {"head": s})
        if s == "do":
            self.i += 1
            body = self.block()
            tail = self.close()
            m = re.match(r"\} while \((.*)\);$", tail)
            if not m:
                raise OutOfScope("do without while")
            return Node("do", m.group(1), k, [body])
        if re.match(r"switch \(.*\)$", s):
            self.i += 1
            if self.peek() != "{":
                raise OutOfScope("switch without block")
            self.i += 1
            cases = []                                 # [(labels, [nodes])]
            while True:
                t = self.peek()
                if t is None:
                    raise OutOfScope("unterminated switch")
                if t.startswith("}"):
                    self.i += 1
                    break
                if re.match(r"(case .*|default):$", t):
                    self.i += 1
                    body = []
                    if self.peek() == "{":             # SPIRV-Cross wraps case bodies in braces
                        body = self.block()
                        self.close()
                    while True:
                        u = self.peek()
                        if u is None or u.startswith("}") or re.match(r"(case .*|default):$", u):
                            break
                        body.append(self.stmt())
                    cases.append((t, body))
                    continue
                raise OutOfScope(f"unexpected line in switch: {t[:40]}")
            return Node("switch", s[8:-1], k, [], {"cases": cases})
        if s == "{":
            body = self.block()
            self.close()
            return Node("scope", "", k, [body])
        self.i += 1
        if s in ("break;", "continue;"):
            return Node(s[:-1], "", k)
        if s.startswith("return"):
            return Node("return", s, k)
        return Node("stmt", s, k)


def uses_of(text):
    out = set()
    for a, kk in SLOT.findall(text):
        out.add(("a", a, int(kk)))
    for a in DYN.findall(text):
        out.add(("A", a))
    stripped = SLOT.sub(" ", text)
    for v in VAR.findall(stripped):
        out.add(("v", v))
        out.add(("A", v))                              # a bare array name (passed whole) keeps all slots live
    return out


def defs_of(text, threadgroup):
    """(defined locations, is_pure) for a statement line."""
    m = re.match(r"^(?:const\s+)?(?:[\w:<>, ]+\s+)?(_\d+)(?:\[(\d+)u?\])?\s*=[^=](.*)$", text)
    side = bool(SIDE.search(text))
    if not m:
        return set(), False
    name, idx, rhs = m.group(1), m.group(2), m.group(3)
    if name in threadgroup:
        return set(), False
    loc = ("a", name, int(idx)) if idx is not None else ("v", name)
    return {loc}, not side and not SIDE.search(rhs)


def is_live(loc, L):
    if loc[0] == "a":
        return loc in L or ("A", loc[1]) in L
    return loc in L


class Live:
    def __init__(self, threadgroup):
        self.tg = threadgroup
        self.after = {}                                # line -> live set right after that statement

    def seq(self, nodes, L, brk, cont):
        for n in reversed(nodes or []):
            L = self.node(n, L, brk, cont)
        return L

    def node(self, n, L, brk, cont):
        k = n.kind
        if k == "stmt":
            self.after[n.line] = L
            defs, pure = defs_of(n.text, self.tg)
            if not defs:                               # store, call, barrier, declaration: every read counts
                return L | uses_of(n.text)
            if pure and not any(is_live(d, L) for d in defs):
                return L                               # faint: nothing it defines is used, so its reads don't count
            return (L - defs) | uses_of(n.text.split("=", 1)[1])
        if k == "return":
            self.after[n.line] = set()
            return uses_of(n.text) | {("A", "out")}
        if k == "break":
            return set(brk) if brk is not None else L
        if k == "continue":
            return set(cont) if cont is not None else L
        if k == "scope":
            return self.seq(n.children[0], L, brk, cont)
        if k == "if":
            t = self.seq(n.children[0], L, brk, cont)
            e = self.seq(n.children[1], L, brk, cont) if n.children[1] is not None else L
            return t | e | uses_of(n.text)
        if k in ("loop", "do"):
            head = n.extra.get("head", "")
            cond_uses = uses_of(head if k == "loop" else n.text)
            H = set(L) | cond_uses
            for _ in range(200):
                if k == "do":
                    end = (H | L | cond_uses) if n.text.strip() != "false" else (L | cond_uses)
                    body_in = self.seq(n.children[0], end, L, end)
                    newH = body_in
                else:
                    body_in = self.seq(n.children[0], H, L, H)
                    newH = body_in | L | cond_uses
                if newH <= H:
                    break
                H = H | newH
            else:
                raise OutOfScope("loop fixpoint did not converge")
            return H | cond_uses if k == "loop" else body_in
        if k == "switch":
            nxt = L                                    # falling off the last case continues after the switch
            has_default = False
            for label, body in reversed(n.extra["cases"]):
                has_default |= label.startswith("default")
                nxt = self.seq(body, nxt, L, cont)
                n.extra.setdefault("ins", []).append(nxt)
            ins = set().union(*n.extra["ins"]) if n.extra["ins"] else set()
            n.extra["ins"] = []
            return ins | uses_of(n.text) | (set() if has_default else L)
        raise OutOfScope(f"node {k}")


def analyse(msl):
    lines, lo, hi = _main_lines(msl)
    tg = set(re.findall(r"threadgroup\s+[\w<>, :]+?\s*&?\s*(_\d+)", msl))
    tree = Parser(list(lines), lo, hi)
    nodes = []
    while tree.peek() is not None:
        nodes.append(tree.stmt())
    live = Live(tg)
    # run twice so loop fixpoints settle before the recorded sets are read
    live.seq(nodes, {("A", "out")}, None, None)
    live.after = {}
    live.seq(nodes, {("A", "out")}, None, None)
    return live.after, tg


def prove_dead(msl, m, cache=None):
    try:
        if cache is not None and "dead" in cache:
            after, tg = cache["dead"]
        else:
            after, tg = analyse(msl)
            if cache is not None:
                cache["dead"] = (after, tg)
    except (OutOfScope, RecursionError) as e:
        return None, f"out of scope: {e}"
    k = m["line"] - 1
    if k not in after:
        return None, "site is not in a plain statement"
    text = msl.split("\n")[k].strip()
    defs, pure = defs_of(text, tg)
    if not defs or not pure:
        return False, "statement has side effects or defines nothing"
    lhs = text.split("=", 1)[0]
    if m["col"] < len(msl.split("\n")[k]) - len(msl.split("\n")[k].lstrip()) + len(lhs):
        return None, "mutation is on the left-hand side"
    L = after[k]
    if any(is_live(d, L) for d in defs):
        return False, "value live after the statement"
    return True, "dead store: the defined value is never used on any path"
