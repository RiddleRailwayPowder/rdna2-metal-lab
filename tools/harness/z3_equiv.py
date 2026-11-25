"""Prove a mutant equivalent by checking its statement with Z3 (Homebrew `z3` binary, SMT-LIB text).

For a mutant that changes an integer constant, the statement it sits in (an assignment's right-hand side, an `if`
condition or a declaration's initialiser) is turned into a bit-vector formula for the reference and for the mutant.
SSA definitions it reads (`T _N = expr;` in main0) are inlined while they are pure integer code; every other value
(registers, loads, float reinterpretations, calls) is a free variable, shared between the two sides when its text is
identical. Z3 is asked for an assignment where the two differ: unsat means no input can tell them apart.

Sound by construction: free variables over-approximate what the program can produce and everything the GPU leaves
undefined (shift by >= width, division by zero, out-of-range bit fields) becomes a fresh unconstrained value on
each side, so a proof never relies on a choice the hardware might make differently.

    ok, why = prove_equivalent(msl, mutant_record)   # ok: True (proved), False (counterexample), None (out of scope)
"""
import itertools
import re
import subprocess

TOKEN = re.compile(r"\s*(as_type<\s*\w+\s*>|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(?:ul|u|f|h)?|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>?:()\[\],.])")
TYPES = {"uint": (32, False), "int": (32, True), "ulong": (64, False), "long": (64, True), "ushort": (16, False),
         "short": (16, True), "uchar": (8, False), "char": (8, True), "bool": (1, False)}


class OutOfScope(Exception):
    pass


class Smt:
    """Builds SMT-LIB terms; expressions are (term, width, signed). width 1 + is_bool = boolean."""

    def __init__(self, var_types, defs, consts, side):
        self.var_types, self.defs, self.consts, self.side = var_types, defs, consts, side
        self.decls = {}
        self.fresh = itertools.count()
        self.depth = 0

    def var(self, name, width, signed):
        key = f"v_{re.sub(r'[^A-Za-z0-9_]', '_', name)}_{width}"
        self.decls[key] = f"(declare-const {key} (_ BitVec {width}))"
        return (key, width, signed, False)

    def unknown(self, width, signed=False, boolean=False):
        key = f"u{self.side}_{next(self.fresh)}"
        self.decls[key] = f"(declare-const {key} {'Bool' if boolean else f'(_ BitVec {width})'})"
        return (key, width, signed, boolean)


def bv(n, width):
    return f"(_ bv{n % (1 << width)} {width})"


def to_width(e, width, signed):
    t, w, s, b = e
    if b:
        return (f"(ite {t} {bv(1, width)} {bv(0, width)})", width, signed, False)
    if w == width:
        return (t, width, signed, False)
    if w > width:
        return (f"((_ extract {width - 1} 0) {t})", width, signed, False)
    return (f"((_ {'sign_extend' if s else 'zero_extend'} {width - w}) {t})", width, signed, False)


def to_bool(e):
    t, w, s, b = e
    return t if b else f"(not (= {t} {bv(0, w)}))"


class Parser:
    def __init__(self, text, smt):
        self.toks = [t for t in TOKEN.findall(text)]
        joined = "".join(self.toks)
        if joined.replace(" ", "") != re.sub(r"\s+", "", text):
            raise OutOfScope("untokenisable text")
        self.i = 0
        self.smt = smt

    def peek(self, k=0):
        return self.toks[self.i + k] if self.i + k < len(self.toks) else None

    def take(self, expect=None):
        t = self.peek()
        if expect is not None and t != expect:
            raise OutOfScope(f"expected {expect!r}, got {t!r}")
        self.i += 1
        return t

    # precedence climbing over C binary operators
    PREC = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"),
            ("*", "/", "%")]

    def expr(self):
        c = self.binary(0)
        if self.peek() == "?":
            self.take()
            a = self.expr()
            self.take(":")
            b = self.expr()
            return self.ite(c, a, b)
        return c

    def ite(self, c, a, b):
        cond = to_bool(c)
        if a[3] and b[3]:
            return (f"(ite {cond} {a[0]} {b[0]})", 1, False, True)
        w = max(a[1] if not a[3] else 32, b[1] if not b[3] else 32)
        sa, sb = to_width(a, w, a[2]), to_width(b, w, b[2])
        return (f"(ite {cond} {sa[0]} {sb[0]})", w, a[2] and b[2], False)

    def binary(self, level):
        if level == len(self.PREC):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in self.PREC[level]:
            op = self.take()
            right = self.binary(level + 1)
            left = self.apply(op, left, right)
        return left

    def apply(self, op, a, b):
        smt = self.smt
        if op in ("&&", "||"):
            return (f"({'and' if op == '&&' else 'or'} {to_bool(a)} {to_bool(b)})", 1, False, True)
        if a[3] and b[3] and op in ("==", "!=", "&", "|", "^"):
            core = {"==": "=", "!=": "distinct", "&": "and", "|": "or", "^": "xor"}[op]
            return (f"({core} {a[0]} {b[0]})", 1, False, True)
        w = max(1 if a[3] else a[1], 1 if b[3] else b[1], 32)
        if op in ("<<", ">>"):
            w = a[1] if not a[3] else 32
            x = to_width(a, w, a[2])
            n = to_width(b, w, False)
            if op == "<<":
                core = f"(bvshl {x[0]} {n[0]})"
            else:
                core = f"({'bvashr' if a[2] else 'bvlshr'} {x[0]} {n[0]})"
            u = smt.unknown(w, a[2])
            return (f"(ite (bvult {n[0]} {bv(w, w)}) {core} {u[0]})", w, a[2], False)
        signed = (a[2] if not a[3] else False) and (b[2] if not b[3] else False)
        x, y = to_width(a, w, signed), to_width(b, w, signed)
        if op in ("==", "!="):
            return (f"({'=' if op == '==' else 'distinct'} {x[0]} {y[0]})", 1, False, True)
        if op in ("<", ">", "<=", ">="):
            core = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge"}[op]
            return (f"(bv{'s' if signed else 'u'}{core} {x[0]} {y[0]})", 1, False, True)
        if op in ("/", "%"):
            core = {("/", False): "bvudiv", ("/", True): "bvsdiv", ("%", False): "bvurem", ("%", True): "bvsrem"}[(op, signed)]
            u = smt.unknown(w, signed)
            return (f"(ite (= {y[0]} {bv(0, w)}) {u[0]} ({core} {x[0]} {y[0]}))", w, signed, False)
        core = {"+": "bvadd", "-": "bvsub", "*": "bvmul", "&": "bvand", "|": "bvor", "^": "bvxor"}[op]
        return (f"({core} {x[0]} {y[0]})", w, signed, False)

    def unary(self):
        t = self.peek()
        if t == "!":
            self.take()
            return (f"(not {to_bool(self.unary())})", 1, False, True)
        if t == "~":
            self.take()
            e = self.unary()
            e = to_width(e, 32 if e[3] else e[1], e[2])
            return (f"(bvnot {e[0]})", e[1], e[2], False)
        if t == "-":
            self.take()
            e = self.unary()
            e = to_width(e, 32 if e[3] else e[1], e[2])
            return (f"(bvneg {e[0]})", e[1], e[2], False)
        return self.postfix(self.primary())

    def postfix(self, e):
        return e

    def args(self):
        self.take("(")
        out = []
        if self.peek() != ")":
            out.append(self.expr())
            while self.peek() == ",":
                self.take()
                out.append(self.expr())
        self.take(")")
        return out

    def primary(self):
        smt = self.smt
        t = self.take()
        if t == "(":
            e = self.expr()
            self.take(")")
            return e
        if t is None:
            raise OutOfScope("unexpected end")
        if re.fullmatch(r"\d+(?:ul|u)?", t):
            n = int(re.match(r"\d+", t).group())
            if t.endswith("ul"):
                return (bv(n, 64), 64, False, False)
            if t.endswith("u"):
                return (bv(n, 32), 32, False, False)
            return (bv(n, 32), 32, True, False)
        if re.fullmatch(r"\d", t[0]):
            raise OutOfScope("float literal")
        if t in ("true", "false"):
            return (t, 1, False, True)
        if t.startswith("as_type<"):
            ty = re.search(r"<\s*(\w+)\s*>", t).group(1)
            if ty not in TYPES or ty == "bool":
                raise OutOfScope(f"as_type<{ty}>")
            (inner,) = self.args()
            w, s = TYPES[ty]
            if inner[3] or inner[1] != w:
                raise OutOfScope("as_type width change")
            return (inner[0], w, s, False)
        if t in TYPES and self.peek() == "(":                        # conversion uint(x), ulong(x), bool(x)
            (inner,) = self.args()
            w, s = TYPES[t]
            if t == "bool":
                return (to_bool(inner), 1, False, True)
            # value conversion: from a narrower signed type sign-extends, else zero-extends / truncates
            src_signed = inner[2] and not inner[3]
            e = to_width((inner[0], inner[1], src_signed, inner[3]), w, src_signed) if inner[1] < w or inner[3] \
                else to_width(inner, w, s)
            return (e[0], w, s, False)
        if t == "extract_bits" and self.peek() == "(":
            x, off, n = self.args()
            w = x[1]
            off, n = to_width(off, w, False), to_width(n, w, False)
            core = (f"(bvand (bvlshr {x[0]} {off[0]}) (bvsub (bvshl {bv(1, w)} {n[0]}) {bv(1, w)}))")
            whole = f"(bvlshr {x[0]} {off[0]})"
            u = smt.unknown(w, x[2])
            ok = f"(bvule (bvadd {off[0]} {n[0]}) {bv(w, w)})"
            body = f"(ite (= {n[0]} {bv(w, w)}) {whole} {core})"
            return (f"(ite {ok} {body} {u[0]})", w, x[2], False)
        if t in ("min", "max") and self.peek() == "(":
            a, b = self.args()
            s = a[2] and b[2]
            w = max(a[1], b[1])
            x, y = to_width(a, w, s), to_width(b, w, s)
            lt = f"(bv{'s' if s else 'u'}lt {x[0]} {y[0]})"
            return (f"(ite {lt} {x[0] if t == 'min' else y[0]} {y[0] if t == 'min' else x[0]})", w, s, False)
        if re.fullmatch(r"_\d+", t):
            name = t
            if self.peek() == "[":                                   # array element
                self.take("[")
                idx = self.expr()
                self.take("]")
                if name in smt.consts and idx[0].startswith("(_ bv"):
                    k = int(re.match(r"\(_ bv(\d+)", idx[0]).group(1))
                    vals = smt.consts[name]
                    if k < len(vals):
                        return (bv(vals[k], 32), 32, False, False)
                if name in smt.consts:
                    vals = smt.consts[name]
                    i32 = to_width(idx, 32, False)
                    e = smt.unknown(32)[0]
                    for k in range(len(vals) - 1, -1, -1):
                        e = f"(ite (= {i32[0]} {bv(k, 32)}) {bv(vals[k], 32)} {e})"
                    return (e, 32, False, False)
                if idx[0].startswith("(_ bv"):
                    return smt.var(f"{name}[{idx[0]}]", 32, False)
                raise OutOfScope("register read with computed index")
            if self.peek() == ".":
                self.take(".")
                field = self.take()
                text = f"{name}.{field}"
                if self.peek() == "[":
                    self.take("[")
                    idx = self.expr()
                    self.take("]")
                    if not idx[0].startswith("(_ bv"):
                        raise OutOfScope("member read with computed index")
                    text += f"[{idx[0]}]"
                if field in ("x", "y", "z", "w"):
                    raise OutOfScope("vector component")
                return smt.var(text, 32, False)
            if name in smt.defs and smt.depth < 12:                   # inline a pure SSA definition
                smt.depth += 1
                try:
                    e = Parser(smt.defs[name], smt).full()
                except OutOfScope:
                    e = None
                smt.depth -= 1
                if e is not None:
                    return e
            w, s = smt.var_types.get(name, (32, False))
            if w == 1:
                key = f"v_{name}_b"
                smt.decls[key] = f"(declare-const {key} Bool)"
                return (key, 1, False, True)
            return smt.var(name, w, s)
        raise OutOfScope(f"token {t!r}")

    def full(self):
        e = self.expr()
        if self.i != len(self.toks):
            raise OutOfScope(f"trailing {self.toks[self.i:self.i + 3]}")
        return e


def _main_body(msl):
    m = re.search(r"\b(fragment|kernel)\s+[\w:<> ]+\s+main0\s*\(", msl)
    return msl[m.start():] if m else msl


def facts(msl):
    """Variable types, pure SSA definitions and constant uint tables of a shader."""
    body = _main_body(msl)
    var_types, defs = {}, {}
    for ty, name, rhs in re.findall(r"^\s*(?:const\s+)?(uint|int|ulong|long|bool|ushort|short)\s+(_\d+)\s*=\s*(.*);\s*$",
                                     body, re.M):
        var_types[name] = TYPES[ty]
        if "as_type<float" not in rhs and "sample" not in rhs and "reinterpret_cast" not in rhs:
            defs.setdefault(name, rhs) if body.count(f" {name} = ") == 1 else None
    for ty, name in re.findall(r"^\s*(uint|int|ulong|long|bool)\s+(_\d+)\s*;", body, re.M):
        var_types[name] = TYPES[ty]
    consts = {}
    for name, vals in re.findall(r"constant spvUnsafeArray<uint, \d+> (_\d+) = spvUnsafeArray<uint, \d+>\(\{([^}]*)\}\)", msl):
        consts[name] = [int(x) for x in re.findall(r"(\d+)u", vals)]
    return var_types, defs, consts


def statement_expr(line):
    """The expression a mutated statement computes, with harness wrappers removed."""
    s = line.strip()
    s = re.sub(r"sbx_n?sel\(\d+, ", "(", s)
    m = re.match(r"if \((.*)\)$", s)
    if m:
        return m.group(1)
    m = re.match(r"^(?:const\s+)?(?:\w+\s+)?[\w.\[\]]+\s*=\s*(.*);$", s)
    if m:
        return m.group(1)
    raise OutOfScope("not an assignment or if")


def prove_equivalent(msl, m, timeout_s=20):
    if m.get("kind") != "int_const":
        return None, "only integer-constant mutants"
    lines = msl.split("\n")
    line = lines[m["line"] - 1]
    col = m["col"]
    if line[col:col + len(m["orig"])] != m["orig"]:
        return None, "site text mismatch"
    mut_line = line[:col] + m["rep"] + line[col + len(m["orig"]):]
    var_types, defs, consts = facts(msl)
    try:
        ea, eb = statement_expr(line), statement_expr(mut_line)
        sa, sb = Smt(var_types, defs, consts, "a"), Smt(var_types, defs, consts, "b")
        a = Parser(ea, sa).full()
        b = Parser(eb, sb).full()
    except OutOfScope as e:
        return None, f"out of scope: {e}"
    except (IndexError, ValueError, RecursionError) as e:
        return None, f"parse failed: {type(e).__name__}"
    if a[3] != b[3]:
        return None, "type mismatch"
    if a[3]:
        goal = f"(distinct {a[0]} {b[0]})"
    else:
        w = max(a[1], b[1])
        goal = f"(distinct {to_width(a, w, a[2])[0]} {to_width(b, w, b[2])[0]})"
    decls = dict(sa.decls)
    decls.update(sb.decls)
    smt = "(set-logic QF_BV)\n" + "\n".join(decls.values()) + f"\n(assert {goal})\n(check-sat)\n"
    try:
        r = subprocess.run(["z3", "-in", f"-T:{timeout_s}"], input=smt, capture_output=True, text=True,
                           timeout=timeout_s + 10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"z3 failed: {type(e).__name__}"
    out = r.stdout.strip()
    if out == "unsat":
        return True, "z3: statement value identical for every input"
    if out == "sat":
        return False, "z3: inputs exist where the statement differs"
    return None, f"z3: {out[:80] or r.stderr.strip()[:80]}"
