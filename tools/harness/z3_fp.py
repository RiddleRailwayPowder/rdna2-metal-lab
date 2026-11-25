"""Float-aware equivalence prover for one mutant statement (Homebrew `z3` binary, SMT-LIB text, logic ALL).

Extends z3_equiv's statement-level check to code that mixes integer bit operations with float values:

* every float value is a 32-bit bit-vector; as_type<float>/as_type<uint>/as_type<int> are identity on the bits;
* float comparisons, abs, negation, isnan/isinf, max/min/clamp, floor/ceil/trunc/rint, int<->float conversions and
  half->float unpacking use FloatingPoint theory on ((_ to_fp 8 24) bits);
* float arithmetic (spvFMul/spvFAdd/spvFSub/fma and the mutant helpers spvMutFAdd/spvMutFSub/spvFmaSwap) is one
  uninterpreted function fma32(a, b, c) over raw bits, reached through each helper's real body
  (spvFMul(l,r)=fma(l,r,+0), spvFAdd(l,r)=fma(1,l,r), spvFSub(l,r)=fma(-1,r,l), spvFmaSwap(a,b,c)=fma(a,c,b)):
  identical call on identical bits gives identical bits, nothing else is assumed.

Hardware freedom (Apple GPUs may flush denormals; Metal leaves some results undefined) is modelled with fresh
symbols. Inside the mutated statement each side gets its own symbols, so a proof never relies on a choice the
hardware might make differently in the two compiled programs:
* every float operand of a comparison, abs, negation, max/min/clamp, rounding or half unpack may be flushed to a
  signed zero when it is subnormal (fresh Bool per occurrence and side);
* NaN results of abs/neg/max/min/conversions carry a fresh NaN payload;
* fast::/unqualified float max/min/clamp with a NaN operand, +-0 ties, float->int out of range or NaN, shifts >= width,
  division by zero, out-of-range bit fields and out-of-bounds register-array reads are fresh unknowns.
Values defined outside the statement (registers, SSA values, inputs) are shared free variables, as in z3_equiv. SSA
definitions the statement reads are inlined when they parse; their own choices are shared between the sides (they are
the same unchanged code) and their reads of mutable storage are renamed per definition, so a register read at the
definition is not confused with a later read of the same register in the statement.

Verdict:
    ok, reason, witness = prove_equivalent(msl, mutant_record)
    ok True  : unsat, no input and no hardware choice can make the statement's value differ
    ok False : sat with a deterministic witness (choices tied across sides, fma32 given IEEE RNE semantics wherever
               no operand or result is subnormal, NaN or tiny); witness maps every free value read by the statement
               (display text, `text@_N` when read inside inlined definition _N) to its 32-bit word or bool, plus
               '=ref' and '=mut' for the two statement values
    ok None  : out of scope, timeout or the values differ only under hardware-dependent choices / unmodelled
               denormal or NaN arithmetic
"""
import itertools
import re
import struct
import subprocess
from collections import Counter, namedtuple

E = namedtuple("E", "t w s k")          # term, width, signed, kind: 'bv' | 'bool' | 'f' (float, 32-bit bits)

TOKEN = re.compile(r"\s*((?:precise|fast|metal)::\w+|as_type<\s*\w+\s*>|\d+\.\d*(?:[eE][-+]?\d+)?[fh]?"
                   r"|\d+[eE][-+]?\d+[fh]?|\d+(?:ul|u)?|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>?:()\[\],.])")
INT_TYPES = {"uint": (32, False), "int": (32, True), "ulong": (64, False), "long": (64, True), "ushort": (16, False),
             "short": (16, True), "uchar": (8, False), "char": (8, True)}
BV32 = "(_ BitVec 32)"
FP32 = "(_ FloatingPoint 8 24)"
F_ZERO, F_ONE, F_MONE = 0x00000000, 0x3F800000, 0xBF800000
MIN_NORMAL = "((_ to_fp 8 24) #x00800000)"
INLINE_BUDGET = 40
FMA_FORMS = {"spvFMul": (2, lambda a: (a[0], a[1], F_ZERO)), "spvFAdd": (2, lambda a: (F_ONE, a[0], a[1])),
             "spvMutFAdd": (2, lambda a: (F_ONE, a[0], a[1])), "spvFSub": (2, lambda a: (F_MONE, a[1], a[0])),
             "spvMutFSub": (2, lambda a: (F_MONE, a[1], a[0])), "fma": (3, lambda a: (a[0], a[1], a[2])),
             "precise::fma": (3, lambda a: (a[0], a[1], a[2])), "spvFmaSwap": (3, lambda a: (a[0], a[2], a[1]))}
ROUNDING = {"floor": "RTN", "ceil": "RTP", "trunc": "RTZ", "rint": "RNE", "round": "RNA"}
OPAQUE_FLOAT_FNS = {"sqrt", "rsqrt", "exp2", "log2", "exp", "log", "sin", "cos", "fract", "pow", "powr", "tan",
                    "atan", "atan2", "acos", "asin", "fmod", "sign", "step", "smoothstep", "mix", "dot", "length"}


class OutOfScope(Exception):
    pass


def bv(n, width):
    return f"(_ bv{n % (1 << width)} {width})"


def fpv(t):
    return f"((_ to_fp 8 24) {t})"


def is_sub(t):
    return f"(and (= ((_ extract 30 23) {t}) #x00) (not (= ((_ extract 22 0) {t}) (_ bv0 23))))"


def const_value(t):
    m = re.fullmatch(r"\(_ bv(\d+) (\d+)\)", t)
    return int(m.group(1)) if m else None


def f32_bits(text):
    v = float(text.rstrip("fh"))
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    if struct.unpack("<f", struct.pack("<I", b))[0] != v:
        raise OutOfScope("inexact float literal")
    return b


def sort_of(e):
    return "Bool" if e.k == "bool" else f"(_ BitVec {e.w})"


def to_width(e, width, signed):
    if e.k == "bool":
        return E(f"(ite {e.t} {bv(1, width)} {bv(0, width)})", width, signed, "bv")
    if e.k == "f":
        raise OutOfScope("float used as integer")
    if e.w == width:
        return E(e.t, width, signed, "bv")
    if e.w > width:
        return E(f"((_ extract {width - 1} 0) {e.t})", width, signed, "bv")
    return E(f"((_ {'sign_extend' if e.s else 'zero_extend'} {width - e.w}) {e.t})", width, signed, "bv")


def to_bool(e):
    if e.k == "f":
        raise OutOfScope("float used as condition")
    return e.t if e.k == "bool" else f"(not (= {e.t} {bv(0, e.w)}))"


# ---------------------------------------------------------------------------------------------- shader facts

Facts = namedtuple("Facts", "types arrays fields defs ssa consts")
_FACTS = {}


def _main_body(msl):
    m = re.search(r"\b(fragment|kernel|vertex)\s+[\w:<> ]+\s+main0\s*\(", msl)
    return msl[m.start():] if m else msl


def facts(msl):
    key = hash(msl)
    if key in _FACTS:
        return _FACTS[key]
    types = {}
    for base, n, ptr, name in re.findall(r"\b(bool|uint|int|ulong|long|ushort|short|uchar|char|float|half)([234]?)"
                                         r"\s*([&*]?)\s*(_\d+|gl_\w+)\b(?!\s*\()", msl):
        t = "ptr" if ptr == "*" else (base, int(n or 1))
        types[name] = t if types.get(name, t) == t else "ambig"
    arrays = {}
    for base, n, name in re.findall(r"spvUnsafeArray<(\w+), (\d+)>\s*&?\s*(_\d+)\b", msl):
        t = (base, int(n))
        arrays[name] = t if arrays.get(name, t) == t else "ambig"
    fields = {}
    for body in re.findall(r"struct \w+\s*\{([^}]*)\}", msl):
        for base, n, name in re.findall(r"^\s*(bool|uint|int|ulong|long|float|half)([234]?)\s+(m_\w+|_m\d+)\b", body,
                                        re.M):
            fields.setdefault(name, set()).add((base, int(n or 1)))
    body = _main_body(msl)
    defs, ssa = {}, set()
    writes = Counter(re.findall(r"(?<![\w.])(_\d+)\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)", body))
    for ty, name, rhs in re.findall(r"^\s*(?:const\s+)?(uint|int|ulong|long|bool|ushort|short|float)\s+(_\d+)\s*=\s*(.*);"
                                    r"\s*$", body, re.M):
        if writes[name] == 1:
            ssa.add(name)
            defs[name] = rhs
    consts = {}
    for name, vals in re.findall(r"constant spvUnsafeArray<uint, \d+> (_\d+) = spvUnsafeArray<uint, \d+>\(\{([^}]*)\}\)",
                                 msl):
        consts[name] = [int(x) for x in re.findall(r"(\d+)u", vals)]
    f = Facts(types, arrays, fields, defs, ssa, consts)
    if len(_FACTS) > 64:
        _FACTS.clear()
    _FACTS[key] = f
    return f


# ---------------------------------------------------------------------------------------------- query state

class Query:
    """Declarations and side constraints shared by both sides of one SMT query."""

    def __init__(self, F, tied, noflush=False, clean=False):
        self.F, self.tied, self.noflush, self.clean = F, tied, noflush, clean
        self.decls = {}
        self.asserts = {}
        self.defs_done = {}
        self.leaves = {}
        self.apps = {}
        self.opaque = set()
        self.n = itertools.count()
        self.inlined = 0

    def declare(self, name, sort):
        self.decls.setdefault(name, f"(declare-const {name} {sort})")
        return name

    def fn(self, name, sig):
        self.decls.setdefault(name, f"(declare-fun {name} {sig})")

    def bind(self, term, sort):
        if len(term) <= 40 or re.fullmatch(r"[\w@.]+", term):
            return term
        name = f"t{next(self.n)}"
        self.decls[name] = f"(define-fun {name} () {sort} {term})"
        return name

    def assert_(self, t):
        self.asserts[t] = None


class Scope:
    """Naming of nondeterministic symbols. Statement side 'a'/'b': private per side unless the query ties them;
    inlined definition '_N': shared by both sides (unchanged code)."""

    def __init__(self, q, tag, is_def):
        self.q, self.tag, self.is_def = q, tag, is_def
        self.k = itertools.count()

    def _name(self, p, tie=True):
        k = next(self.k)
        if self.is_def:
            return f"{p}d{self.tag}_{k}"
        if self.q.tied and tie:
            return f"{p}s_{k}"
        return f"{p}{self.tag}_{k}"

    def choice(self):
        name = self.q.declare(self._name("c"), "Bool")
        if self.q.noflush:
            self.q.assert_(f"(not {name})")
        return name

    def avoid(self, when):
        """Witness search in clean mode stays off paths that read an undefined value or NaN payload."""
        if self.q.clean:
            self.q.assert_(f"(not {when})")

    def unknown(self, w=32, when="true"):
        self.avoid(when)
        return self.q.declare(self._name("u"), f"(_ BitVec {w})")

    def nan(self, when="true"):
        self.avoid(when)
        n = self.q.declare(self._name("n"), BV32)
        self.q.assert_(f"(fp.isNaN {fpv(n)})")
        return n

    def aux(self):
        return self.q.declare(self._name("x", tie=False), BV32)


# ---------------------------------------------------------------------------------------------- parser

class Parser:
    PREC = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"),
            ("*", "/", "%")]

    def __init__(self, text, q, sc, depth=0):
        self.toks = TOKEN.findall(text)
        if "".join(self.toks).replace(" ", "") != re.sub(r"\s+", "", text):
            raise OutOfScope("untokenisable text")
        self.i, self.q, self.sc, self.depth = 0, q, sc, depth

    # -- token helpers
    def peek(self, k=0):
        return self.toks[self.i + k] if self.i + k < len(self.toks) else None

    def take(self, expect=None):
        t = self.peek()
        if expect is not None and t != expect:
            raise OutOfScope(f"expected {expect!r}, got {t!r}")
        self.i += 1
        return t

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

    def full(self):
        e = self.expr()
        if self.i != len(self.toks):
            raise OutOfScope(f"trailing {self.toks[self.i:self.i + 3]}")
        return e

    # -- float helpers
    def as_float(self, e):
        if e.k == "f":
            return e.t
        c = const_value(e.t) if e.k == "bv" else None
        if c is not None:
            v = c - (1 << e.w) if e.s and c >> (e.w - 1) else c
            if abs(v) <= 1 << 24:
                return bv(struct.unpack("<I", struct.pack("<f", float(v)))[0], 32)
        raise OutOfScope("integer value used as float")

    def bindf(self, t):
        return self.q.bind(t, BV32)

    def flushin(self, t):
        """Operand as the hardware may read it: a subnormal may become a signed zero."""
        c = const_value(t)
        if c is not None and not (c & 0x7F800000 == 0 and c & 0x7FFFFF):
            return t
        t = self.bindf(t)
        self.sc.avoid(is_sub(t))
        return self.bindf(f"(ite (and {self.sc.choice()} {is_sub(t)}) (bvand {t} #x80000000) {t})")

    def fp_bits(self, fpterm, may_nan=True):
        """Bits of an FP-sorted value: exact for non-NaN values, fresh NaN payload otherwise."""
        T = self.q.bind(fpterm, FP32)
        r = self.sc.aux()
        if not may_nan:
            self.q.assert_(f"(= {fpv(r)} {T})")
            return r
        self.q.assert_(f"(or (fp.isNaN {T}) (= {fpv(r)} {T}))")
        return self.bindf(f"(ite (fp.isNaN {T}) {self.sc.nan(f'(fp.isNaN {T})')} {r})")

    def fma(self, a, b, c):
        q = self.q
        q.fn("fma32", f"({BV32} {BV32} {BV32}) {BV32}")
        a, b, c = (self.bindf(x if isinstance(x, str) else bv(x, 32)) for x in (a, b, c))
        key = (a, b, c)
        if key not in q.apps:
            app = q.bind(f"(fma32 {a} {b} {c})", BV32)
            if app == f"(fma32 {a} {b} {c})":
                app = f"t{next(q.n)}"
                q.decls[app] = f"(define-fun {app} () {BV32} (fma32 {a} {b} {c}))"
            A, B, C = fpv(a), fpv(b), fpv(c)
            R = q.bind(f"(fp.fma RNE {A} {B} {C})", FP32)
            g = f"g{next(q.n)}"
            q.decls[g] = (f"(define-fun {g} () Bool (and (not {is_sub(a)}) (not {is_sub(b)}) (not {is_sub(c)}) "
                          f"(not (fp.isNaN {R})) (or (fp.gt (fp.abs {R}) {MIN_NORMAL}) "
                          f"(and (or (fp.isZero {A}) (fp.isZero {B})) (fp.isZero {C})))))")
            q.apps[key] = (app, g, R)
        return E(q.apps[key][0], 32, False, "f")

    def fneg(self, x):
        x = self.flushin(x)
        return E(self.bindf(f"(ite (fp.isNaN {fpv(x)}) {self.sc.nan(f'(fp.isNaN {fpv(x)})')} (bvxor {x} #x80000000))"),
                 32, False, "f")

    def fabs(self, x):
        x = self.flushin(x)
        return E(self.bindf(f"(ite (fp.isNaN {fpv(x)}) {self.sc.nan(f'(fp.isNaN {fpv(x)})')} (bvand {x} #x7fffffff))"),
                 32, False, "f")

    def fminmax(self, op, a, b, fast):
        a, b = self.flushin(a), self.flushin(b)
        A, B = fpv(a), fpv(b)
        lt = f"(fp.lt {A} {B})"
        pick = f"(ite {lt} {b} {a})" if op == "max" else f"(ite {lt} {a} {b})"
        r = f"(ite (and (fp.isZero {A}) (fp.isZero {B})) (ite {self.sc.choice()} {a} {b}) {pick})"
        if fast:
            anynan = f"(or (fp.isNaN {A}) (fp.isNaN {B}))"
            r = f"(ite {anynan} {self.sc.unknown(32, anynan)} {r})"
        else:
            both = f"(and (fp.isNaN {A}) (fp.isNaN {B}))"
            r = f"(ite (fp.isNaN {A}) (ite (fp.isNaN {B}) {self.sc.nan(both)} {b}) (ite (fp.isNaN {B}) {a} {r}))"
        return self.bindf(r)

    def fclamp(self, x, lo, hi, fast):
        r = self.fminmax("min", self.fminmax("max", x, lo, fast), hi, fast)
        if fast:
            bad = f"(fp.gt {fpv(lo)} {fpv(hi)})"
            r = f"(ite {bad} {self.sc.unknown(32, bad)} {r})"
        return E(self.bindf(r), 32, False, "f")

    def f2i(self, x, signed):
        X = fpv(x)
        if signed:
            ok = f"(and (fp.leq ((_ to_fp 8 24) #xcf000000) {X}) (fp.lt {X} ((_ to_fp 8 24) #x4f000000)))"
            core = f"((_ fp.to_sbv 32) RTZ {X})"
        else:
            ok = f"(and (fp.gt {X} ((_ to_fp 8 24) #xbf800000)) (fp.lt {X} ((_ to_fp 8 24) #x4f800000)))"
            core = f"((_ fp.to_ubv 32) RTZ {X})"
        return E(f"(ite {ok} {core} {self.sc.unknown(32, f'(not {ok})')})", 32, signed, "bv")

    def i2f(self, e):
        if e.k == "bool":
            return E(f"(ite {e.t} {bv(F_ONE, 32)} {bv(F_ZERO, 32)})", 32, False, "f")
        if e.k == "f":
            return e
        if e.w != 32:
            raise OutOfScope("float() of non-32-bit integer")
        x = self.bindf(e.t)
        uf = "i2f" if e.s else "u2f"
        self.q.fn(uf, f"({BV32}) {BV32}")
        if e.s:
            small = f"(and (bvsle #xff000000 {x}) (bvsle {x} #x01000000))"
            val = f"((_ to_fp 8 24) RNE {x})"
        else:
            small = f"(bvule {x} #x01000000)"
            val = f"((_ to_fp_unsigned 8 24) RNE {x})"
        r = self.sc.aux()
        self.sc.avoid(f"(not {small})")
        self.q.assert_(f"(=> {small} (= {fpv(r)} {val}))")
        return E(self.bindf(f"(ite {small} {r} ({uf} {x}))"), 32, False, "f")

    # -- grammar
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
        if a.k == "f" or b.k == "f":
            return E(f"(ite {cond} {self.as_float(a)} {self.as_float(b)})", 32, False, "f")
        if a.k == "bool" and b.k == "bool":
            return E(f"(ite {cond} {a.t} {b.t})", 1, False, "bool")
        w = max(a.w if a.k != "bool" else 32, b.w if b.k != "bool" else 32)
        sa, sb = to_width(a, w, a.s), to_width(b, w, b.s)
        return E(f"(ite {cond} {sa.t} {sb.t})", w, a.s and b.s, "bv")

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
        sc = self.sc
        if op in ("&&", "||"):
            return E(f"({'and' if op == '&&' else 'or'} {to_bool(a)} {to_bool(b)})", 1, False, "bool")
        if a.k == "f" or b.k == "f":
            if op not in ("==", "!=", "<", ">", "<=", ">="):
                raise OutOfScope("raw float arithmetic operator")
            A, B = fpv(self.flushin(self.as_float(a))), fpv(self.flushin(self.as_float(b)))
            if op == "!=":
                return E(f"(not (fp.eq {A} {B}))", 1, False, "bool")
            core = {"==": "fp.eq", "<": "fp.lt", ">": "fp.gt", "<=": "fp.leq", ">=": "fp.geq"}[op]
            return E(f"({core} {A} {B})", 1, False, "bool")
        if a.k == "bool" and b.k == "bool" and op in ("==", "!=", "&", "|", "^"):
            core = {"==": "=", "!=": "distinct", "&": "and", "|": "or", "^": "xor"}[op]
            return E(f"({core} {a.t} {b.t})", 1, False, "bool")
        w = max(1 if a.k == "bool" else a.w, 1 if b.k == "bool" else b.w, 32)
        if op in ("<<", ">>"):
            w = a.w if a.k != "bool" else 32
            x, n = to_width(a, w, a.s), to_width(b, w, False)
            core = f"(bvshl {x.t} {n.t})" if op == "<<" else f"({'bvashr' if a.s else 'bvlshr'} {x.t} {n.t})"
            okn = f"(bvult {n.t} {bv(w, w)})"
            return E(f"(ite {okn} {core} {sc.unknown(w, f'(not {okn})')})", w, a.s, "bv")
        signed = (a.s if a.k != "bool" else False) and (b.s if b.k != "bool" else False)
        x, y = to_width(a, w, signed), to_width(b, w, signed)
        if op in ("==", "!="):
            return E(f"({'=' if op == '==' else 'distinct'} {x.t} {y.t})", 1, False, "bool")
        if op in ("<", ">", "<=", ">="):
            core = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge"}[op]
            return E(f"(bv{'s' if signed else 'u'}{core} {x.t} {y.t})", 1, False, "bool")
        if op in ("/", "%"):
            core = {("/", False): "bvudiv", ("/", True): "bvsdiv", ("%", False): "bvurem", ("%", True): "bvsrem"}[
                (op, signed)]
            z = f"(= {y.t} {bv(0, w)})"
            if signed:      # INT_MIN / -1 overflows: undefined
                z = f"(or {z} (and (= {x.t} {bv(1 << (w - 1), w)}) (= {y.t} {bv(-1, w)})))"
            return E(f"(ite {z} {sc.unknown(w, z)} ({core} {x.t} {y.t}))", w, signed, "bv")
        core = {"+": "bvadd", "-": "bvsub", "*": "bvmul", "&": "bvand", "|": "bvor", "^": "bvxor"}[op]
        r = f"({core} {x.t} {y.t})"
        if signed and op in ("+", "-", "*"):      # signed overflow is undefined behaviour
            xt, yt = self.q.bind(x.t, f"(_ BitVec {w})"), self.q.bind(y.t, f"(_ BitVec {w})")
            if op == "*":
                wide = f"(bvmul ((_ sign_extend {w}) {xt}) ((_ sign_extend {w}) {yt}))"
                ovf = f"(not (= {wide} ((_ sign_extend {w}) (bvmul {xt} {yt}))))"
            else:
                rt = f"({core} {xt} {yt})"
                sx, sy, sr = (f"((_ extract {w - 1} {w - 1}) {v})" for v in (xt, yt, rt))
                same = "=" if op == "+" else "distinct"
                ovf = f"(and ({same} {sx} {sy}) (distinct {sr} {sx}))"
            r = f"(ite {ovf} {sc.unknown(w, ovf)} ({core} {xt} {yt}))"
        return E(r, w, signed, "bv")

    def unary(self):
        t = self.peek()
        if t == "!":
            self.take()
            return E(f"(not {to_bool(self.unary())})", 1, False, "bool")
        if t in ("~", "-"):
            self.take()
            e = self.unary()
            if e.k == "f":
                if t == "~":
                    raise OutOfScope("~ on float")
                return self.fneg(e.t)
            e = to_width(e, 32 if e.k == "bool" else e.w, e.s)
            return E(f"({'bvnot' if t == '~' else 'bvneg'} {e.t})", e.w, e.s, "bv")
        return self.primary()

    def leaf(self, display, kind, w, s, ns):
        q = self.q
        san = re.sub(r"[^A-Za-z0-9_]", "_", display)
        key = f"v_{san}_{w if kind != 'bool' else 'b'}" + (f"_at{ns}" if ns else "")
        sort = "Bool" if kind == "bool" else f"(_ BitVec {w})"
        q.declare(key, sort)
        q.leaves[display + (f"@{ns}" if ns else "")] = (key, sort)
        return E(key, w, s, kind)

    def typed_leaf(self, display, ty, ns):
        if ty in (None, "ambig", "ptr"):
            raise OutOfScope(f"unknown type of {display}")
        base, n = ty
        if n != 1:
            raise OutOfScope(f"vector value {display}")
        if base == "float":
            return self.leaf(display, "f", 32, False, ns)
        if base == "bool":
            return self.leaf(display, "bool", 1, False, ns)
        if base in INT_TYPES:
            return self.leaf(display, "bv", *INT_TYPES[base], ns)
        raise OutOfScope(f"{base} value {display}")

    def primary(self):
        q, sc, F = self.q, self.sc, self.q.F
        t = self.take()
        if t is None:
            raise OutOfScope("unexpected end")
        if t == "(":
            e = self.expr()
            self.take(")")
            if self.peek() in (".", "["):
                raise OutOfScope("member of parenthesised expression")
            return e
        if re.fullmatch(r"\d+(?:ul|u)?", t):
            n = int(re.match(r"\d+", t).group())
            uns = t.endswith("u") or t.endswith("ul")
            w = 64 if t.endswith("ul") or n >= (1 << 32 if uns else 1 << 31) else 32
            if n >= 1 << (64 if uns else 63):
                raise OutOfScope("integer literal too large")
            return E(bv(n, w), w, not uns, "bv")
        if t[0].isdigit():
            if t.endswith("h"):
                raise OutOfScope("half literal")
            return E(bv(f32_bits(t), 32), 32, False, "f")
        if t in ("true", "false"):
            return E(t, 1, False, "bool")
        if t.startswith("as_type<"):
            ty = re.search(r"<\s*(\w+)\s*>", t).group(1)
            (x,) = self.args()
            if ty == "float":
                if x.k == "bool" or x.w != 32:
                    raise OutOfScope("as_type<float> width")
                return E(x.t, 32, False, "f")
            if ty in ("uint", "int"):
                if x.k == "bool" or x.w != 32:
                    raise OutOfScope("as_type width change")
                return E(x.t, 32, ty == "int", "bv")
            if ty in ("ulong", "long") and x.k == "bv" and x.w == 64:
                return E(x.t, 64, ty == "long", "bv")
            raise OutOfScope(f"as_type<{ty}>")
        if t == "float2" and self.peek() == "(" and self.peek(1) == "as_type<half2>":
            return self.half_unpack()
        if self.peek() == "(":
            return self.call(t)
        return self.path(t)

    def half_unpack(self):
        self.take("(")
        self.take("as_type<half2>")
        (x,) = self.args()
        self.take(")")
        self.take(".")
        comp = self.take()
        if x.k != "bv" or x.w != 32 or comp not in ("x", "y"):
            raise OutOfScope("half2 unpack form")
        xb = self.q.bind(x.t, BV32)
        h = self.q.bind(f"((_ extract {15 if comp == 'x' else 31} {0 if comp == 'x' else 16}) {xb})", "(_ BitVec 16)")
        sub = f"(and (= ((_ extract 14 10) {h}) #b00000) (not (= ((_ extract 9 0) {h}) #b0000000000)))"
        h = self.q.bind(f"(ite (and {self.sc.choice()} {sub}) (bvand {h} #x8000) {h})", "(_ BitVec 16)")
        return E(self.fp_bits(f"((_ to_fp 8 24) RNE ((_ to_fp 5 11) {h}))"), 32, False, "f")

    def call(self, name):
        sc = self.sc
        if name in FMA_FORMS:
            n, form = FMA_FORMS[name]
            a = self.args()
            if len(a) != n:
                raise OutOfScope(f"{name} arity")
            return self.fma(*form([self.as_float(x) for x in a]))
        if name in INT_TYPES or name in ("bool", "float"):
            (x,) = self.args()
            if name == "float":
                return self.i2f(x)
            if name == "bool":
                return E(to_bool(x), 1, False, "bool")
            w, s = INT_TYPES[name]
            if x.k == "f":
                if w != 32:
                    raise OutOfScope(f"{name}(float)")
                return self.f2i(x.t, s)
            src_signed = x.s and x.k != "bool"
            if x.k == "bool" or x.w < w:
                e = to_width(E(x.t, x.w, src_signed, x.k), w, src_signed)
            else:
                e = to_width(x, w, s)
            return E(e.t, w, s, "bv")
        a = self.args()
        isf = any(x.k == "f" for x in a)
        base = name.split("::")[-1]
        fast = not name.startswith("precise::")
        if name in ("isnan", "isinf") and len(a) == 1 and a[0].k == "f":
            return E(f"(fp.is{'NaN' if name == 'isnan' else 'Infinite'} {fpv(a[0].t)})", 1, False, "bool")
        if isf and base in ("max", "min", "fmax", "fmin") and len(a) == 2:
            return E(self.fminmax(base[-3:], self.as_float(a[0]), self.as_float(a[1]), fast), 32, False, "f")
        if isf and base == "clamp" and len(a) == 3:
            return self.fclamp(*[self.as_float(x) for x in a], fast)
        if isf and base == "saturate" and len(a) == 1:
            return self.fclamp(a[0].t, bv(F_ZERO, 32), bv(F_ONE, 32), True)
        if name == "abs" and len(a) == 1:
            x = a[0]
            if x.k == "f":
                return self.fabs(x.t)
            if x.k == "bv" and not x.s:
                return x
            if x.k == "bv":
                t = self.q.bind(x.t, f"(_ BitVec {x.w})")
                mn = bv(1 << (x.w - 1), x.w)
                return E(f"(ite (= {t} {mn}) {sc.unknown(x.w, f'(= {t} {mn})')} (ite (bvslt {t} {bv(0, x.w)}) (bvneg {t}) {t}))",
                         x.w, True, "bv")
        if isf and base in ROUNDING and len(a) == 1:
            x = self.flushin(a[0].t)
            return E(self.fp_bits(f"(fp.roundToIntegral {ROUNDING[base]} {fpv(x)})"), 32, False, "f")
        if isf and base in OPAQUE_FLOAT_FNS:
            fn = f"uf_{base}_{len(a)}"
            self.q.fn(fn, f"({' '.join([BV32] * len(a))}) {BV32}")
            self.q.opaque.add(base)
            sc.avoid("true")
            return E(f"({fn} {' '.join(self.as_float(x) for x in a)})", 32, False, "f")
        if not isf and name in ("min", "max", "clamp") and all(x.k == "bv" for x in a):
            s = all(x.s for x in a)
            w = max(x.w for x in a)
            xs = [to_width(x, w, s).t for x in a]
            lt = lambda p, r: f"(bv{'s' if s else 'u'}lt {p} {r})"
            mx = lambda p, r: f"(ite {lt(p, r)} {r} {p})"
            mi = lambda p, r: f"(ite {lt(p, r)} {p} {r})"
            if name == "min" and len(xs) == 2:
                return E(mi(*xs), w, s, "bv")
            if name == "max" and len(xs) == 2:
                return E(mx(*xs), w, s, "bv")
            if name == "clamp" and len(xs) == 3:
                v = mi(mx(xs[0], xs[1]), xs[2])
                return E(f"(ite {lt(xs[2], xs[1])} {sc.unknown(w, lt(xs[2], xs[1]))} {v})", w, s, "bv")
        if name == "extract_bits" and len(a) == 3 and a[0].k == "bv":
            x, off, n = a
            w = x.w
            off, n = to_width(off, w, False), to_width(n, w, False)
            core = f"(bvand (bvlshr {x.t} {off.t}) (bvsub (bvshl {bv(1, w)} {n.t}) {bv(1, w)}))"
            if x.s:      # signed extract sign-extends the field
                sh = f"(bvsub {bv(w, w)} {n.t})"
                core = f"(bvashr (bvshl {core} {sh}) {sh})"
            body = f"(ite (= {n.t} {bv(w, w)}) {x.t} {core})"
            okf = f"(bvule (bvadd {off.t} {n.t}) {bv(w, w)})"
            return E(f"(ite {okf} {body} {sc.unknown(w, f'(not {okf})')})", w, x.s, "bv")
        if name == "insert_bits" and len(a) == 4 and a[0].k == "bv":
            base_, ins, off, n = a
            w = base_.w
            ins, off, n = to_width(ins, w, False), to_width(off, w, False), to_width(n, w, False)
            mask = f"(bvshl (bvsub (bvshl {bv(1, w)} {n.t}) {bv(1, w)}) {off.t})"
            full = f"(ite (= {n.t} {bv(w, w)}) {ins.t} (bvor (bvand {base_.t} (bvnot {mask})) " \
                   f"(bvand (bvshl {ins.t} {off.t}) {mask})))"
            okf = f"(bvule (bvadd {off.t} {n.t}) {bv(w, w)})"
            return E(f"(ite {okf} {full} {sc.unknown(w, f'(not {okf})')})", w, base_.s, "bv")
        if name == "reverse_bits" and len(a) == 1 and a[0].k == "bv":
            x = self.q.bind(a[0].t, f"(_ BitVec {a[0].w})")
            parts = " ".join(f"((_ extract {i} {i}) {x})" for i in range(a[0].w))
            return E(f"(concat {parts})", a[0].w, a[0].s, "bv")
        raise OutOfScope(f"call {name}")

    def path(self, root):
        q, F = self.q, self.q.F
        ns = self.sc.tag if self.sc.is_def else None
        if not re.fullmatch(r"\w+", root) or root in ("if", "while", "for"):
            raise OutOfScope(f"token {root!r}")
        if self.peek() not in (".", "["):
            if not re.fullmatch(r"_\d+|gl_\w+", root):
                raise OutOfScope(f"token {root!r}")
            if root in F.defs and self.depth < 12:
                e = self.inline(root)
                if e is not None:
                    return e
            return self.typed_leaf(root, F.types.get(root), None if root in F.ssa else ns)
        if self.peek() == "[" and re.fullmatch(r"_\d+", root) and (root in F.consts or root in F.arrays):
            self.take("[")
            idx = self.expr()
            self.take("]")
            if self.peek() in (".", "["):
                raise OutOfScope("nested array access")
            if idx.k != "bv":
                raise OutOfScope("non-integer index")
            i32 = to_width(idx, 32, False)
            if root in F.consts:
                vals = F.consts[root]
                k = const_value(i32.t)
                if k is not None and k < len(vals):
                    return E(bv(vals[k], 32), 32, False, "bv")
                e = self.sc.unknown(32, f"(bvuge {i32.t} {bv(len(vals), 32)})")
                for j in range(len(vals) - 1, -1, -1):
                    e = f"(ite (= {i32.t} {bv(j, 32)}) {bv(vals[j], 32)} {e})"
                return E(e, 32, False, "bv")
            ty = F.arrays[root]
            if ty == "ambig" or ty[0] not in INT_TYPES:
                raise OutOfScope(f"array {root} type")
            w, s = INT_TYPES[ty[0]]
            arr = f"A{root}" + (f"_at{ns}" if ns else "")
            q.declare(arr, f"(Array {BV32} (_ BitVec {w}))")
            k = const_value(i32.t)
            rd = f"(select {arr} {i32.t})"
            if k is not None:
                if k >= ty[1]:
                    return E(self.sc.unknown(w, "true"), w, s, "bv")
                term = rd
            else:
                inb = f"(bvult {i32.t} {bv(ty[1], 32)})"
                term = f"(ite {inb} {rd} {self.sc.unknown(w, f'(not {inb})')})"
            disp = f"{root}[{k}u]" if k is not None else f"{root}[...]"
            q.leaves[disp + (f"@{ns}" if ns else "")] = (term, f"(_ BitVec {w})")
            return E(term, w, s, "bv")
        # field / component path with constant indices
        text, ty = root, F.types.get(root)
        while self.peek() in (".", "["):
            if self.take() == ".":
                fld = self.take()
                if fld in ("x", "y", "z", "w") and isinstance(ty, tuple) and ty[1] > 1:
                    ty = (ty[0], 1)
                elif fld in F.fields and len(F.fields[fld]) == 1:
                    ty = next(iter(F.fields[fld]))
                else:
                    raise OutOfScope(f"member {fld}")
                text += f".{fld}"
            else:
                idx = self.expr()
                self.take("]")
                k = const_value(idx.t) if idx.k == "bv" else None
                if k is None:
                    raise OutOfScope("member read with computed index")
                text += f"[{k}]"
        return self.typed_leaf(text, ty, ns)

    def inline(self, name):
        q = self.q
        if name not in q.defs_done:
            if q.inlined >= INLINE_BUDGET:
                return None
            q.defs_done[name] = None
            q.inlined += 1
            apps = dict(q.apps)
            try:
                e = Parser(q.F.defs[name], q, Scope(q, name, True), self.depth + 1).full()
                e = E(q.bind(e.t, sort_of(e)), e.w, e.s, e.k)
            except (OutOfScope, IndexError, ValueError, RecursionError):
                q.apps = apps
                e = None
            q.defs_done[name] = e
        return q.defs_done[name]


# ---------------------------------------------------------------------------------------------- driver

def statement_text(line, col, length):
    """Expression a statement computes (harness lane-guard wrappers removed); the mutation must lie inside it."""
    m = re.match(r"^\s*if \((.*)\)\s*$", line) or \
        re.match(r"^\s*(?:const\s+)?(?:\w+\s+)?[\w.\[\]]+\s*=\s*(.*);\s*$", line)
    if not m:
        raise OutOfScope("not an assignment or if")
    lo, hi = m.span(1)
    if not (lo <= col and col + length <= hi):
        raise OutOfScope("mutation outside the computed expression")
    for w in re.finditer(r"sbx_n?sel\(\d+, ", line):
        if w.start() <= col < w.end():
            raise OutOfScope("mutation in lane-guard id")
    return re.sub(r"sbx_n?sel\(\d+, ", "(", m.group(1))


def _build(F, ta, tb, tied, axioms, guards, noflush=False, fix=None, block=(), equal=False, clean=False):
    """SMT query for 'the two statement values differ' (or 'are equal' with equal=True). fix pins free values
    (display -> SMT literal); block excludes earlier witnesses."""
    q = Query(F, tied, noflush, clean)
    a = Parser(ta, q, Scope(q, "a", False)).full()
    b = Parser(tb, q, Scope(q, "b", False)).full()
    if (a.k == "bool") != (b.k == "bool") or (a.k == "f") != (b.k == "f"):
        raise OutOfScope("type mismatch")
    if a.k in ("bool", "f"):
        ra, rb, sort = a.t, b.t, sort_of(a)
    else:
        w = max(a.w, b.w)
        ra, rb, sort = to_width(a, w, a.s).t, to_width(b, w, b.s).t, f"(_ BitVec {w})"
    lines = ["(set-logic ALL)", "(set-option :produce-models true)"]
    lines += list(q.decls.values())
    lines += [f"(assert {t})" for t in q.asserts]
    for app, g, R in q.apps.values():
        if axioms:
            lines.append(f"(assert (=> {g} (= ((_ to_fp 8 24) {app}) {R})))")
        if guards:
            lines.append(f"(assert {g})")
    names, by_disp = {}, {}
    for i, (disp, (term, lsort)) in enumerate(q.leaves.items()):
        nm = f"w_{i}"
        lines.append(f"(define-fun {nm} () {lsort} {term})")
        names[nm], by_disp[disp] = disp, nm
    for disp, lit in (fix or {}).items():
        if disp in by_disp:
            lines.append(f"(assert (= {by_disp[disp]} {lit}))")
    for prev in block:
        eqs = [f"(= {by_disp[d]} {lit})" for d, lit in prev.items() if d in by_disp]
        if eqs:
            lines.append(f"(assert (not (and true {' '.join(eqs)})))")
    lines.append(f"(define-fun r_ref () {sort} {ra})")
    lines.append(f"(define-fun r_mut () {sort} {rb})")
    names["r_ref"], names["r_mut"] = "=ref", "=mut"
    lines.append(f"(assert ({'=' if equal else 'distinct'} r_ref r_mut))")
    lines.append("(check-sat)")
    lines.append(f"(get-value ({' '.join(names)}))")
    return "\n".join(lines) + "\n", names, q


def _run(smt, timeout_s):
    try:
        r = subprocess.run(["z3", "-in", f"-T:{timeout_s}"], input=smt, capture_output=True, text=True,
                           timeout=timeout_s + 10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error {type(e).__name__}", ""
    out = r.stdout.strip()
    first = out.split("\n", 1)[0].strip() if out else (r.stderr.strip()[:80] or "no output")
    return first, out


def _model(out, names):
    """(display -> SMT literal) for free values and display -> python value incl. '=ref'/'=mut'."""
    lits, vals = {}, {}
    for nm, val in re.findall(r"\((w_\d+|r_ref|r_mut) (#x[0-9a-f]+|#b[01]+|true|false)\)", out):
        disp = names[nm]
        if not disp.startswith("="):
            lits[disp] = val
        vals[disp] = (val == "true") if val in ("true", "false") else int(val[2:], 16 if val.startswith("#x") else 2)
    return lits, vals


def prove_equivalent(msl, m, timeout_s=20, attempts=3):
    """(True | False | None, reason, witness-or-None) for one mutant record.

    False: the witness inputs make the two values differ under every hardware choice (checked separately).
    None with a witness: the values differ there only under some hardware choices or through an undefined value."""
    if m.get("kind") not in ("int_const", "float_op"):
        return None, "unsupported mutant kind", None
    lines = msl.split("\n")
    if not (1 <= m["line"] <= len(lines)):
        return None, "line out of range", None
    line = lines[m["line"] - 1]
    col = m["col"]
    if line[col:col + len(m["orig"])] != m["orig"]:
        return None, "site text mismatch", None
    mut_line = line[:col] + m["rep"] + line[col + len(m["orig"]):]
    F = facts(msl)
    try:
        ta = statement_text(line, col, len(m["orig"]))
        tb = statement_text(mut_line, col, len(m["rep"]))
        if ta == tb:
            return None, "out of scope: mutation not visible in expression", None
        smt, names, q = _build(F, ta, tb, tied=False, axioms=False, guards=False)
    except OutOfScope as e:
        return None, f"out of scope: {e}", None
    except (IndexError, ValueError, RecursionError, KeyError) as e:
        return None, f"parse failed: {type(e).__name__}", None
    res, _ = _run(smt, timeout_s)
    if res == "unsat":
        return True, "z3: statement bits identical for every input and every hardware choice", None
    if res != "sat":
        return None, f"z3 strict: {res}", None
    has_apps, opaque = bool(q.apps), sorted(q.opaque)
    note = f" (treats {', '.join(opaque)} as uninterpreted)" if opaque else ""
    weak, tried = None, []
    for noflush, clean in [(True, True)] * (attempts - 2) + [(True, False), (False, False)]:
        smt, names, _ = _build(F, ta, tb, tied=True, axioms=True, guards=True, noflush=noflush, block=tried,
                               clean=clean)
        res, out = _run(smt, timeout_s)
        if res != "sat":
            continue
        lits, vals = _model(out, names)
        tried.append(lits)
        weak = weak or vals
        smt, _, _ = _build(F, ta, tb, tied=False, axioms=True, guards=False, fix=lits, equal=True)
        if _run(smt, timeout_s)[0] == "unsat":
            return False, "z3: witness differs under every hardware choice" + note, vals
    if weak is not None:
        return None, "differs at witness only under some hardware choices or undefined values" + note, weak
    if has_apps:
        smt, _, _ = _build(F, ta, tb, tied=False, axioms=True, guards=False)
        if _run(smt, timeout_s)[0] == "unsat":
            return True, "z3: identical for every input and choice, given IEEE RNE fma on the normal range", None
        smt, _, _ = _build(F, ta, tb, tied=True, axioms=True, guards=False)
        if _run(smt, timeout_s)[0] == "sat":
            return None, "differs only where arithmetic meets denormal/NaN/tiny values (unmodelled)", None
    return None, "differs only under a hardware-dependent choice (denormal flush, NaN payload, undefined result)", None


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: z3_fp.py <ref.metal> '<mutant record json>'")
    res = prove_equivalent(open(sys.argv[1]).read(), json.loads(sys.argv[2]))
    print(json.dumps({"ok": res[0], "reason": res[1], "witness": res[2]}))
