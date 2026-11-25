"""Synthetic cases for z3_fp: negatives must never return True; positives should return True."""
import os  # noqa: E401
import json
import sys

sys.path.insert(0, os.environ.get("GAME_ROOT", ".") + "/tools/harness")
import z3_fp  # noqa: E402

HEAD = """fragment main0_out main0(main0_in in [[stage_in]])
{
    spvUnsafeArray<uint, 32> _5;
    float _10 = as_type<float>(_5[1u]);
    float _11 = as_type<float>(_5[2u]);
    uint _12 = _5[3u];
    bool _13 = _5[4u] != 0u;
"""


def case(stmt, orig, rep, kind="int_const", occ=1):
    msl = HEAD + "    " + stmt + "\n}\n"
    line = msl.split("\n").index("    " + stmt) + 1
    src = "    " + stmt
    col = -1
    for _ in range(occ):
        col = src.index(orig, col + 1)
    return msl, {"kind": kind, "line": line, "col": col, "orig": orig, "rep": rep, "src": stmt}


# (name, stmt, orig, rep, kind, occurrence, expectation) ; expectation 'neq' = must not be True, 'eq' = should be True
CASES = [
    ("denormal == 0.0 select", "_5[6u] = (_10 == 0.0) ? 0u : 1u;", "0u", "2u", "int_const", 1, "neq"),
    ("flush threshold +1 ulp", "float _20 = (abs(_10) < as_type<float>(8388608u)) ? as_type<float>(as_type<uint>(_10) & 2147483648u) : _10;", "8388608", "8388609", "int_const", 1, "neq"),
    ("flush sign mask +1", "float _20 = (abs(_10) < as_type<float>(8388608u)) ? as_type<float>(as_type<uint>(_10) & 2147483648u) : _10;", "2147483648", "2147483649", "int_const", 1, "neq"),
    ("-0 vs +0 via ternary", "_5[6u] = as_type<uint>((_10 == 0.0) ? _10 : as_type<float>(0u));", "0u", "2147483648u", "int_const", 1, "neq"),
    ("NaN payload via abs", "_5[6u] = as_type<uint>(abs(_10)) & 4194303u;", "4194303", "4194304", "int_const", 1, "neq"),
    ("denormal constant flush", "float _20 = as_type<float>(0u);", "0u", "1u", "int_const", 1, "neq"),
    ("fmul vs fadd", "_5[6u] = as_type<uint>(spvFMul(_10, _11));", "spvFMul(", "spvMutFAdd(", "float_op", 1, "neq"),
    ("fma swap distinct", "_5[6u] = as_type<uint>(fma(_10, _11, as_type<float>(1065353216u)));", "fma(", "spvFmaSwap(", "float_op", 1, "neq"),
    ("max +0/-0 tie", "_5[6u] = as_type<uint>(precise::max(_10, as_type<float>(0u)));", "0u", "2147483648u", "int_const", 1, "neq"),
    ("fast clamp NaN", "_5[6u] = as_type<uint>(fast::clamp(_10, 0.0, 1.0)) & 0u;", "0u", "1u", "int_const", 1, "neq"),
    ("isnan with mask", "uint _20 = isnan(as_type<float>(_12)) ? 0u : _12;", "0u", "1u", "int_const", 1, "neq"),
    ("int to float big", "float _20 = float(_12 | 16777216u);", "16777216", "16777217", "int_const", 1, "neq"),
    ("float->int range", "_5[6u] = uint(as_type<float>(_12 & 2147483647u));", "2147483647", "2147483648", "int_const", 1, "neq"),
    ("shift past width", "_5[6u] = _12 << 31u;", "31", "32", "int_const", 1, "neq"),
    # positives
    ("compare on masked bits, unreachable boundary", "_5[6u] = ((_12 & 2147483647u) > 2139095040u) ? 1u : (((_12 & 2147483647u) > 2139095041u) ? 2u : 1u);", "2139095041", "2139095042", "int_const", 1, "eq"),
    ("fma swap equal args", "_5[6u] = as_type<uint>(fma(_10, _11, _11));", "fma(", "spvFmaSwap(", "float_op", 1, "eq"),
    ("isnan then bits never NaN-dependent", "_5[6u] = isnan(_10) ? 0u : (isnan(_10) ? 7u : 1u);", "7u", "8u", "int_const", 1, "eq"),
    ("abs < normal: compare to 0 under flush robust", "_5[6u] = (abs(_10) < as_type<float>(8388608u)) ? 0u : ((abs(_10) < as_type<float>(8388608u)) ? 3u : 1u);", "3u", "4u", "int_const", 1, "eq"),
    ("masked high bits", "_5[6u] = (as_type<uint>(_10) >> 23u) & 255u & 255u;", "255u", "511u", "int_const", 2, "eq"),
    ("float compare unaffected", "_5[6u] = (as_type<float>(_12 & 2147483647u) < 0.0) ? 5u : 1u;", "5u", "6u", "int_const", 1, "eq"),
]

if __name__ == "__main__":
    bad = 0
    for name, stmt, orig, rep, kind, occ, exp in CASES:
        msl, m = case(stmt, orig, rep, kind, occ)
        ok, why, wit = z3_fp.prove_equivalent(msl, m)
        flag = ""
        if exp == "neq" and ok is True:
            flag, bad = "UNSOUND", bad + 1
        if exp == "eq" and ok is not True:
            flag = "missed"
        print(f"{exp:3} {str(ok):5} {flag:8} {name}: {why} {json.dumps(wit) if wit else ''}")
    print("unsound:", bad)
