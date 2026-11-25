# z3_fp: float-aware equivalence prover for mutants

`tools/harness/z3_fp.py` extends `z3_equiv.py` (which it does not import or change) to statements that mix integer bit
operations with float values. Same approach: the mutated statement's value is built for the reference and the mutant
as SMT-LIB terms; the Homebrew `z3` binary (`-T:20`, logic `ALL`) is asked whether the two can differ.

```python
import z3_fp
ok, reason, witness = z3_fp.prove_equivalent(msl, mutant_record)
# ok True : unsat, equal for every input and every hardware choice
# ok False: witness inputs make the values differ under every hardware choice (checked by a second query)
# ok None : out of scope, timeout or a difference that needs a particular hardware choice (witness may still be set)
# witness : {"_294[3u]": 0x..., "_1078[28u]@_3197": 0x..., ..., "=ref": ..., "=mut": ...}  (32-bit words or bools)
```

Witness keys are the source text of each free value. `text@_N` means the value was read while computing the inlined
SSA definition `_N` (register reads there are kept apart from later reads of the same register in the statement).

## Float model (matches `MTLCompileOptions.mathMode = .safe`)

- Every float is a 32-bit bit-vector. `as_type<float/uint/int>` is identity on the bits. Results are compared as bits
  so -0 and +0 differ and NaN payloads count.
- Comparisons, `abs`, unary minus, `isnan`/`isinf`, `max`/`min`/`clamp`, `floor`/`ceil`/`trunc`/`rint`/`round`,
  int<->float conversions and `float2(as_type<half2>(x)).x/.y` use FloatingPoint theory on `((_ to_fp 8 24) bits)`.
  NaN and Inf are never assumed away.
- Arithmetic goes through the helpers' real bodies onto one uninterpreted function `fma32(a, b, c)` over raw bits:
  `spvFMul(l,r)=fma(l,r,+0)`, `spvFAdd`/`spvMutFAdd(l,r)=fma(1,l,r)`, `spvFSub`/`spvMutFSub(l,r)=fma(-1,r,l)`,
  `spvFmaSwap(a,b,c)=fma(a,c,b)`. Same call on the same bits gives the same bits; nothing else is assumed for
  proofs (so `x*0` is never folded and fma stays fused). Raw float `+ - * /` operators are out of scope because the
  compiler may contract them.
- Witness search (never proofs) adds IEEE RNE `fp.fma` semantics, guarded: no subnormal or NaN operand and a result
  that is not NaN with magnitude above the smallest normal (or an exact zero from a zero product plus zero).
  A third query may prove with these guarded semantics; that path is labelled "given IEEE RNE fma" and fired 0 times.

## Denormals and undefined results

Unsure whether Apple GPU fp32 flushes under `.safe`, the prover allows both: every float operand of a comparison,
abs, negation, max/min/clamp, rounding or half unpack may be read as a signed zero when subnormal. Each such place
gets a fresh Bool per side, so a proof holds only if it holds whichever way each side flushes. Pure bit moves
(`as_type`, ternary selects) never flush. Fresh per-side unknowns also stand for: NaN payloads of abs/neg/max/min
and conversions, fast:: or unqualified float max/min/clamp with a NaN operand, +-0 ties in max/min, float->int out
of range or NaN, int->float above 2^24 (uninterpreted), shift by >= width, division by zero, INT_MIN/-1, signed
`+ - *` overflow, out-of-range bit fields and out-of-bounds register-array reads.

Values computed outside the statement are shared free variables (as in z3_equiv). SSA definitions the statement
reads are inlined (up to 40, depth 12); their own flush and undefined choices are shared by both sides because that
code is unchanged.

## Validation (run_150_v3, 723 rows, sequential, nice)

Script: `scratchpad/agent_fp/validate.py` (results in `results.jsonl` beside it; `--resume` appends; `--summary`).
Synthetic negatives and positives: `scratchpad/agent_fp/synthetic.py` (14 must-not-prove cases: denormal `== 0.0`,
flush threshold, sign mask, -0/+0, NaN payload, denormal constant, fmul vs fadd, fma swap, max +-0 tie, fast::clamp
on NaN, isnan, big int->float, float->int range, shift past width; all pass, 6 positives proved).

Soundness: 382 killed mutants checked (359 run_set `killed` in run_150_v3.jsonl, all present in the
`mutants_pinned.json` lists, plus 23 `killed_by_hunt`; the resolved file has no `killed_by_gate` rows yet).
Proved equivalent: **0**. 365 got a robust witness.

Power on 279 survivors (215 unobservable, 64 unresolved):

| outcome | count |
|---|---|
| proved equivalent | 11 (all int_const) |
| robust witness (False) | 226 (149 int_const, 77 float_op) |
| witness that needs a hardware choice or undefined value | 4 |
| differs only through unmodelled denormal/NaN arithmetic | 26 |
| differs only under a side-dependent choice | 1 |
| out of scope | 11 (raw `*` 3, store-target form 2, spvQuadSwap, spvSubgroupShuffle, spvMulExtended, float3, vector value, raw float op) |

No z3 timeouts. z3_equiv's 8 proofs are all re-proved. Of the 62 rows already marked equivalent, 26 are proved here;
the rest were proved by identical IR, which sees downstream context a statement-level prover cannot.

Most survivors differ at statement level and survive because the difference is not observed downstream. The 226
witnesses are concrete inputs (register words) for steering. Example: `8388608u -> 8388609u` in the flush idiom differs
at x = 0x00800000 (the smallest normal is kept by the reference but zeroed by the mutant).

## Three proofs checked by hand

1. `_294[3u] = as_type<uint>((abs(_1348) < as_type<float>(8388608u)) ? as_type<float>(as_type<uint>(_1348) & 2147483648u) : _1348);`
   mask `2147483648 -> 2147483649`. `_1348` is the legacy multiply `(_1329 == 0.0 || _1337 == 0.0) ? 0.0 : flush(_31)`.
   Its value is +0, a signed zero from the inner flush or a value with |x| >= 2^-126 (or NaN/Inf): never subnormal,
   whichever way the comparisons flush. So whenever the outer test is true `_1348` is +-0 with bit 0 clear and
   `& 0x80000001` equals `& 0x80000000`; otherwise both return `_1348`. Genuinely equivalent (the outer flush is
   idempotent). Nine more proofs are this same pattern.
2. `float y = (abs(x) < as_type<float>(8388608u)) ? as_type<float>(K & 2147483648u) : x;`
   threshold `8388608 -> 8388609` with `x = as_type<float>(K)` for a constant K (about 0.99996). Both thresholds are
   tiny so the test is false on both sides and the value is `x`. Equivalent.
3. `_249[5u] = as_type<uint>(fma(as_type<float>(_249[1u]), as_type<float>(1056964608u), as_type<float>(1056964608u)));`
   `fma( -> spvFmaSwap(` swaps the last two arguments, which are both 0.5. Same call on the same bits. Equivalent.

## Limitations

- Statement-local: a difference that downstream code masks is not proved (IR comparison covers some of that).
- Assumes identical fma calls on identical bits give identical bits in both programs. Also assumes integer bit
  idioms stay integer operations (the compiler does not turn them into flushing float ops).
- Shared free values assume code outside the statement computes the same values in both programs.
- Out of scope: raw float operators, vectors, subgroup/quad ops, user function calls, half arithmetic, stores whose
  target index is mutated, computed indices into non-array storage.
- Transcendentals (`fract`, `sqrt`...) are uninterpreted; witnesses through them are flagged.
- About 6 s per mutant on average (max 100 s on fma-heavy witness queries), one z3 at a time.
