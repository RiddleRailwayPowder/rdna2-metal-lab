# Ready-made tools survey (GitHub, 2026-11-24)

Found with `gh search repos/code` and READMEs read through `gh api`. Stars and dates as of today.

## Changes the plan

| Repo | What | Why it matters for us |
|---|---|---|
| ThatRecompDev/five-star-recomp (163, C#) | RDNA -> structured HLSL for this game: all 25,908 programs compile; branches become if/else and loops (39% smaller) | Output is HLSL/DXIL: not usable (Metal only, no D3D in the path). Worth reading only for its control-flow structuring, to do the same in our MSL lifter |
| boykopovar/AnyPS5 (61k, C++) core/shader/recompiler | RDNA -> SPIR-V in milliseconds; americas-recomp (offline now) measured >= 99.6% valid SPIR-V per stage on Xbox Series shaders. AnyBox (boykopovar/AnyBox) is its Xbox layer, rebox uses it on Linux | A second independent translator on the same Metal route as Palmetto (SPIR-V -> SPIRV-Cross -> MSL) |
| mattias800/prosper tools/rdna2_ref (Python) | Independent RDNA2 CPU interpreter from AMD guide 70648: scalar ALU, control flow, EXEC, scalar memory, integer VALU; stops with `unsupported:<op>` | Partial ground-truth oracle (no float VALU or images yet); decoder cross-checked against llvm-mc |
| xcl-dev/xcl (7.5k, Rust) | README's "recorded and replayed" means in-process translation; no capture to disk, no pipeline journal (read in depth below) | Shows where a recorder hooks in; the recorder itself is still ours to write |
| ZigZagBoi1/ScarlettPort (140, Rust, on XCL) | In the open world on an RTX 3060 (Series S profile); precompiles all 26,084 programs; `XCL_FLAT=1` grey stand-in shaders | Windows side is further along than assumed: traces can come from here |

## Mac-specific

| Repo | What |
|---|---|
| ranveersoni98/Xenora (122, Rust) | Xbox Series -> Apple Silicon, gfx1030 -> MSL on Metal, no Wine/Rosetta/Vulkan. Early: synthetic shaders only, 1 KiB reference interpreter. Machine-readable RDNA2 semantics in `crates/isa/semantics/rdna2/*.toml` |
| KytyPS5/KytyPS5 (12.7k, C++) | PS5 (also RDNA2) layer, macOS via Rosetta + MoltenVK, has its own RDNA shader recompiler |
| buberlo/madeira-anyps5 | AnyPS5 titles on iPad (Wine ARM64EC + FEX + MoltenVK) |
| LibertyPrime27/d12mt | DXIL/DXBC -> MSL plus root signature -> Metal argument buffer layout (D3D-on-Metal compiler half) |
| Traverse-Research/saxaboom | Rust wrapper around Apple's metal_irconverter (DXIL -> metallib) |

## Testing tools

- google/amber: multi-API shader test framework (scripted inputs and expected outputs).
- google/graphicsfuzz: spirv-fuzz and spirv-reduce, which shrink a failing shader to a minimal case.
- booomxd/rdna2-dis: RDNA2 disassembler and CFG visualiser matching llvm-objdump.
- Rafa23xz/victory-shader-census: per-stage census of VIctory's RDNA2 -> DXIL translation.

## Index

- A curated awesome-list of every Recent Retail Game translation/recomp/format project (a Windows port
  and americas-recomp are offline since 2026-11-23).

## What this suggests

Target is Metal only: no HLSL, DXIL, Metal Shader Converter or D3D anywhere in the path.
Mutation testing (1a+) only measures how well inputs exercise the code. A second translator on the same route
(AnyPS5 SPIR-V -> SPIRV-Cross -> MSL, beside Palmetto) lets the harness run two Metal versions of each shader against
each other: disagreement flags a translator bug without traces. prosper's interpreter adds a ground truth for the
integer/scalar subset; Xenora's semantics files are data for our own MSL lifter.

## Read in depth (code, 2026-11-24)

### five-star-recomp, src/FiveStar.Shaders (commit f4388e9)
- Structurer is language-neutral and worth porting to an MSL emitter: Cooper-Harvey-Kennedy dominators
  (`Structure/DominatorTree.cs`), natural loops (`LoopForest.cs`), per-loop acyclic scope graphs joined at the
  immediate post-dominator (`ScopeGraph.cs`), multi-exit loops via a `loop_exitN` selector, small exit tails copied
  into arms (<= 64 lines), per-scope `flowN` state-machine fallback when copying blows up (> 1.5x + 256 lines or
  nesting > 120), whole-program `switch(pc)` fallback for irreducible CFGs (`RegionEmitter.cs`, `BodyWriter.cs`).
  `s_getpc/s_add/s_addc/s_setpc` long jumps rewritten to `s_branch` (`Core/LongJumpRewriter.cs`).
- Not solved there: EXEC stays an explicit per-lane `active` bool, branch conditions are wave-uniform
  (`vw_any(active)`), VALU writes are `select(active, v, old)`; no SSA, no register promotion, no uniformity analysis.
  So the two things our MSL needs most (per-thread `if` from EXEC regions, mem2reg) have to be ours.
- "39% smaller" = DXIL bytes against their own older dispatch-loop output, not text size.
- Validation: structured vs unstructured on WARP with 3 input sets, bit-exact; 415 of 1,390 sampled ran equal.

### prosper rdna2_ref (commit e35a74a)
- Python, ~2,300 lines. Scalar ALU, SOPP branches, EXEC/saveexec, SMEM loads, integer VALU/VOPC, readlane/
  writelane/mbcnt. No float VALU, no VOP3 modifiers, no buffer/global/LDS/image/export, no DPP/SDWA.
- Measured on our corpus (1,500 random shaders from carved/asm): 1 shader fully inside its subset; on average 52.3%
  of a shader's instructions are supported. Top gaps: v_mul_legacy_f32, v_fmac_f32, exp, v_fma_f32, v_add_f32.
- Not an oracle as it stands. Also: GPU approximations (v_rcp_f32, v_rsq_f32, v_exp_f32...) can't be matched bit for
  bit on a CPU without AMD's exact tables, so a CPU oracle is only exact for integer/exact float ops.

### Xenora (commit 0fdb995, single commit)
- Translator is fixture-only: code <= 1 KiB (`crates/shader/src/lib.rs:25`), fixed binding (`s[0:3]` load, `s[4:7]`
  store, one store), forward branches only, no loops, wave32 only (`ds_bpermute` doesn't decode), EXEC must be a
  compile-time constant; cross-lane compare masks done by each thread recomputing all 32 lanes, no simdgroup ops.
  Doesn't scale to our wave64, EXEC-heavy corpus.
- Worth taking as data: `crates/isa/data/rdna2.jsonl` generated from AMD's machine-readable RDNA2 ISA XML
  (`generate_rdna2_meta.py`, XML pinned by SHA-256): 4,367 variants, 1,465 base forms, 1,153 instructions, operands
  with in/out/implicit flags. No pseudocode. Semantics TOML: 151 integer forms only, small operation vocabulary.
- Worth copying into our harness: Metal math mode pinned to Safe with precise functions (`crates/metal/src/compute.rs`
  ~1176-1189); check that `MTL_SHADER_VALIDATION=1` is really active (~1236-1253); BC1-5/BC7/BC6H (single subset)
  decoders (`crates/texture/src/bc*.rs`) for when we feed real game textures.

### AnyPS5 core/shader/recompiler (commit 03064c2)
- Real independent RDNA -> SPIR-V translator: structured CFG (OpSelectionMerge/OpLoopMerge), EXEC as per-lane bool
  with `Select(exec, new, old)`, all cross-lane ops as subgroup shuffles, s_getpc/s_setpc/s_swappc resolved
  statically, software image_bvh_intersect_ray, image and global atomics; unsupported instructions throw.
- Not a plain library: `Recompile(RecompileRequest)` (`Recompiler.hpp:1046`) needs a GuestContext with user SGPRs,
  per-stage register state and guest memory regions it reads at compile time to resolve descriptors. Depends on
  `core/libs/prx/libc/include/*.hpp` (not in core/shader) and glslang/SPIRV-Headers. Full AnyPS5 build is blocked
  on arm64; core/shader itself has no x86-only code, so it needs its own CMake plus stubbed headers.
- Its ABI is far from Palmetto's: memory through a `bda_pagetable` (one entry {0, max, heapBase} reproduces heap+VA),
  per-stage typed image/sampler arrays at binding stage*65+kind with spec-constant sizes, used-only user SGPRs packed
  into 128-byte ShaderData, extra sets for big LDS and scratch. A second Metal path through it = a sizeable adapter.
- Decodes as gfx10.1/gfx1013: VOP3 0x140 = v_mad_legacy_f32. llvm-mc reads the same bytes as v_fma_legacy_f32 for
  gfx1030. Palmetto, XPortPC and our carved/asm also use gfx1013; Palmetto has hardware probes for
  v_mad_legacy_f32 (`tools/mad_denorm_probe.py`). 5,986 of our 25,734 shaders use it. All translators share this
  assumption, so cross-checking them can't catch it if it's wrong; only real Xbox hardware output can.

### XCL (commit c160d86)
- No capture to disk. Compute command lists written by console code (`XCL_CONSOLE_LISTS=1`) are translated to D3D12
  at Close in memory (`hle/d3d12x/console_list.rs`); graphics calls go straight to the PC list.
- No pipeline journal. Closest: `used-pipelines.txt` (one combined hash per pipeline, appended, survives crashes),
  `[SHADER] <stage> <fnv64>` log lines, full pipeline state only with `XCL_TRACE_FRAME` (first 400).
- Shader identity: fnv64 of the whole container (`gpu/shader.rs:141-148`); `xcl shaders <rpf> --direct --hash H`
  finds the record, so it maps to our carved names.
- Dumps: `XCL_FRAME_DUMP` (BMP), `XCL_DUMP_SHADERS` (bin/FSSM/disasm/HLSL), `XCL_DXIL_DUMP`. Nothing writes the
  memory a draw reads, its inputs or its outputs.
- Where a recorder goes: end of `bind_abi` (`bind.rs:856-903`) for pipeline, user SGPRs, argument table and draw
  args; title memory and descriptor slots at `q_execute_command_lists` (`real.rs:5382`); textures/RTs by readback
  after the fence (`present.rs:690-760` pattern). T# word 0 is overwritten with a PC heap index, so original T#s must
  be captured when views are created.
