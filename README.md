# rdna2-metal-lab

Tools for running a console game's RDNA2 GPU shaders natively on Apple Silicon through Metal plus tools for proving the
translated shaders are right. No game files are in this repository: every shader, input and result is generated
locally from your own copy.

## Why

Projects that run the Xbox Series build of this game on other platforms are held up on the GPU side. Translating the
shaders is the easy part; knowing that the translation is right is not. This repository is the checking half:
a differential Metal harness, input search and mutation testing that give each translated shader a measured score.

Chain: RDNA2 (gfx1013) machine code -> Palmetto `rdna2spirv.py` (link: TBD) -> SPIR-V -> SPIRV-Cross MSL
-> Metal (language 3.1, `mathMode = .safe`).

## What is here

| Path | What |
|---|---|
| `tools/harness/harness.swift` | Metal harness: runs a shader and a variant on the same inputs and compares every output (render targets, depth, writable textures, the whole memory window), with coverage, comparison logging and read tracking |
| `tools/harness/harness_tex.swift` | The same with textures and samplers decoded from each T#/S# the shader looks up |
| `tools/harness/prepare.py` | Instruments translated MSL: sandboxed addressing, coverage, lane guards, comparison logging |
| `tools/harness/fuzz_inputs.py` | Input search: coverage-guided fuzzing, input-to-state solving, a descriptor model with valid T#s, directed search on unreached blocks |
| `tools/harness/run_set.py` | Mutation testing over a set of shaders (float-op swaps, fma operand swaps, integer constant +1) |
| `tools/harness/prove_equiv.py` | Equivalence proofs with no GPU: IR identity, bit-vector SMT (`z3_equiv.py`), float-aware SMT (`z3_fp.py`), dead stores (`dead_store.py`) |
| `tools/harness/resolve_survivors.py` | Per-survivor hunts for inputs that kill a surviving mutant |
| `tools/harness/goal_report.py` | The score: kill rate over non-equivalent mutants in reached code, per sample, plus output-block coverage |
| `tools/gamedata/` | Readers for the game's archive and texture formats (no data) |
| `notes/metal_shader_abi.md` | What a runtime must provide to run the generated MSL |
| `notes/z3_fp.md` | The float-aware equivalence prover and its soundness rules |
| `notes/gamedata_recon.md` | Where shader bindings, textures and constants live in the game data |
| `notes/tools_survey.md` | Related projects read in depth |

## Setup

1. macOS on Apple Silicon with Xcode command-line tools, Python 3.12, `z3`, `spirv-cross`, `spirv-tools`.
2. Clone Palmetto into `upstream/palmetto` (pinned commit: TBD).
3. Put your own copy's `shaders.rpf` in the repository root, carve and translate the shaders, then prepare a set
   (commands: TBD).
4. Build the harness: `swiftc -O -o tools/harness/harness tools/harness/harness.swift`.
5. Scripts that need the game folder read `GAME_ROOT` (default: the current folder).

## Where help is wanted

- Surviving mutants: most change values that reach an output through code that runs and are then hidden by
  degenerate outputs, select masking, special float values or exact boundaries that random search never reaches.
- Wave64 on 32-wide Apple GPUs: cross-lane operations in vertex and pixel shaders are not exact yet.
- Vertex, mesh and geometry stages on Metal.
- Typed inputs from the shader binding tables in the game's shader archive.

## Rules

- Never commit anything made from game files: shader binaries, disassembly, translated code, inputs or results.
  `.gitignore` blocks the usual paths.
- Fast math is wrong for these shaders: always compile with `mathMode = .safe` (and `-fno-fast-math` on the CLI).
