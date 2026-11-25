# Game data recon for shader testing (2026-11-25)

Read-only survey of the newly added files: `data.rpf`, `textures.rpf`, `common.rpf`, `script.rpf`, the three `.ttpf`
pools, `Catalog000.bin` plus the parts of `shaders.rpf` we had not used. Nothing under the game folder was modified;
the exe was never run (only read as bytes for a string list). New scripts are in `tools/gamedata/` (section 7).
Evidence files (carve indexes, PNGs, DDS) were written to the session scratchpad, which is not durable: section 7
has the commands to regenerate every number below.

## Short answer

| Question | Answer |
|---|---|
| 1. List archive entries | No TOC (encrypted, 7.9-8.0 bits/byte). Entries are recovered by carving the data area: every zlib stream is found, sized and typed (index-only carve, nothing written; a 32 MiB slice takes about 1.5 s). |
| 2. Pipeline / shader metadata | Yes, in `shaders.rpf` itself. Every one of the 26,503 shader records has a binding table with cbuffer layouts (size, member offsets, types), push-constant layouts, SRV/UAV/texture registers. Names are joaat hashes: 179 of 2,656 member names resolved so far (11.7% of uses). Effects (932) pair VS/MS with PS through passes and index raster/depth/blend rows. The pipeline library (25,471 graphics + 2,693 compute pipelines) adds input layouts. No default values. Render-target formats not located. |
| 3. Textures | Yes. `textures.rpf`: 1,356 dictionaries with 3,406 named textures, decoded with our port of the 6stars detiler. `.ttpf`: 1,107 textures, decoded by Palmetto's `ttpf_mips.py` (Gfx10 swizzle + BCPack). Sampled 11, all coherent images. |
| 4. Real constants | Yes, three sources: material constant buffers in `data.rpf` drawables (77,287 values, keyed by the same hashes as the cbuffer members), timecycle (27 weather cycles x 1,142 variables x 13 hours) plus 4,593 modifiers in `common.rpf`, the visual settings table (2,617 keys: tonemap, bloom, exposure, TAA...). |
| 5. Plan | Section 6: about 3-4 days for real textures, 5-9 days for constants plus pipeline info (the cbuffer-to-harness-memory mapping is the uncertain part). |

## 1. Archives: listing entries

### Headers (first 0x40 bytes, read directly)

| Archive | Size | Data start (`+4`) | Trailer (`+12`) | u32 at `+0x24` | TOC entropy |
|---|---:|---:|---:|---:|---:|
| `data.rpf` | 481,544,304 | 0x11060 | 22,640 | 699 | 7.998 |
| `common.rpf` | 6,642,192 | 0x9580 | 14,864 | 418 | 7.995 |
| `textures.rpf` | 192,585,024 | 0x790 | 544 | 22 | 7.911 |
| `script.rpf` | 4,358,960 | 0x260 | 128 | 2 | 7.604 |
| `shaders.rpf` | 411,695,600 | 0x8640 | 160 | 3 | 7.994 |

All have magic `9FPR` (`RPF9`). The TOC between 0x40 and the data start is encrypted (Palmetto `docs/rpf.md`,
Asset-Info notes `formats/01-rpf9.md` agree; neither has a key). `+0x24` is NOT a reliable entry count: it gives 3 for
`shaders.rpf`, which fits (two SGD5 databases + the pipeline library), but 22 for `textures.rpf`, which holds 1,356
texture dictionaries. Its meaning is open. So file names, paths and exact file boundaries are not available.

### What carving gives (Palmetto `rpf9_carve.py --index-only`, full archives)

| Archive | Streams | Compressed | Inflated | Unread | Stream heads |
|---|---:|---:|---:|---:|---|
| `common.rpf` | 954 | 5.6 MB | 50.2 MB | 0.57 MB (8.6%) | 257 XML docs (plus their 64 KiB continuations), text tables |
| `textures.rpf` | 2,749 | 188.3 MB | 487.9 MB | 0.9 MB (0.5%) | 1,356 resource system pages + their graphics pages |
| `data.rpf` | 28,509 | 423.3 MB | 1,403 MB | 20.3 MB (4.2%) | 3,132 graphics pages (`42 40 00 00`), 2,281 `PSIN`, 1,263 `RBF0`, 87 `GFX*` (Scaleform), resource system pages |
| `script.rpf` | 1,057 | 2.6 MB | 6.2 MB | 0.44 MB | 1,012 `PSIN` |

Compression: zlib (`78 DA`), files split into 64 KiB inflated streams (a stream that inflates to exactly 65,536 bytes
continues in the next). Resources start at 4 KiB + 0x10 with a `0x5xxxxxxx` virtual pointer at `+8`; their graphics
data follows as `0x6xxxxxxx` physical pages. No encryption of the data area. Names come from inside the payload
(texture names in dictionaries, `name.#dr` in drawables, effect names in SGD5), not from the archive.

The numbers match Palmetto's measurements on the same package (e.g. `data.rpf` 481,544,304 bytes with 377 unread
ranges in `docs/rpf.md`), so this copy is the 1.0.2.1 build their docs describe.

### Other files

- `Catalog000.bin` (815,104 bytes): starts `MZ` / `PE` (AMD64, one section) with a payload at 7.98 bits/byte and no
  readable strings. An encrypted or compressed package catalog; nothing usable for shaders.
- 6stars "PSO metadata" is RAGE parser metadata (`PSIN` files with `LSCH` schema sections, `atHash` member names) used
  for map placements and UI styles. It is not GPU pipeline state.

## 2. Pipeline and shader metadata

There are three layers, all in `shaders.rpf`.

### 2a. Per-shader binding table (new finding, the most useful one)

Each shader record in the two SGD5 databases is
`<len><name>\0 <tag> <u32 size> <DXBC container, size bytes> <152-byte block> <u32 n> <n-byte binding table>`.
`offset` in `carved/index.json` is the record start when the length byte is printable (label of 32+ characters:
24,795 records) and record start + 1 otherwise (1,708 records), because `shader_carve.py` starts its match at a
printable byte. Example: `TemporalAA:CS_ApplyTemporalAA_b2b938f9`, record 0xf2fdbaa = index offset 254,794,666.
Join by offset (start or start + 1), never by name: carved file names replace `:` with `_`, keep a printable length
byte as first character and add `~N` for duplicates.

Binding table layout (inferred here, cross-checked below):
- header: u16 resource rows, u16 total member rows, u16 0, u16 table size;
- resource rows, 24 bytes: u8 kind, u8 ?, u16 register, u32 flags, u32 0, u32 size, u32 name hash, u32 same hash.
  Flags low byte = member count; flags high u16 = byte offset from this row to its first member row (members are
  NOT in row order: in `CS_AlphaTransmissionDenoise` the later cbuffer's member comes first);
- member rows, 24 bytes: u16 type, u16 array count, u32 byte offset, 8 zero bytes, u32 name hash, u32 0.
  The per-row counts sum to the header's member total in all 26,503 tables (the parser checks it).

Kinds checked against the container's own `PSV0` resource list for `CS_ApplyTemporalAA`: kind 0x30 = CBV (b3 with
0x1e0 bytes / 9 members, b18 with 0x1a0 bytes / 18 members), kind 0x02 = texture SRV (t128, t129, t132-t140), kind 0x05 =
structured SRV (t16, t72, t141), kind 0x1c = typed UAV (u10-u12). All 19 rows match a PSV0 entry by type and
register. PSV0's two samplers (s0, s1) are not in the table.

Kind 0xb0 is the push-constant block: its name hash 0x88850413 resolves to `PushConstants_500` (from the exe
strings), it has a size (0x10 in the Denoise sample) and member rows like a cbuffer. 18,038 programs have one
(17,218 DB0 + 820 DB1) with 42,931 members. This gives the layout of the harness's `push` dwords per shader. The
remaining kinds (0x82, 0x85, 0x07, 0x42, 0x81...) are not identified yet.

Corpus census (`sgd5_bindings.py --census`, both databases):

| | DB0 | DB1 |
|---|---:|---:|
| programs | 24,804 | 1,699 |
| programs with a cbuffer | 21,711 | 964 |
| cbuffers | 54,527 | 982 |
| programs with a push-constant block | 17,218 | 820 |
| member rows (cbuffer + push constant) | 1,023,140 | 12,230 |
| distinct resource names | 2,356 | 201 |
| distinct member names | 2,656 | 271 |

24,804 + 1,699 = 26,503: every carved shader has a table, all parse. 1,018,042 member offsets fall inside their
block's size; 17,328 (1.7%) do not (the size field means something else for some blocks, open). Member type codes:
9 = float, 10 = float2, 11 = float3 (from `emissiveMultiplier`, `tintPaletteSelector`, `specMapIntMask`), 12 on
16-byte strides (float4) and 14 on 64-byte strides (float4x4) in the TAA sample; 15 = a struct-typed cbuffer whose
single member carries the cbuffer's own hash. The rest are not mapped.

Names are joaat (lower-cased Jenkins one-at-a-time): `gViewInverse`, `gViewProj`, `gPrevViewProj` match. Resolution
by dictionary source (DB0):

| Dictionary | Resource names named | Member names named (share of uses) |
|---|---:|---:|
| exe string identifiers (94,699 words) | 26 / 2,356 | 35 / 2,249 (0.7%) |
| Previous-title shader parameter names (CodeWalker `ShaderParams.cs`, 1,688) | 28 | 99 (9.6%) |
| all of the above + effect names + SGD5 strings + `common.rpf` identifiers | 59 | 179 / 2,656 (11.7%) |

(The first two rows were counted before push-constant members were parsed: 2,249 cbuffer member names.) The exe
gives few names (`STs` from it is likely a collision). The top unnamed members are the 7 shared per-view
constants that appear in 16,782 programs (same cbuffer as `gViewProj`). Names are a convenience, not a blocker: the
material constants in section 4 join by hash.

### 2b. Effects: VS/PS pairing and render states

`upstream/palmetto/tools/sgd5_effects.py` reads the effect records: DB0 at 0x9000 (910 effects, program groups
11,018 VS/MS, 10,944 PS, 195 GS, 2,647 CS), DB1 at 0x10287000 (22 effects). Each effect has techniques, each pass
names one program per stage plus a rasterizer row (16 B), depth/stencil row (24 B) and blend row (68 B: 4-byte
prefix + 8 render-target entries with enable and write mask), 28,798 passes in all (Palmetto
`docs/sgd5_render_states.md`). Fill/cull, depth func/write and blend enable/write mask are identified; blend factors
are not decoded. Each effect also has an effect-level binding block (same 24-byte rows), a shared table of 55 named
sampler states (`PointClamp`, `TrilinearWrap`, `Anisotropic16XWrap`...) and per-program binding maps. One effect's
post-pass blocks were decoded this far (`postfx_lut`) and hold no default values; the 7.7 MB of post-pass data across
DB0 is otherwise unread, so defaults are not ruled out.

Coverage of the harness set (`harness_work/set.json`, 150 shaders), joined by offset: all 150 found, 114 have a
cbuffer, 55 have a push-constant block (2,212 members, 291 named), all 150 are in an effect pass, all 100 pixel
shaders have a VS in the same pass (the 50 CS run alone).

### 2c. Pipeline library (`PS\0\x09`)

Inflated from the zlib chain at 0x10579000 and checked on this file: magic `PS 00 09`, name `0x65F41EDD`, version
`930.048.020`, counts 25,471 graphics / 2,693 compute pipelines, 54,576 blobs, 21,522 programs. Per Palmetto
`docs/shader_library.md` (read, not re-measured beyond the header): graphics records name VS or MS + PS programs
(13,383 VS+PS, 10,726 MS+PS); the pipeline blob holds the vertex input layout with semantic names (`POSITION`,
`TEXCOORD`, `NORMAL`, `TANGENT`, `BLENDWEIGHT`...); programs link to carved shaders by the `HASH` digest (21,509 of
21,522). The 388-472-byte state blob, which should hold render-target formats, is undecoded. Tool:
`upstream/palmetto/tools/shader_library.py`.

### Not available or not found

- Constant default values: not found (one effect checked, see 2b). Real values come from materials, timecycle and
  settings (section 4).
- Render-target formats: not located (library state blob undecoded; blend rows have no format).
- Readable names inside the containers: none (no `RDEF`/`STAT`; `DXIL` is replaced by Xbox `XDAT`/`XBCH` data).

## 3. Textures

### `textures.rpf`: texture dictionaries

1,356 resources, each a `pgDictionary<grcTexture>` (hash table at +0x20, texture pointers at +0x30, texture struct
0x70 bytes: data pointer +0x18, name +0x30, width/height +0x38, DXGI format +0x3F, mips +0x42). 3,406 textures:
2,634 BC6H (in 1,317 dictionaries; names seen include `reference_hdr`, `waterfog`), 236 BC7, 227 BC3, 113 RGBA16F, 52 BC1, 34 BC7_SRGB,
42 RGBA8, 18 BC1_SRGB, 13 BC5, 8 BC4, plus a few R8/A8/R16/R16F/R32F/R10G10B10A2. Shader-relevant contents seen by
name: 110 colour-grading LUTs (17x17x17 RGBA16F, e.g. `content_drugscocaine`, `player_focusmodeentity`),
`cloudnoise`, `curlnoise` (32^3 R10G10B10A2), `windnoisevolume` (32^3), `ocean_sim_gaussian_noise`,
`snow_coverage_noise`, `waterbump`, `waterfoam`, `stars`, `moon-new`, `airglow`, `gradientmaps`, `heightmap`
(1536x1748 R16), `underwaterfog`, `nightlighting`, `groundlayerwind`, `blank_*` placeholders.

No BCPack here: plain BCn/raw blocks in 4 KiB tiles (Gfx10 `SW_4KB_S` bit orders per element size; 64 KiB macro
tiles when flags mode = 0), mips stored smallest first. `tools/gamedata/txd_decode.py` is a Python port of 6stars
`TextureLayout` plus the Asset-Info notes mip-offset rule, with Palmetto `ttpf_mips.decode_pixels` for BC1-BC7.

### `.ttpf` pools (`RDS5`)

| Pool | Textures | Chunk codecs |
|---|---:|---|
| `data/ttdb/graphics/ttdb_graphics.ttpf` | 912 (765 RGBA8, 46 BC1, 41 BC4, 31 BC7, 3 RGBA16F...; 8 arrays, 6 cubes, 3 volumes) | 1,402 zlib, 147 BCPack (BC7 27, BC5 100, BC1 19, BC3 1) |
| `data/ttdb/ui/platform.ttpf` | 1 (BC7) | 2 zlib |
| `pack/core_restofworld/data/ttdb/textures_loose.ttpf` | 194 (BC1 89, BC4 67, BC7 30, BC5 5, BC6H 2) | 664 zlib, 11 BCPack BC1 |

Palmetto `ttpf_mips.py scan`: 1,107 of 1,107 descriptors lay out (0 layout or format failures). Textures are keyed
by a 64-bit hash with no known name function. BCPack chunks need `--allow-provisional` (BC5 history entry 106 is
unconfirmed); the decoded result is clean (see g78 below).

### Samples decoded (PNG/DDS in the scratchpad, viewed)

| Texture | Source | Format | What it is |
|---|---|---|---|
| `cloudnoise` | `textures.rpf` dict 0x2263010 | BC1 512^2, 8 mips | tileable fBm cloud noise |
| `ocean_sim_gaussian_noise` | dict 0x9eea010 | R32_FLOAT 256^2 | Gaussian white noise: mean 0.0005, range -4.01..3.75 |
| `snownormalmap` | dict 0x8dae010 | BC5 1024^2 | tangent-space normal map (RG) |
| `waterbump` | dict 0x9eea010 | BC3 256^2 | swizzled water normal/bump map (data in G/A) |
| `nightlighting` | dict 0xb602010 | BC7 512^2 | map-space night light mask of the city |
| `gradientmaps` | dict 0x225b010 | BGRA8 256^2 | colour ramps in the top ~36 rows, rest black |
| ttdb_graphics #832 | `ttpf_mips export` | RGBA16F 256x2 | 1D colour ramp (0.075,0.023,0.045 to 0.93,0.90,0.92; A 1.85-2.62) |
| ttdb_graphics #200 | `ttpf_mips export` | RGBA16F 32x64x32 volume | precomputed LUT, RGB 0-0.67, A constant 0.1591 (about 1/2pi) |
| ttdb_graphics #78 | BCPack, provisional | BC5 256^2 x 32 slices | detail normal map array |
| textures_loose #0 | zlib | BC1 512^2 | decal/blob normal map |
| platform #0 | zlib | BC7 1024x512 | controller UI image |

Not verified: 3D textures in dictionaries (the 17^3 LUTs, `curlnoise`). Read linearly or as one 4 KiB tile per slice
the LUT shows no colour-to-axis correlation (best |r| 0.46, all on z), so the dictionary 3D layout is still open.
TTPF volumes are handled by Palmetto (`docs/textures_arrays.md`).

## 4. Realistic constants

1. Material constants in drawables (`data.rpf`; same layout in `levels.rpf`, which this copy lacks). Each material
   ("atem" record, Asset-Info notes `formats/07-models.md` 7.6) has a parameter table `{u32 joaat name, u32 descriptor
   = floats << 22 | byte offset << 8 | buffer}` and the constant buffer bytes. The offsets match the effect's cbuffer
   member offsets from 2a, so values drop straight into a cbuffer by hash, no names needed. Checked on one case:
   the 56 programs in `standard_emissive_layer` passes have `emissiveMultiplier` at byte 24 and `tintPaletteSelector`
   at byte 0 of cbuffer b4, where the material table puts them (the material's buffer byte, 3, is not the register). `material_constants.py`
   over `data.rpf`: 3,301 materials, 77,287 constants (top shaders `Weapon` 2,733, `standard_inst_dynamictint` 203,
   `STANDARD` 198). Sample, `standard_emissive_layer`: `emissiveMultiplier`=1000 (another 100),
   `tintPaletteSelector`=(0,0), `0x7ab0fd74`=(1,1,1), `0x823d5749`=-1; `standard_glass`: `specMapIntMask`=(1,0,0),
   `0xf08653a5` (`tintoverride` per Asset-Info notes)=(1,1,1,0), `0x8f296ec1`=-0.3.
2. Timecycle (`common.rpf`): `time_data` with 13 keyframe hours (0, 4, 5, 6, 7, 9, 12, 16, 18, 19, 20, 21, 22);
   27 `timecycle_keyframe_data` cycles (Sunny, Rain, Fog, OVERCAST, THUNDER, Hurricane, SandStorm, CLOUDSCAPE_*,
   calibration, underwater_deep...), each 1,142 variables x 13 hours for regions GLOBAL and URBAN; 49 modifier files
   with 4,593 modifiers over 1,056 variables. Example, HIGHPRESSURE `light_dir_mult` by hour:
   2.5, 1.5, 1.0, 1.0, 1.0, 1.197, 1.298, 1.298, 1.197, 1.0, 1.0, 1.0, 1.298; `postfx_intensity_bloom` 0.1498.
3. Visual settings (`common.rpf` stream 0x640000, 2,617 data lines, 134 prefixes): `Tonemapping.bright.filmic.A`
   0.22 (B 0.3, C 0.10, D 0.2...), `bloom.threshold.min` 0.4, `Exposure.curve.scale` -1.15,
   `Exposure.curve.verticalOffset` 6.9, `Adaptation.step.size.mult` 0.65, plus `temporalaa.*`, `cascadeshadow.*`,
   `sky.*`, `water.*`, `lights.*`, `motionblur.*`, `HDR10ToSDR.*`.

Sources 2 and 3 carry names but the engine maps them to cbuffer members in code; to use them per member we need the
member names (12.4% resolved) or a value-range join. Source 1 needs neither.

## 5. Where things live (quick reference)

| Need | File | Offset / locator | Reader |
|---|---|---|---|
| cbuffer layouts, slots per shader | `shaders.rpf` SGD5 records | record = `carved/index.json` offset | `tools/gamedata/sgd5_bindings.py` |
| VS/PS pairs, raster/depth/blend rows | `shaders.rpf` effects | DB0 0x9000, DB1 0x10287000 | `upstream/palmetto/tools/sgd5_effects.py` |
| pipelines, input layouts | `shaders.rpf` library | zlib chain at 0x10579000 | `upstream/palmetto/tools/shader_library.py` |
| named textures | `textures.rpf` | 1,356 dictionaries | `tools/gamedata/rpf_peek.py --txd`, `txd_decode.py` |
| hashed textures | `.ttpf` pools | `ETPF` index | `upstream/palmetto/tools/ttpf_mips.py` |
| material constants | `data.rpf` drawables | "atem" records | `tools/gamedata/material_constants.py` |
| timecycle, visual settings | `common.rpf` | 0x1b1000 (`time_data`), 0x37a000.. (cycles), 0x640000 (settings) | `tools/gamedata/timecycle_summary.py` |

## 6. Plan: feeding real data into the harness

Estimates are for one agent; "uncertain" marks steps whose cost depends on an unknown.

### (a) Real textures (about 3-4 days)

1. Export a texture library (0.5 day): run `txd_decode.py --raw` and `ttpf_mips.py export` into
   `harness_work/textures/` as linear block data + JSON (format, size, mips, source, name or hash). Start with a
   shader-relevant set of 200-300: noises, LUTs, normal maps, ramps, HDR probes, placeholders (`blank_*`).
2. Harness loader (1 day): a variant field `"tex": [[slot, file]]` that uploads the linear BCn/RGBA blocks as-is into
   an `MTLTexture` as BC1-BC7 pixel formats (to confirm on this Mac's GPU with `tools/metal_gpu_info.swift`), so no
   CPU pixel decode. Keep `tex_seed` as the
   fallback per slot.
3. Pick textures per slot (1-1.5 days): the binding table gives each shader's texture registers and name hashes;
   effects carry `{texture hash, parameter hash}` records (Asset-Info notes `12-materials.md`). Choose by parameter
   name where known (`bumptex` gets a normal map), else by the format the shader samples (BC5/RG to normal maps,
   R16F/R32F to noise or heights, 3D to LUT/volume). Measure with the existing kill-rate runs: real textures vs
   seeded textures on the 150.
4. Finish 3D dictionary layout (0.5-1 day, uncertain): needed only for the 110 grading LUTs and the curl/wind
   volumes.

### (b) Real constants and pipeline info (about 5-9 days)

1. Binding JSON per shader (0.5 day): extend `sgd5_bindings.py` to write `<prep>/<shader>/bindings.json` (cbuffers,
   members, SRV/UAV/texture registers). Map member type codes to HLSL types from offsets and strides (0.5 day).
2. Register to harness memory (2-4 days, uncertain): the harness feeds constants through title memory reached by
   user-SGPR pointers plus push dwords. The push-constant block layout (2a) maps the `push` dwords directly for the
   55 of 150 shaders that have one: start there (about 0.5 day). For cbuffers we must learn which user SGPR /
   descriptor-table entry serves b4 space 1 etc. Start from
   `upstream/palmetto/tools/root_sig_contract.py`, `rootsig_census.py`, `shader_meta.py` and the `PRIV` part of each
   container; validate on TAA (two cbuffers of known size) by checking that loads land inside the sized window.
3. Value pools (1-2 days): per member hash, a value set from the 77,287 material constants (works unnamed); for
   per-view/postfx members, values from timecycle + visual settings where names resolve, plus generated but realistic
   matrices for view/projection members (`gViewProj`, `gPrevViewProj`, `gViewInverse` are named). Feed them as a new
   input-variant origin ("realistic") beside the fuzzed ones; grow the name dictionary as a side task.
4. Pipeline pairing and state (1-2 days): use the VS from the same SGD5 pass (all 100 PS in the set) instead of
   a generic VS; input layout from the library pipeline blob; depth func/write and blend enable/write mask from the
   pass rows. Render-target formats stay unknown until the library state blob is decoded (open, not estimated).

## 7. Tools and reproduction

Scripts written for this survey (all read-only, Python standard library, run with `python3 -I`):

- `tools/gamedata/sgd5_peek.py SHADERS_RPF [--summary] [--effect NAME] [--program-sample N]`: effect region sizes,
  one effect's bytes and string pool, the 152-byte block and trailer of sample programs.
- `tools/gamedata/sgd5_bindings.py SHADERS_RPF --name SUBSTR [--dict WORDS]` / `--census [--db 0|1] [--dict WORDS]`:
  binding tables per shader; corpus census with dictionary resolution.
- `tools/gamedata/rpf_peek.py ARCHIVE INDEX.json --txd | --dump OFFSET [--n N] | --grep REGEX`: reads streams straight
  from the archive using a carve index; lists texture dictionaries.
- `tools/gamedata/txd_decode.py ARCHIVE INDEX.json DICT_OFFSET NAME... --out DIR [--raw]`: dictionary texture to PNG
  (mip 0) plus optional linear block bytes.
- `tools/gamedata/timecycle_summary.py ARCHIVE INDEX.json [--show VAR...] [--json OUT]`: keyframe and modifier census.
- `tools/gamedata/material_constants.py ARCHIVE INDEX.json --dict WORDS [--max-resources N] [--show K]`: material
  constant values from drawables.

Reproduce (W = any work folder outside the game folder):

```sh
cd "<game root>"
for a in common textures data script; do
  nice python3 -I upstream/palmetto/tools/rpf9_carve.py $a.rpf W/carve_$a --index-only; done
python3 -I tools/gamedata/rpf_peek.py textures.rpf W/carve_textures/index.json --txd
python3 -I tools/gamedata/txd_decode.py textures.rpf W/carve_textures/index.json 0x2263010 cloudnoise --out W/tex
python3 -I upstream/palmetto/tools/ttpf_mips.py scan data/ttdb/graphics/ttdb_graphics.ttpf
python3 -I upstream/palmetto/tools/ttpf_mips.py export data/ttdb/graphics/ttdb_graphics.ttpf 78 W/g78.dds --allow-provisional --mips 1
python3 -I upstream/palmetto/tools/dds_png.py W/g78.dds W/g78.png --layer 9
python3 -I tools/gamedata/sgd5_bindings.py shaders.rpf --name CS_ApplyTemporalAA_b2b938f9
python3 -I tools/gamedata/sgd5_bindings.py shaders.rpf --census --dict W/words.txt
python3 -I tools/gamedata/timecycle_summary.py common.rpf W/carve_common/index.json --show light_dir_mult
python3 -I tools/gamedata/material_constants.py data.rpf W/carve_data/index.json --dict W/words.txt --max-resources 100000
```

`W/words.txt` used here: identifier strings of the exe (`strings -n 3 | grep -oE '[A-Za-z_][A-Za-z0-9_]{2,80}'`), the
names in CodeWalker `CodeWalker.Core/GameFiles/Resources/ShaderParams.cs`, the 932 effect names, identifiers from
the SGD5 effect region and from every `common.rpf` document (dotted keys also as CamelCase and snake_case).

Palmetto tooling to reuse before writing more parsers (checked, not all run here):
`tools/ttpf_mips.py` + `tools/bcpack.py` / `bcpack_tables*.py` (used above), `tools/bcpack_calibration/` (Rust; builds
local BC6H/BC7 history tables passed with `--history`; Palmetto documents BC6H/BC7 as still needing
`--allow-provisional` too), `tools/calibrated_texture.py`,
`tools/amd_swizzle.py` (full Gfx10 addrlib swizzle equations with a smoothness search: the right tool to settle the
3D dictionary layout of plan step a4), `tools/tex_headers.py` + `tools/tex_export.py` (dictionary textures from a
written carve; `txd_decode.py` exists because it reads straight from the archive without writing a carve),
`tools/rpf9dump` and `tests/Rpf9DumpDiagnostics` (C# bulk exporters), `tools/psin_*.py` and `tools/gamedata_*.py`
(PSIN and `common.rpf` readers). `levels.rpf` (71 GB, the bulk of the drawables and so of material constants) was
still downloading during this survey and was not read; rerun `material_constants.py` on it once it is complete.

External references read: 6stars (`/tmp/portwatch/6stars`, `Sources/RPFKit`: ArchiveIndex, TextureDB,
TextureDictionary, PRD0, PSO, Timecycle), Asset-Info notes (`formats/01-rpf9.md`, `03`, `04`, `05-ttpf.md`,
`07-models.md` 7.6, `12-materials.md`, `structure/06-models-shaders.md`), Palmetto docs (`rpf.md`,
`shader_library.md`, `pso_cache.md`, `sgd5_render_states.md`, `textures.md`, `gamedata.md`).
