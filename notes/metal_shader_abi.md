# Metal shader runtime ABI (Palmetto RDNA2 -> SPIR-V -> SPIRV-Cross MSL)

Written 2026-11-24 from the code as it stood that day. The goal: a third-party Metal runtime (for example a native
Metal Xbox Series emulator) can run the shaders this pipeline produces without reading our code.

Sources and the versions read (line numbers refer to these):

| Short name | File | State read |
|---|---|---|
| `harness` | `tools/harness/harness.swift` | 742 lines, mtime 2026-11-24 21:09. `harness_fast.swift` will replace it; function names are given with each line so the facts can be found again. |
| `prepare` | `tools/harness/prepare.py` | 541 lines, mtime 2026-11-24 22:28 |
| `xlat` | `upstream/palmetto/tools/rdna2spirv.py` | 19,265 lines (Palmetto copy, not a git checkout) |
| `set0` | `upstream/palmetto/tools/spirv_metal_set0.py` | 216 lines |
| `bindless.h` | `upstream/palmetto/runtime/src/vulkan_bindless.h` | the binding table the set-0 fix reads |
| `dtable.h` | `upstream/palmetto/runtime/src/descriptor_table.h` | Palmetto's host hash table |
| `gfx10.h` | `upstream/palmetto/runtime/src/gfx10_descriptors.h` | T#/S#/V# field layouts |
| `d3d12doc` | `upstream/palmetto/docs/d3d12_vulkan.md` | sections 3.3, 3.5, 5, 5.1-5.3 (the Vulkan-side contract) |
| examples | `harness_work/prep_h/<shader>/{ref.metal,vs.metal,meta.json}`, `harness_work/spirv/index.json` | the 150-shader working set |

Each fact is tagged with the layer that sets it:

- **[P]** Palmetto contract: fixed by the translator, the same on Vulkan and Metal.
- **[M]** Metal mapping: fixed by `set0` plus the SPIRV-Cross flags in `prepare` (required for any Metal runtime).
- **[H]** harness or `prepare.py` addition: test-only instrumentation or a fake input. Not ABI.

Anything not read directly in code is marked **(inferred)**.

Example shaders used below (all in `harness_work/prep_h/`):

| Shader | Why |
|---|---|
| `<pixel shader B>` | pixel shader: one 2D texture, a literal S#, two hash lookups, format tables `_146`/`_147` |
| `<compute shader A>` | compute 8x8x1, wave64 emulated, LDS 2,612 bytes, storage images (bindings 8-10) |
| `<pixel shader D>` | pixel shader writing depth only; no descriptor set, so its push constants sit at `buffer(0)` |
| `<compute shader C>` | compute 64x1x1 using title memory only; push constants at `buffer(0)` |

---

## 0. Which MSL to run

The ABI is the SPIRV-Cross output after the set-0 fix, before `prepare.py` touches it:

```
python3 -I upstream/palmetto/tools/spirv_metal_set0.py in.spv fixed.spv          # prepare:451
spirv-cross --msl --msl-version 30100 --msl-argument-buffers \
            --msl-argument-buffer-tier 1 --msl-decoration-binding fixed.spv --output out.metal   # prepare:455-456
```

`prepare.py` keeps that text only in a temp dir (prepare:449-461). `ref.metal` is that text plus three rewrites
(prepare:462-464). Two of them change behaviour:

| Rewrite | Effect | Required? |
|---|---|---|
| `sandbox()` (prepare:340-374): every `reinterpret_cast<device T*>(x)` becomes `reinterpret_cast<device T*>(sbx(x))` | wraps every title-memory address into the test window (section 3) | **[H]** Not ABI. It becomes the identity only for an address already inside the window, so it changes results for any address outside it. |
| `instrument()` (prepare:197-242): adds `device atomic_uint* sbx_cov [[buffer(29)]]` to `main0` (prepare:207), `sbx_hit(i)` per block, `SBX_LOOP_GUARD` at the top of every loop | `SBX_LOOP_GUARD` is **not** behind `SBX_COV` (prepare:60): every loop breaks after `SBX_LOOP_CAP` (default 4,096, prepare:48-50) iterations, with one counter `sbx_iter` shared by all loops of `main0` (prepare:230) | **[H]** Not ABI. A runtime that runs `ref.metal` as-is must bind a buffer at index 29 and accepts a silent 4,096-iteration cap. |
| `instrument_guards()` (prepare:267-300): `X = c ? a : X` becomes `X = sbx_sel(id, c) ? a : X` | value-preserving (`c` evaluated once; prepare:58-59) | **[H]** |

`cmplog.metal` (prepare:145-181, `SBX_CMPLOG`, `sbx_cl`, `sbx_kl`) and `SBX_READS` (prepare:67-87) are search-only
instrumentation **[H]**.

A runtime should regenerate the MSL with the two commands above. If it must use `ref.metal`, it has to compile it
with `SBX_BASE` = the arena's `gpuAddress`, `SBX_MASK` = arena size - 1 (power-of-two arena), `SBX_LOOP_CAP`
raised past any real loop count and a writable dummy buffer at index 29 (inferred: no one runs it that way).

---

## 1. Entry points, stages, threadgroups and wave size

| Item | Value | Source |
|---|---|---|
| Entry point name | `main0` in every shader **[M]** | SPIRV-Cross default; harness looks it up by that name (harness:438, 491) |
| Stages produced in the working set | `kernel` (cs, 50) and `fragment` (ps, 100) | `index.json` |
| Stages the harness runs | `cs` and `ps` only; anything else is `unsupported` | harness:591 |
| Pixel-shader vertex stage in tests | generated `vs.metal`, entry `vs_main` **[H]** | prepare:393-415, harness:491 |
| Threadgroup size | `local_size` (index/meta) **[P]**. MSL kernels do not declare it, but the code hard-wires it (for example `gl_LocalInvocationID.y * 8u` in TemporalAA). The host must dispatch exactly `local_size` threads per threadgroup. | xlat:17607-17620 (from `COMPUTE_PGM_RSRC2` and words 0x13C-0x144 of the FSSM), harness:451-452 |
| Threadgroup memory | declared inside `main0` as `threadgroup` arrays (LDS plus the wave64 exchange buffer). The host sets no threadgroup memory length. Total = `lds_bytes`. | TemporalAA `ref.metal:288-289`; xlat:18678-18683 |
| Threadgroup count | the console dispatch's grid **[P]**. The harness uses 4x1x1 (or 2x2x1 when `local_size.y > 1`) **[H]** | harness:474 |
| Max threads check | harness refuses `local_size` above `maxTotalThreadsPerThreadgroup` | harness:453-456 |

**Compute system values [P].** After the 32 push-constant dwords are copied into `s0..s31` (xlat:15913-15918), the
enabled workgroup-ID components overwrite `s[user_sgprs + k]` (xlat:15962-15967) and the local invocation ID goes to
`v0..v2`. Metal builtins used: `threadgroup_position_in_grid` and `thread_position_in_threadgroup`. No grid base, no
`threadgroups_per_grid` in any of the 50 kernels: a dispatch with a base offset is not representable (inferred).
`TG_SIZE_EN` (wave index in the group) is synthesized from the local ID (xlat:15970-16007).

**Wave64 handling** (RDNA2 wave64 on Apple simd32). The flags used to translate the 150 are recorded nowhere read;
the per-shader fields in `index.json` say what was done:

| Case | Count in the 150 | What it means for the runtime |
|---|---|---|
| CS, `wave64: true`, `wave64_exchange: "workgroup"`, `subgroup_size: null`, `control_flow: "emulated wave64 ..."` | 47 | Exact on any SIMD width: no SIMD-group ops; cross-lane data goes through threadgroup memory (`--wave64-emulate`, xlat:17762-17765). Translated with that flag (inferred: the paused rest-of-corpus run shows no such flag). |
| CS, `wave64: false`, `subgroup_size: 32` | 3 | Native wave32: SIMD-group ops on 32 lanes, matches Apple's simd32. |
| PS, `wave64: true`, `subgroup_size: 64`, `wave64_emulated: true` | 100 | Metal gives 32-wide SIMD groups, so every 64-lane wave runs as two independent halves. Exact for `wave64_graphics` tier **A** (65) and **B** (3, exact up to atomic order). **Not exact** for tier **C** (32): cross-half exchange is lost. `wave64_graphics_reasons` names the construct. Definitions: `docs/design.md`, row "Wave64 graphics". |

PS SIMD builtins in use: `thread_index_in_simdgroup` (77 PS), `threads_per_simdgroup` (13 PS, needed by
`spvSubgroupBallotFindLSB`). Every descriptor lookup result is passed through `simd_broadcast_first`
(`spvSubgroupBroadcastFirst`, xlat:12527-12535). Metal forbids SIMD-group functions in vertex functions, so a
translated VS with a lookup fails to compile (TODO.md, "Findings so far": 7 of 19 VS fail).

---

## 2. Bindings

### 2.1 Buffer indices

| Metal index | Contents | Present when | Layer |
|---|---|---|---|
| `buffer(0)` | argument buffer `spvDescriptorSetBuffer0` (set 0) | the shader uses any image, sampler or the hash table | [M] |
| `buffer(1)` | push constants (144 bytes, section 4) | the shader has a descriptor set | [M] |
| `buffer(0)` | push constants | the shader has **no** descriptor set (13 PS, 10 CS in the 150; for example LinearDepthOpaque, ClearBitVolume) | [M] |
| `buffer(29)` | `sbx_cov` coverage / loop-guard buffer, 2^18 uint32 | every `ref.metal`/`cmplog.metal` | [H] |

The push-constant index is recorded nowhere: the harness finds it by reflection. It treats any buffer binding whose
struct has no texture, sampler or pointer members as the push block and fails if there are two (harness:344-358).
Rule observed on all 150 (inferred from SPIRV-Cross numbering): `buffer(1)` if `spvDescriptorSet0` exists, else
`buffer(0)`.

Title memory is **not** bound to any index. The shader reaches it through a raw 64-bit GPU address built from push
dwords 32-33 (section 3). The host must make the buffer resident with
`useResource(memory, usage: [.read, .write])` (compute, harness:472) or `useResource(..., stages: .fragment)`
(render, harness:554), or put it in a heap passed to `useHeap`.

### 2.2 Argument buffer (set 0) layout [M]

`set0` gives each set-0 binding a fixed first ID = sum of the capacities of the lower bindings (set0:42-48,
98-154). Capacities come from `bindless.h:100-116`. `--msl-decoration-binding` turns them into `[[id(N)]]`.
Output of `spirv_metal_set0.py --print-layout`:

| Binding | Vulkan type | Count | Metal IDs | MSL member type | Holds |
|---|---|---|---|---|---|
| 0 | sampled image | 4096 | 0..4095 | `array<texture2d<uint>, 4096>` | 2D sampled (also `texture2d<int>`/`<float>` aliases, below) |
| 1 | sampler | 1024 | 4096..5119 | `array<sampler, 1024>` | samplers (S#) |
| 2 | storage buffer | 1 | 5120 | `const device _N* m` (`struct _N { uint _m0[1]; }`) | descriptor hash table (section 5) |
| 3 | sampled image | 1024 | 5121..6144 | `array<texture2d_array<uint>, 1024>` | 2D array and cube (6-layer arrays) |
| 4 | storage image | 1024 | 6145..7168 | `array<texture2d<uint, access::read_write>, 1024>` | r32ui 2D, image atomics |
| 5 | sampled image | 256 | 7169..7424 | `array<texture1d<uint>, 256>` | 1D sampled |
| 6 | sampled image | 256 | 7425..7680 | `array<texture3d<uint>, 256>` | 3D sampled |
| 7 | storage image | 256 | 7681..7936 | (not seen) | r32ui 3D, image atomics |
| 8 / 9 / 10 | storage image | 256 each | 7937 / 8193 / 8449 | `array<texture2d<float / uint / int, access::write>, 256>` | 2D storage, float / uint / sint class |
| 11 / 12 / 13 | storage image | 256 each | 8705 / 8961 / 9217 | `texture2d_array<..., access::write>` | 2D-array storage, f / u / s |
| 14 / 15 / 16 | storage image | 256 each | 9473 / 9729 / 9985 | `texture3d<..., access::write>` | 3D storage, f / u / s |
| 17 / 18 / 19 | storage image | 256 each | 10241 / 10497 / 10753 | (not seen) | 1D storage, f / u / s |
| 20..25 | sampled image | 256 each | 11009..12544 | (not seen) | multisampled 2D / 2D array per class |

Binding formulas [P]: sampled 2D/2D-array/3D/1D = 0/3/6/5; storage = `8 + 3 * dim + class`
(dim 0 2D, 1 2D array, 2 3D, 3 1D; class 0 float, 1 uint, 2 sint); r32ui = 4 (2D) or 7 (3D)
(bindless.h:155-167, xlat:13609-13613).

Members a shader does not use are absent from its struct. IDs stay fixed, so one argument-buffer layout serves every
shader (set0 docstring). Bindings seen in the 150 (`index.json` `bindings`): 0, 1, 2 (127 shaders), 3 (17), 4 (3),
5 (1), 6 (10), 8-10 (21), 11-16 (7).

**Typed aliases of the sampled arrays.** SPIRV-Cross keeps one member per ID and comments the others out as
"Overlapping binding". In MSL the sampled arrays are declared `uint` (one shader,
`Hscattering_..._DensityShadowBlur_H`, declares only `texture2d<float>` at ID 0). The other classes are reached by
`reinterpret_cast` of the same array inside `main0`:

```
constant auto &_9  = reinterpret_cast<constant array<texture2d<int>,   4096> &>(spvDescriptorSet0.m_8);   // PS_Textured ref.metal:207
constant auto &_10 = reinterpret_cast<constant array<texture2d<float>, 4096> &>(spvDescriptorSet0.m_8);   // :208
```

The shader picks float, uint or sint per access from the T# FORMAT (section 5.4). So the texture the host encodes at
element `i` must have a pixel format whose class matches its T# format class (float-sampleable for UNORM/SNORM/FLOAT,
`*Uint` for UINT, `*Sint` for SINT). Encoding a texture into a slot whose declared type differs (a float texture
into `array<texture2d<uint>>`) is accepted by the tier-1 encoder (inferred; the harness never does it with a float
texture, section 10).

**Encoding.** Tier 1: the layout is opaque, so the host must use `function.makeArgumentEncoder(bufferIndex: 0)`
(harness:359-361), `setTextures` / `setSamplerStates` over ID ranges and `setBuffer(hashTable, offset: 0, index: 5120)`
(harness:362-385). Samplers in an argument buffer need `MTLSamplerDescriptor.supportArgumentBuffers = true`
(harness:261). Every texture and the hash table must be made resident with `useResource` (harness:376, 381, 473, 555).
11 shaders declare the hash-table pointer as non-const `device`; none writes through it (checked on the 150).

**Index range.** The lookup result indexes the array with no clamp (xlat:13123-13131, `OpAccessChain` on the raw
index). An index must be below the capacity of every array the shader uses it with: 4096 for 2D sampled but 256
for the 2D storage arrays (inferred consequence; Palmetto writes one index into a dimension's sampled array and its
storage arrays together, bindless.h `write_image`).

**Index 0** is the miss result. It must hold a valid fallback in every array: Palmetto uses a 1x1 RGBA8_UNORM image,
a default sampler and dummy storage images (d3d12doc 5.1). The shader treats index 0 as float-class whatever the
format bits say (xlat:13751-13755: `hit = idx != 0` gates the uint/sint paths).

---

## 3. Title memory model

**Contract [P]** (xlat:109-114, 1529-1536; d3d12doc 3.3):

```
host_address = heap_base + console_va          // 64-bit add, heap_base = push dwords 32 (lo) and 33 (hi)
```

Every buffer, descriptor-table and code access goes through it. `console_va` is a 48-bit console GPU VA:
- V# base = `word0 | (word1 & 0xFFFF) << 32` (gfx10.h:676-679; visible in MSL as
  `(ulong(w1 & 65535u) << 32ul) | ulong(w0 & 4294967292u)`).
- Descriptor-table pointers from user SGPRs are 32-bit (`ptr32_lo`): the shader forces the high dword to 0
  (`_251[9u] = 0u` in PS_Textured; `descriptor_access.truncates_table_ptr_high` in the index). So shader-visible
  descriptor heaps must sit below 4 GiB of console VA (d3d12doc 3.3).
- V# loads are bounds-checked in the shader against NUM_RECORDS and stride and return 0 out of range
  (PS_Textured ref.metal:272-282).
- T# base addresses are never dereferenced: images are reached only through the hash table (section 5).

So a Metal runtime needs one `MTLBuffer` (the arena) holding every buffer, descriptor heap and FSSM code blob at its
console VA as a byte offset, with `heap_base = arena.gpuAddress`. A V# can't point outside the arena: biasing a
foreign buffer by `-heap_base` was measured to fault for V#s on Vulkan (d3d12doc 3.3, D3D-53). There is no bounds
check on non-V# accesses: a VA past the arena reads or writes whatever lies there (inferred).

**What the harness does [H]** (harness:106-111, 117-129):

| Item | Value |
|---|---|
| Window size | `--window-mb`, default **16 MiB** (harness:107). The file header (harness:16) still says 64 MiB: stale. |
| Guard | 1 MiB after the window, in the same `MTLBuffer` (harness:107, 110); not compared |
| Buffer | `makeBuffer(length: window + guard, .storageModeShared)`; `heap_base = memory.gpuAddress` (harness:110-111) |
| `SBX_BASE` | `heap_base` as `0x...ul`, passed as a compile-time macro (harness:153). The library is tied to that buffer address. |
| `SBX_MASK` | `window - 1` = `0xFFFFFF` (harness:154). The prelude default `0x3FFFFFF` (64 MiB, prepare:37-39) applies only when the macro is missing. |
| Wrap | `sbx(a) = SBX_BASE + ((a - SBX_BASE) & SBX_MASK)` (prepare:40), so the access goes to `heap_base + (console_va mod window)`. Only the base pointer is wrapped; an element index past it lands in the guard. |
| Contents | seeded values (harness:93-102, 119), the FSSM blob copied at `codeVA = window - 2 MiB` (harness:108, 120-127) |
| Table pointers | each `ptr*_lo` user SGPR gets `0x100000 + t * 0x10000` (harness:109, 139); `ptr*_hi` gets 0 (harness:140) |

---

## 4. Push constants (user SGPRs) [P]

Block declared at xlat:15634-15646, loaded at xlat:15905-15918. `push_constant_bytes` = 144.

| Dword | Byte offset | MSL member (`struct _N`) | Contents |
|---|---|---|---|
| 0..31 | 0..127 | `uint _m0[32]` | `s0..s31` user SGPRs, copied into the SGPR file at entry |
| 32 | 128 | `_m1` | `heap_base` low 32 bits |
| 33 | 132 | `_m2` | `heap_base` high 32 bits |
| 34 | 136 | `_m3` | `code_base` low: console VA of this shader's FSSM blob |
| 35 | 140 | `_m4` | `code_base` high |

`s_getpc_b64` becomes `code_base + offset of the next instruction in the blob` (64-bit, xlat:10191-10209). The result
is a console VA, so the blob must be in the arena at `code_base` (d3d12doc 5, "Code-embedded data"). Only shaders with
`uses_code_base: true` in the index read dwords 34-35 (5 of the 150). Translator options that change this block
(`--pc-tail-ubo`, `--mesh-compute`: 152 bytes) were not used for the 150 (inferred: no `pc_tail`/`mesh_compute` keys
in the index).

**Filling s0..s31 from console state.** Each shader lists what it reads in `user_sgpr_reads`
(`[{"kind", "sgpr"}]`): `const32` (a 32-bit value), `ptr32_lo` (a 32-bit descriptor-table VA; the high half is forced
to 0), `ptr64_lo`/`ptr64_hi`, `vsharp`, `tsharp`, `ssharp` (d3d12doc 3.5). The 150 contain only `const32` (177
reads) and `ptr32_lo` (308). The mapping from a D3D12 root signature to SGPRs is Palmetto's model, not ours:

- Compute: base **s4**, measured. In root order, one dword per 32-bit constant, two per root CBV/SRV/UAV (64-bit VA),
  **one** per descriptor table (its 32-bit VA). `s0..s3` are a driver block the layer leaves zero; `s[0:1]` holds a
  spill pointer on the console. (d3d12doc 3.5, D3D-448, checked against 29 FidelityFX signatures.)
- VS/PS/GS base s8 and mesh base s16: a hypothesis from the read census only (d3d12doc 3.5).
- One 144-byte block is shared by every stage of a pipeline, so a VS and a PS that need different user data or both
  need a code base can't be fed (d3d12doc 5.2).

**Harness fill [H]** (harness:133-144): `push` is 64 dwords; s0..s31 seeded; `ptr*_lo` -> table VAs (section 3);
dwords 32-35 as above; variants may patch dwords 0..31 only (harness:680). Sent with `setBytes` (256 bytes) at the
reflected index (harness:468, 550).

---

## 5. Descriptor hash table

### 5.1 Buffer layout [P]

Bound as the pointer at argument-buffer ID 5120 (Vulkan set 0 binding 2). Layout (dtable.h:60-66, 182-194;
d3d12doc 5.1):

```
word 0                 mask = slot count - 1        (slot count a power of two)
words 10*s+1 .. 10*s+8 key[0..7] of slot s
word  10*s+9           index of slot s
word  10*s+10          pad: Palmetto writes 0xA11CA7ED when occupied, 0 when empty; the shader never reads it
```

### 5.2 Lookup the shader runs [P]

`desc_probe` (xlat:12537-12599; MSL in PS_Textured ref.metal:146-173):

```
h = 2166136261
for i in 0..7: h = (h ^ key[i]) * 16777619          // FNV-1a over 8 dwords, not bytes
for k in 0..7:
    s = ((h + k) & mask) * 10
    if table[s+1..s+8] == key[0..7]: return table[s+9]
return 0
```

All 8 slots are compared. An empty slot does not stop the probe, so a hole breaks no chain (dtable.h:134-139). A key
of eight zero words would match an empty slot and return its index 0, which is the miss result anyway. The result is
passed through `simd_broadcast_first` (section 1).

### 5.3 Keys and how a host builds the table [P]

Only images and samplers are hashed. **V#s are not:** buffers are dereferenced through `heap_base` (section 3).

| Descriptor | Key (8 dwords) | Source |
|---|---|---|
| T# (image) | the 8 dwords exactly as the shader holds them in SGPRs | dtable.h:41-45 |
| S# (sampler) | the 4 dwords, then `0xFFFFFFFF` x 4 | dtable.h:47-58; xlat:13076-13090 (`desc_lookup_sgpr` pads to 8) |

Comparison is bitwise on all 8 words. A T# that differs in any bit (MIN_LOD, swizzle, a field the game patches in
SGPRs before the sample) misses and gets index 0.

Host procedure (Palmetto `DescriptorTable`, dtable.h:68-110; d3d12doc 5.1):
1. For each T#/S# the game creates or writes into a descriptor heap (also on `CopyDescriptors` and heap scans):
   decode it, create the Metal texture or sampler, give it a free array index, insert `key -> index`.
2. Insert at the first free slot among `(h + k) & mask`, `k = 0..7`. A duplicate key updates its index in place.
3. Keep the load at or below 50% and grow (double, rehash) when the 8 slots are full. Palmetto first tries moving
   one resident to a free slot within that resident's own 8-probe window (dtable.h:249-310).
4. Pre-register each shader's `literal_samplers` (S# values the shader builds from immediates, listed per shader in
   `index.json`, for example `[146, 12582912, 407896064, 0]` in PS_Textured) at pipeline creation. Without that the
   lookup misses and samples with the fallback sampler (Palmetto `docs/design.md`, "Literal samplers").
5. Index 0 is reserved for the fallbacks (section 2.2).

**Harness [H]** (harness:264-285): 1,024 slots (mask 1023), buffer `(1024 * 10 + 16) * 4` bytes, zeroed per variant.
Entries come only from a test variant's `"hash": [[k0..k7, index], ...]`, so with none every lookup misses
(index 0). A slot counts as empty when its index word is 0 (harness:278); the pad word stays 0. When all 8 probes
are taken the entry is silently dropped (no relocation, no growth).

### 5.4 Format tables (T# FORMAT -> view class) [P]

Each image access that may meet an integer format reads `FORMAT = (T# word1 >> 20) & 0x1FF` (9 bits, word 1 bits
20-28; gfx10.h:116) and tests it in two 512-bit tables, each a `constant spvUnsafeArray<uint, 16>` indexed
`table[FORMAT >> 5]` bit `FORMAT & 31` (xlat:13617-13629, 13663-13681). The MSL names are per shader (`_146`/`_147`
in PS_Textured, `_191`/`_192` in `%PS_MaterialBlendMod`, `Cscattering_..._BilateralBlurX`, `PS_DrawMap`): identify
them by content. The first is the UINT mask, the second the SINT mask (built from `tools/gfx10_formats.json` names
ending `_UINT` / `_SINT`):

```
uint mask: {135530528, 1346437636, 2336, 0, 35864576, 0, 0, 0, 357913600, 4, 0, 0, 0, 0, 0, 0}
sint mask: {271061056, 2692875272, 4672, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0}
```

Selection (xlat:13726-13760; PS_Textured ref.metal:312-334):

```
idx = lookup(T#)
if (sint_bit(FORMAT) && idx != 0)  -> texture<int>   view, result bits as uint
elif (uint_bit(FORMAT) && idx != 0) -> texture<uint>  view
else                                 -> texture<float> view, result bits via as_type<uint4>
```

Two shaders carry four tables: the extra two (UNORM/SNORM and 32-bit-float masks, xlat:13631-13661) drive the d16
rounding rule. Typed buffer loads decode V# formats in the shader with further private tables (`%fmt_desc_tab`,
`%fmt_size_tab`); they need nothing from the host.

The runtime consequence: the class the host gives the texture at index `i` must be the class these tables give for
the FORMAT of the T# it registered under `i`.

---

## 6. Textures and samplers

**T# -> Metal texture** (no Metal code exists for this; Palmetto's Vulkan decoder is the reference, gfx10.h:76-205):

| T# field | Bits | Use |
|---|---|---|
| BASE_ADDRESS | word0 = VA >> 8, word1[7:0] = VA >> 40 | where the console image lives (needs detiling; Palmetto has `runtime/src/gfx10_tiling.h`) |
| MIN_LOD | word1[19:8], 4.8 fixed | view LOD clamp |
| FORMAT | word1[28:20] | GFX10_FORMAT code -> pixel format (gfx10.h `format_table`) and the class of section 5.4 |
| WIDTH | word1[31:30] + word2[11:0], minus 1 | |
| HEIGHT | word2[27:14], minus 1 | |
| DST_SEL_X/Y/Z/W | word3[11:0] | swizzle (Metal texture swizzle) |
| BASE_LEVEL / LAST_LEVEL | word3[15:12] / [19:16] | view mip range |
| TYPE | word3[31:28] | 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array, 14 2D MSAA, 15 2D MSAA array |
| DEPTH / BASE_ARRAY | word4[12:0] / [28:16] | 3D depth or last layer / first layer |
| MAX_MIP | word5[7:4] | resource mips - 1 |

Which array a texture goes in is fixed per instruction by its `dim:` modifier at translation time; the class is
chosen at run time (section 5.4). Cube and cube-array images must be presented as 2D-array views of 6 (6N) layers:
the shader computes the face itself (gfx10.h:192-205; xlat `cube_coords`). Storage writes pick binding
`8 + 3 * dim + class` by the same FORMAT test; image atomics use the r32ui arrays (4, 7).

**S# -> MTLSamplerState** (gfx10.h:495-501): CLAMP_X/Y/Z word0[2:0]/[5:3]/[8:6], MAX_ANISO_RATIO [11:9],
DEPTH_COMPARE_FUNC [14:12], FORCE_UNNORMALIZED [15], FILTER_MODE [30:29]; MIN_LOD/MAX_LOD word1[11:0]/[23:12] (4.8);
LOD_BIAS word2[13:0] (signed 5.8), XY_MAG_FILTER [21:20], XY_MIN_FILTER [23:22], MIP_FILTER [27:26];
BORDER_COLOR_PTR word3[11:0], BORDER_COLOR_TYPE [31:30]. `gather4` with explicit LOD is emulated in the shader from
the S# words (xlat `gather_addr_modes`, `gather_border_one`): border type 3 (table colour) reads as transparent black,
half-border modes 4/5 as clamp-to-edge when the S# is not a literal.

**Harness [H]** (harness:174-263): one texture per (texture type, data type, access, depth, member for writable) fills
**every** element of its array (harness:375). 64x64 (1D: 64x1; 3D: 64x64x16; 2D array: 6 slices); 7 mips for
read-only 2D/2D-array. Formats from the **declared** element type: uint -> `rgba32Uint` (read) / `r32Uint` (write),
int -> `rgba32Sint` / `r32Sint`, float -> `rgba16Float` (read) / `rgba32Float` (write), depth -> `depth32Float`
(harness:180-188). `typeCube` becomes a 6-slice `type2DArray` (harness:223). One sampler for everything: linear
min/mag/mip, repeat on all axes (harness:257-263).

---

## 7. Pixel shader I/O

**Inputs [P][M]** (`struct main0_in`, `[[stage_in]]`):

| Item | MSL | Meaning |
|---|---|---|
| Attribute n | `float4 m_K [[user(locnN)]]` | console PS attribute n = location n (xlat:14258-14300). Raw 32-bit words, interpolated as floats unless flat. Which VS export feeds attribute n is console state (SPI_PS_INPUT_CNTL); the translator assumes `exp paramN` -> attribute N (inferred). |
| Flat attribute | `[[user(locnN), flat]]` | listed in `ps_flat_interpolants` (14 inputs in the 150) |
| f16 halves | extra `float4` inputs, for example locations 26/27 for param 1 | `--f16-interp-fallback` (xlat:89-101): for each param in `f16_halves` ({param: [low_loc, high_loc]}) the producing VS must also export that param's low and high f16 halves decoded to f32 at those locations. 18 of the 150 PS have `f16_halves`, so the set was translated with that flag (inferred). |
| `gl_FragCoord` | `float4 [[position]]` | 46 PS; `ps_system_inputs` "0".."3" = components x, y, z, w (inferred from the names). Whether Metal's `.w` matches the console's is unverified. |
| `gl_FrontFacing` | `bool [[front_facing]]` | `ps_system_inputs` "front"; converted to +-1.0 (`#PS_PedEntityMaskD`) |
| SIMD lane / width | `[[thread_index_in_simdgroup]]`, `[[threads_per_simdgroup]]` | section 1 |

`index.json` `inputs` lists every location the PS reads (including f16-half locations), `ps_interpolants` the
count. No barycentric, sample-ID or sample-mask input appears in the 150.

**Outputs [P][M]** (`struct main0_out`):

| Output | MSL | Notes |
|---|---|---|
| Colour | `float4 m_K [[color(N)]]` | always `float4` (168 outputs in the 150). `exp mrtK` writes the K-th declared SV_Target, not RT K (xlat:15033-15041). Compressed exports are unpacked from f16 pairs (xlat:15001-15017). Unexported components keep the previous value (`partial_color_outputs`). |
| Depth | `float gl_FragDepth [[depth(any)]]` | 6 PS (`writes_depth: true`). Only Z of an uncompressed `exp mrtz`; stencil, sample mask and alpha from mrtz are not modelled (xlat:14985-14999). |
| Discard | `discard_fragment()` | 77 PS; inline, no host state |

Integer render targets: the output is always `float4` holding the export's raw bits as floats, so a UINT/SINT RT
needs the output retyped or bit-cast (inferred gap; none of the 150 has one).

**Harness [H]** (harness:489-587): vertex function `vs_main` from `vs.metal`, one full-screen triangle
(`drawPrimitives(.triangle, 0, 3)`), each varying a fixed linear ramp (prepare:393-415), position z = 0.2 + 0.3x.
Colour targets 32x32 (`--size`): `rgba32Float` for float outputs, `rgba32Uint`/`rgba32Sint` for uint/int,
cleared to (0.25, 0.5, 0.75, 1) or 7; no blending. Depth `depth32Float` cleared to 0.5, compare `.always`, write on.
With no attachment it sets `renderTargetWidth/Height` explicitly. Only fragment bindings are reflected and bound.

---

## 8. Compile options

| Option | Value | Source | Layer |
|---|---|---|---|
| `languageVersion` | `.version3_1` (matches `--msl-version 30100`) | harness:151, prepare:455 | [M] |
| `mathMode` | **`.safe`** | harness:152 | [M], required |
| `SBX_BASE` | `0x<heap_base>ul` | harness:153 | [H] (ref.metal only) |
| `SBX_MASK` | `0x<window-1>ul` | harness:154 | [H] |
| `SBX_COV`, `SBX_READS`, `SBX_RD_SHIFT`, `SBX_CMPLOG`, `SBX_LOOP_CAP` | instrumentation switches | harness:155-157, prepare:48 | [H] |

Fast math is unsound for these shaders. The translator emulates RDNA2 float behaviour bit for bit: it declares
`SignedZeroInfNanPreserve` (32) and `DenormPreserve` (16) (index `float_controls`), tests NaN/Inf by bit pattern,
converts f16 with integer code and uses `precise::` functions (95 of the 150) plus explicit `fma` (127). Fast math
lets the compiler fold `x * 0`, drop NaN checks and reassociate. When our IR equivalence check used the `metal` CLI
default (fast math) it "proved" shaders equal that differ on NaN/Inf (HANDOVER.md section 4; resolve_survivors.py:71-73).
Offline compiles must pass `-fno-fast-math`.

Toolchain the 150 were checked with: Apple metal 32023.864, macOS 15.6.1, M1 Pro (Apple7). The SPIRV-Cross build is
not recorded; the set0 docstring names 1.4.363.0 as the version whose overlap behaviour it works around. Two shaders stall Apple's compiler and are excluded (HANDOVER.md section 9).

---

## 9. Per-shader metadata

Two files. `index.json` (written by `rdna2spirv.py`, xlat:18588-18700, plus `tools/shader_meta.py`) is the
translator's record; `meta.json` (prepare:469-487, 189-193) is a harness subset. A runtime should read `index.json`.

**`meta.json` fields** (the harness reads `stage`, `name`, `user_sgpr_reads`, `fssm`, `local_size`, `outputs`, the
`cov_*` fields, `cmp_ids`, `key_ids`):

| Field | Meaning | Layer |
|---|---|---|
| `name`, `stage` (`ps`/`cs`) | | [P] |
| `local_size`, `user_sgprs` | threadgroup size, CS user-SGPR count (workgroup IDs follow) | [P] |
| `user_sgpr_reads` | section 4 | [P] |
| `wave64`, `lds_bytes`, `push_constant_bytes` (defaults to 144 when absent) | | [P] |
| `fssm` | path of the FSSM blob to place at `code_base` | [P] |
| `outputs` | PS: `[{name, kind: color/depth/sample_mask, slot, type, width}]` parsed from `main0_out` (prepare:418-429) | [M] |
| `sandboxed_casts`, `cov_blocks`, `loop_guards`, `cov_guards`, `cov_guards_inverted`, `cov_buffer_index`, `cmp_ids`, `key_ids`, `cmp_ops` | instrumentation | [H] |

**`index.json` fields `meta.json` drops** that a runtime needs: `bindings`, `literal_samplers`, `f16_halves`,
`inputs`, `outputs` (RT indices), `ps_flat_interpolants`, `ps_fp16_interpolants`, `ps_interpolants`,
`ps_system_inputs`, `writes_depth`, `uses_code_base`, `subgroup_size`, `wave64_emulated`, `wave64_exchange`,
`wave64_graphics` (+ `_reasons`, `_idioms`), `control_flow`, `descriptor_access`, `descriptor_strides`,
`input_signature` (ISG1, the PSO-matching key), `fssm_sha256` / `src_sha256` (the PSO lookup key: the console PSO
carries DXBC with the FSSM inside, d3d12doc 3.6), `float_controls`, `capabilities`, `features`, `fma`,
`dynamic_format`, `fixed_format_codes`.

**Missing from both** (needed to create a Metal pipeline):
- the push-constant buffer index (reflect it, or use the section 2.1 rule);
- render-target pixel formats, sample count, blend, depth/stencil state, write masks (console PSO state);
- the vertex function and its output locations (no translated VS in the working set);
- the root-signature -> user-SGPR mapping for graphics (section 4);
- the console's PS input mapping (SPI_PS_INPUT_CNTL), assumed identity;
- the translator flags each shader was built with (`--wave64-emulate`, `--f16-interp-fallback`; inferred from fields).

---

## 10. Known gaps and unverified areas

**Stages and features not covered**
1. Vertex, mesh, geometry and hull/domain stages: no Metal path tested. The harness refuses anything but cs/ps
   (harness:591). VS with descriptor lookups fail to compile on Metal (`simd_broadcast_first` in a vertex function).
   GS (22 of 22 sampled) also fail; mesh shaders compile but need `--mesh-compute` or Metal mesh pipelines (TODO.md).
2. Tier C wave64 pixel shaders (32 of 100 PS) are not exact on simd32 (section 1).
3. Threadgroup memory: Apple7 allows 32 KiB; one CS in the set needs 33,868 bytes (TODO.md). RDNA2 allows 64 KiB.
4. One 144-byte push block per pipeline: VS+PS pairs with different user data or two code bases don't fit
   (d3d12doc 5.2).
5. Multisampled images (bindings 20-25), 1D storage (17-19) and r32ui 3D (7) appear in no shader of the 150.
6. Integer render targets, stencil export, sample-mask output: not modelled (section 7).
7. Ray tracing (64 shaders in the corpus) and the 50 tier D shaders: outside this pipeline.

**What the harness fakes (so these paths are unverified on Metal)**
1. **Texture classes:** one texture per declared type fills every element, typed by the declared `uint` array, so a
   lookup miss or a float FORMAT samples an `rgba32Uint` texture through the `texture2d<float>` alias. A real
   runtime with float textures at index `i` through the `uint`-declared array is untested.
2. **Same index, different textures:** a sampled read and a storage write of the same index hit different Metal
   textures (one per array), unlike Palmetto, which writes one image into all arrays of its dimension.
3. **Descriptors:** no T# or S# is ever decoded; contents, sizes, mips, swizzles and formats are synthetic.
   Hash entries exist only when a search variant adds them (random keys, not real T#s).
4. **Sampler:** one linear/repeat sampler for every index; no S# -> `MTLSamplerDescriptor` mapping exists.
5. **Title memory:** a 16 MiB window with `sbx()` wrapping replaces the arena; addresses that would fault for real
   are folded into the window.
6. **Push constants:** seeded; table pointers are made up (`0x100000 + t * 0x10000`).
7. **Pipeline state:** synthetic full-screen VS, `rgba32Float` targets, no blending, depth compare `.always`,
   32x32 targets, a 4-threadgroup dispatch.
8. **Loops:** capped at 4,096 iterations by the guard; capped runs are dropped as inconclusive.

**Unverified conventions**
- `gl_FragCoord.w` and pixel-centre conventions between console and Metal.
- The identity PS attribute mapping and `f16_halves` producer pairing (no translated VS paired with a PS on Metal).
- Root-signature -> user-SGPR layout for graphics (s8/s16 bases are a hypothesis; compute s4 is measured).
- Encoding textures of one class into an argument-buffer array declared with another (works on Apple7 as far as the
  harness shows; not checked under Metal API validation).
