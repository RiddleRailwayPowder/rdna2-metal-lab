// Differential Metal shader harness.
//
// Runs shader A and shader B (Metal source from prepare.py) on identical generated inputs and compares
// every output: render targets (pixel shaders), writable textures and the whole title-memory window.
//
//   harness --meta meta.json --a A.metal --b B.metal [--vs vs.metal] [--seed N] [--size 32]
//
// Batch mode (input search): --inputs FILE runs a JSON list of variants {"seed": N, "mem": [[word, value], ...],
// "push": [[dword, value], ...]} (seeded inputs, then the patches); --b none runs only A; --all runs every variant
// instead of stopping at the first difference; --reads 1 records which 16-byte chunks of title memory A touched.
// Each variant gets a row in "variants": hits (coverage ids), written outputs, output hashes, equal (with B).
// Prints one JSON object. status: ok | compile_a | compile_b | pipeline_a | pipeline_b | gpu_a | gpu_b |
// timeout_a | timeout_b | unsupported. With status ok, "equal" says whether every output matched bit for bit.
//
// Inputs (same for A and B):
//   title memory   64 MiB window (+1 MiB guard) of seeded values: small ints, aligned offsets, floats, specials
//   push constants s0-s31: table pointers into the window for ptr32 user SGPRs, values for the rest;
//                  dwords 32-33 heap base (the window's GPU address), 34-35 code base (where the FSSM blob is)
//   textures       seeded noise, mipmapped; one texture per (type, data type, access) shared by an array
//   samplers       linear, repeat
//   hash table     empty: every descriptor lookup falls back to index 0
//
// harness_tex (this copy): for every T#/S# a variant puts in the hash table, the element at its index gets a texture
// decoded from the T# (FORMAT class/layout via Palmetto's gfx10_formats.json, size, mips, layers, swizzle) and a
// sampler decoded from the S#; every other element keeps the default above. --formats PATH overrides the json path.
// Counters go to result["descriptors"]; per-index storage textures are compared as "texture:<member>[<index>]".
import Foundation
import Metal

// MARK: - arguments

var args: [String: String] = [:]
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        if k.hasPrefix("--") { args[String(k.dropFirst(2))] = it.next() ?? "" }
    }
}
func arg(_ k: String) -> String? { args[k] }

var result: [String: Any] = [:]

// Phase log: every step is announced on stderr with the elapsed time and its duration is added to
// result["phases_ms"]. A run that hangs or crashes still shows its last phase in stderr; every JSON result
// (also failures) names the phase it ended in (result["phase"]).
let startTime = Date()
var currentPhase = "start", phaseStart = Date()
var phases: [String: Double] = [:]
func log(_ msg: String) {
    FileHandle.standardError.write(String(format: "[harness +%8.3fs] %@\n", Date().timeIntervalSince(startTime), msg).data(using: .utf8)!)
}
func mark(_ next: String) {
    let ms = Date().timeIntervalSince(phaseStart) * 1000
    phases[currentPhase] = (phases[currentPhase] ?? 0) + ms
    log(String(format: "%@ done (%.1f ms) -> %@", currentPhase, ms, next))
    currentPhase = next; phaseStart = Date()
}
func finish(_ status: String, code: Int32 = 0) -> Never {
    let failedIn = currentPhase
    mark("finish")
    result["phase"] = failedIn
    result["phases_ms"] = phases.mapValues { ($0 * 10).rounded() / 10 }
    result["elapsed_ms"] = (Date().timeIntervalSince(startTime) * 10000).rounded() / 10
    if status != "ok" { log("FAILED: status \(status) in phase \(failedIn)") }
    result["status"] = status
    let data = try! JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    exit(code)
}

guard let metaPath = arg("meta"), let aPath = arg("a"), let bPath = arg("b") else {
    FileHandle.standardError.write("usage: harness --meta M --a A.metal --b B.metal [--vs VS] [--seeds 1,2,...] [--coverage 1]\n".data(using: .utf8)!)
    exit(2)
}
let meta = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: metaPath))) as! [String: Any]
let stage = meta["stage"] as? String ?? "?"
// Several seeds run in one process: A and B are compiled once. The run stops at the first seed that tells them apart.
let seedList: [UInt64] = (arg("seeds") ?? arg("seed") ?? "1").split(separator: ",").compactMap { UInt64($0) }
var seed: UInt64 = seedList.first ?? 1
var texSeed: UInt64 = seedList.first ?? 1   // texture contents; a variant can set it apart from the memory seed
let stepTimeout = Double(arg("compile-timeout") ?? "30") ?? 30
let rtSize = Int(arg("size") ?? "32") ?? 32
result["name"] = meta["name"]
result["stage"] = stage
result["seeds"] = seedList

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { finish("no_device", code: 1) }

// MARK: - seeded values

struct SplitMix { var s: UInt64
    mutating func next() -> UInt64 { s &+= 0x9E3779B97F4A7C15; var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) }
}
let specials: [UInt32] = [0, 0x3F80_0000, 0xBF80_0000, 0x3F00_0000, 0x4000_0000, 0x7FC0_0000, 0x7F80_0000, 0xFF80_0000,
                          0x7F7F_FFFF, 0x0000_0001, 0x8000_0000, 0xFFFF_FFFF, 1, 2, 4, 16, 64, 256]
func value(_ r: UInt64) -> UInt32 {
    let sel = r % 100, x = r >> 8
    switch sel {
    case 0..<30: return UInt32(x % 4096)
    case 30..<45: return UInt32(x % (1 << 22)) & ~15
    case 45..<85: return (Float(Double(x % 1_000_000) / 125_000.0 - 4.0)).bitPattern
    case 85..<95: return specials[Int(x % UInt64(specials.count))]
    default: return UInt32(truncatingIfNeeded: x)
    }
}

// MARK: - title memory

// Title-memory window: --window-mb (default 16). Generating and comparing it dominates the per-seed cost.
let window = (Int(arg("window-mb") ?? "16") ?? 16) << 20, guardBytes = 1 << 20
let codeVA: UInt32 = UInt32(window - (2 << 20))   // the FSSM blob sits 2 MiB below the window's end
let tableVA: UInt32 = 0x0010_0000
guard let memory = device.makeBuffer(length: window + guardBytes, options: .storageModeShared) else { finish("no_memory", code: 1) }
let heapBase = memory.gpuAddress
let totalWords = (window + guardBytes) / 4
// pristine: the current variant's title memory (seeded base + "mem" patches). Raw allocations, so no hidden
// copy-on-write copies of 17 MiB.
let pristine = UnsafeMutablePointer<UInt32>.allocate(capacity: totalWords)
var push = [UInt32](repeating: 0, count: 64)
let fssmBlob: Data? = (meta["fssm"] as? String).flatMap { FileManager.default.contents(atPath: $0) }
result["fssm_bytes"] = fssmBlob?.count ?? 0

// Seeded bases, a few seeds kept (inputs.json alternates seeds 1, 2, 3).
let maxCachedSeeds = 4
var baseCache: [UInt64: UnsafeMutablePointer<UInt32>] = [:]
var baseOrder: [UInt64] = []   // least recently used first
func generateBase(_ s: UInt64, into p: UnsafeMutablePointer<UInt32>) {
    // SplitMix is counter based: word i is value(mix(s0 + (i + 1) * gamma)), so chunks are exact.
    let s0 = s &* 0x1000193, n = totalWords, chunks = 4
    DispatchQueue.concurrentPerform(iterations: chunks) { c in
        let lo = n * c / chunks, hi = n * (c + 1) / chunks
        var rng = SplitMix(s: s0 &+ UInt64(lo) &* 0x9E3779B97F4A7C15)
        for i in lo..<hi { p[i] = value(rng.next()) }
    }
    if let blob = fssmBlob {
        blob.withUnsafeBytes { raw in
            let n = min(blob.count, window - Int(codeVA))
            UnsafeMutableRawPointer(p).advanced(by: Int(codeVA)).copyMemory(from: raw.baseAddress!, byteCount: n)
        }
    }
}
func basePointer(_ s: UInt64) -> UnsafeMutablePointer<UInt32> {
    if let p = baseCache[s] { baseOrder.removeAll { $0 == s }; baseOrder.append(s); return p }
    let p: UnsafeMutablePointer<UInt32>
    if baseCache.count >= maxCachedSeeds { let old = baseOrder.removeFirst(); p = baseCache.removeValue(forKey: old)! }
    else { p = UnsafeMutablePointer<UInt32>.allocate(capacity: totalWords) }
    generateBase(s, into: p)
    baseCache[s] = p; baseOrder.append(s)
    return p
}

// MARK: - push constants

func setupPush() {
    var rng = SplitMix(s: seed &* 0x5bd1e995)
    for i in 0..<32 { push[i] = value(rng.next()) }
    var t: UInt32 = 0
    for r in (meta["user_sgpr_reads"] as? [[String: Any]]) ?? [] {
        guard let kind = r["kind"] as? String, let s = r["sgpr"] as? Int, s < 32 else { continue }
        if kind.hasPrefix("ptr") && kind.hasSuffix("_lo") { push[s] = tableVA + t * 0x1_0000; t += 1 }
        else if kind.hasPrefix("ptr") && kind.hasSuffix("_hi") { push[s] = 0 }
    }
    push[32] = UInt32(truncatingIfNeeded: heapBase); push[33] = UInt32(truncatingIfNeeded: heapBase >> 32)
    push[34] = codeVA; push[35] = 0
}

// MARK: - compile

func compile(_ path: String, coverage: Bool = false) -> MTLLibrary? {
    guard let src = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
    let o = MTLCompileOptions()
    o.languageVersion = .version3_1
    o.mathMode = .safe
    var macros: [String: NSObject] = ["SBX_BASE": NSString(string: "0x\(String(heapBase, radix: 16))ul"),
                                      "SBX_MASK": NSString(string: "0x\(String(window - 1, radix: 16))ul")]
    if coverage { macros["SBX_COV"] = NSNumber(value: 1) }
    if coverage && wantReads { macros["SBX_READS"] = NSNumber(value: 1); macros["SBX_RD_SHIFT"] = NSNumber(value: readShift) }
    if coverage && wantCmplog { macros["SBX_CMPLOG"] = NSNumber(value: 1) }
    o.preprocessorMacros = macros
    var lib: MTLLibrary?, err: Error?
    let sem = DispatchSemaphore(value: 0)
    device.makeLibrary(source: src, options: o) { l, e in lib = l; err = e; sem.signal() }
    if sem.wait(timeout: .now() + stepTimeout) == .timedOut { result["compile_seconds"] = stepTimeout; return nil }
    if lib == nil { result["compile_error"] = String(String(describing: err).prefix(6000)); log("compile error: \(String(describing: err).prefix(2000))") }
    if let lib { libSource[ObjectIdentifier(lib)] = src }
    return lib
}
var compileTimedOut: Bool { result["compile_seconds"] != nil }

// Pipelines are built once per library and reused for every seed.
var computePipelines: [ObjectIdentifier: (MTLComputePipelineState, MTLComputePipelineReflection?)] = [:]
var renderPipelines: [ObjectIdentifier: (MTLRenderPipelineState, MTLRenderPipelineReflection?)] = [:]

// MARK: - resources

struct TexKey: Hashable { let type: UInt; let data: UInt; let write: Bool; let depth: Bool; let member: Int }
// Textures live for the whole process (one object per key, shared by A and B and every variant); their contents
// are refilled when the texture seed changes, from cached bytes per (key, seed) when that seed was seen before.
final class TexEntry {
    let tex: MTLTexture, key: TexKey, depth: Bool, write: Bool
    var seed: UInt64? = nil
    var saved: [(Int, Int, Data)] = []   // (slice, level, bytes) for the current seed
    init(tex: MTLTexture, key: TexKey, depth: Bool, write: Bool) { self.tex = tex; self.key = key; self.depth = depth; self.write = write }
}
struct SeedKey: Hashable { let key: TexKey; let seed: UInt64 }
var texCache: [TexKey: TexEntry] = [:]
var texBytesCache: [SeedKey: [(Int, Int, Data)]] = [:]
var bindSeen: [TexKey: MTLTexture] = [:]           // keys met in the current bind() call (the old per-run `textures`)
var bindEntries: [TexEntry] = []
var bindWritable: [(String, MTLTexture)] = []
var writable: [(String, MTLTexture)] = []
var writableSaved: [ObjectIdentifier: TexEntry] = [:]
var initialBytes: [String: Data] = [:]   // output name -> bytes before the shader ran

func pixelFormat(_ data: MTLDataType, write: Bool, depth: Bool) -> MTLPixelFormat {
    if depth { return .depth32Float }
    switch data {
    case .uint, .ushort, .uchar: return write ? .r32Uint : .rgba32Uint
    case .int, .short, .char: return write ? .r32Sint : .rgba32Sint
    case .half: return .rgba16Float
    default: return write ? .rgba32Float : .rgba16Float
    }
}

func fillBytes(_ format: MTLPixelFormat, count: Int, rng: inout SplitMix) -> Data {
    var d = Data(count: count)
    d.withUnsafeMutableBytes { (p: UnsafeMutableRawBufferPointer) in
        switch format {
        case .rgba16Float:
            let h = p.bindMemory(to: Float16.self)
            for i in 0..<h.count { let r = rng.next(); h[i] = r % 20 == 0 ? Float16(Double(r >> 8 % 600) / 100 - 2) : Float16(Double((r >> 8) % 1000) / 1000) }
        case .rgba32Float, .depth32Float:
            let f = p.bindMemory(to: Float.self)
            for i in 0..<f.count { f[i] = Float(Double((rng.next() >> 8) % 1000) / 1000) }
        default:
            // integer textures: the shader may reinterpret their bits as floats (Palmetto's untyped path), so each
            // texture picks a style from its seed: the title-memory value mix, float bits in [0, 1), small ints
            let u = p.bindMemory(to: UInt32.self)
            let style = rng.next() % 3
            for i in 0..<u.count {
                let r = rng.next()
                switch style {
                case 0: u[i] = value(r)
                case 1: u[i] = Float(Double((r >> 8) % 1000) / 1000).bitPattern
                default: u[i] = UInt32((r >> 8) % 1024)
                }
            }
        }
    }
    return d
}

func makeTexture(_ type: MTLTextureType, _ data: MTLDataType, write: Bool, depth: Bool, member: Int, name: String) -> MTLTexture? {
    let key = TexKey(type: type.rawValue, data: data.rawValue, write: write, depth: depth, member: write ? member : -1)
    if let t = bindSeen[key] { return t }
    if let e = texCache[key] {
        bindSeen[key] = e.tex; bindEntries.append(e)
        if write { bindWritable.append((name, e.tex)) }
        return e.tex
    }
    let d = MTLTextureDescriptor()
    var t2 = type
    if t2 == .typeCube { t2 = .type2DArray }
    d.textureType = t2
    d.pixelFormat = pixelFormat(data, write: write, depth: depth)
    d.width = 64; d.height = t2 == .type1D ? 1 : 64
    d.depth = t2 == .type3D ? 16 : 1
    d.arrayLength = t2 == .type2DArray ? 6 : 1
    let mipOK = !write && !depth && (t2 == .type2D || t2 == .type2DArray)
    d.mipmapLevelCount = mipOK ? 7 : 1
    d.usage = write ? [.shaderRead, .shaderWrite] : [.shaderRead]
    d.storageMode = depth ? .private : .shared
    if depth { d.mipmapLevelCount = 1 }
    guard let tex = device.makeTexture(descriptor: d) else { return nil }
    let e = TexEntry(tex: tex, key: key, depth: depth, write: write)
    texCache[key] = e
    bindSeen[key] = tex; bindEntries.append(e)
    if write { bindWritable.append((name, tex)) }
    return tex
}

// Gives the texture the contents the old harness generated for (key, texSeed). Writable textures only get their
// saved bytes here: resetInputs writes them before every run.
func ensureFilled(_ e: TexEntry) {
    if e.depth || e.seed == texSeed { return }
    let tex = e.tex, key = e.key
    let sk = SeedKey(key: key, seed: texSeed)
    var saved: [(Int, Int, Data)]
    let bpp = tex.pixelFormat == .rgba16Float ? 8 : (tex.pixelFormat == .rgba32Float || tex.pixelFormat == .rgba32Uint || tex.pixelFormat == .rgba32Sint ? 16 : 4)
    if let s = texBytesCache[sk] { saved = s } else {
        if texBytesCache.count >= 512 { texBytesCache.removeAll() }   // bound memory for long batches
        // Swift's hashValue changes per process: derive the texture seed from the key's fields instead.
        let keyBits = UInt64(key.type) &* 31 &+ UInt64(key.data) &* 1009 &+ (key.write ? 7 : 0) &+ (key.depth ? 13 : 0) &+ UInt64(bitPattern: Int64(key.member)) &* 65537
        var rng = SplitMix(s: texSeed &+ keyBits &+ 77)
        saved = []
        for slice in 0..<tex.arrayLength {
            for level in 0..<tex.mipmapLevelCount {
                let w = max(1, tex.width >> level), h = max(1, tex.height >> level), z = max(1, tex.depth >> level)
                saved.append((slice, level, fillBytes(tex.pixelFormat, count: w * h * z * bpp, rng: &rng)))
            }
        }
        texBytesCache[sk] = saved
    }
    if !e.write {
        for (slice, level, bytes) in saved {
            let w = max(1, tex.width >> level), h = max(1, tex.height >> level), z = max(1, tex.depth >> level)
            bytes.withUnsafeBytes { tex.replace(region: MTLRegionMake3D(0, 0, 0, w, h, z), mipmapLevel: level, slice: slice,
                                                withBytes: $0.baseAddress!, bytesPerRow: w * bpp, bytesPerImage: w * h * bpp) }
        }
    }
    e.saved = saved
    e.seed = texSeed
}

let samplerState: MTLSamplerState = {
    let s = MTLSamplerDescriptor()
    s.minFilter = .linear; s.magFilter = .linear; s.mipFilter = .linear
    s.sAddressMode = .repeat; s.tAddressMode = .repeat; s.rAddressMode = .repeat
    s.supportArgumentBuffers = true
    return device.makeSamplerState(descriptor: s)!
}()
// Descriptor hash table (Palmetto): word 0 = slot mask, slot i at words 10i+1...10i+9 = 8 key words (T#/S#) + index.
// Empty unless a variant gives entries ("hash": [[k0..k7, index], ...]); a miss gives index 0.
let hashSlots = 1024
let hashTable = device.makeBuffer(length: (hashSlots * 10 + 16) * 4, options: .storageModeShared)!
func fillHashTable(_ entries: [[UInt32]]) {
    let w = hashTable.contents().bindMemory(to: UInt32.self, capacity: hashSlots * 10 + 16)
    memset(hashTable.contents(), 0, (hashSlots * 10 + 16) * 4)
    w[0] = UInt32(hashSlots - 1)
    for e in entries where e.count == 9 {
        var h: UInt32 = 2166136261
        for k in 0..<8 { h = (h ^ e[k]) &* 16777619 }
        for probe in 0..<8 {
            let slot = Int((h &+ UInt32(probe)) & UInt32(hashSlots - 1))
            let base = slot * 10
            if w[base + 9] == 0 || (0..<8).allSatisfy({ w[base + 1 + $0] == e[$0] }) {
                for k in 0..<8 { w[base + 1 + k] = e[k] }
                w[base + 9] = e[8]
                break
            }
        }
    }
}
var lastKeys: [[UInt32]] = []
// Coverage buffer (prepare.py instrumentation, [[buffer(29)]]): [0] loop guard tripped, [i] block i ran.
let covIndex = (meta["cov_buffer_index"] as? Int) ?? 29
let covWords = 1 << 18   // [0, 32768) coverage ids, [32768, 65536) read bitmap, [65536, ...) comparison log
let coverage = device.makeBuffer(length: covWords * 4, options: .storageModeShared)!
func readCoverage(tag: String) {
    let w = coverage.contents().bindMemory(to: UInt32.self, capacity: covWords)
    result["loop_guard_\(tag)"] = ((result["loop_guard_\(tag)"] as? Bool) ?? false) || w[0] != 0
    if tag == "a" { lastLoopA = w[0] != 0 } else { lastLoopB = w[0] != 0 }
    if tag == "a" && wantCoverage {
        // ids 1...blocks are blocks; blocks+1...blocks+guards are lane guards (prepare.py sbx_sel/sbx_nsel)
        let blocks = (meta["cov_blocks"] as? Int) ?? 0
        let guards = (meta["cov_guards"] as? Int) ?? 0
        let before = Set((result["cov_hits"] as? [Int]) ?? [])
        result["cov_hits"] = before.union((1..<max(1, min(blocks + guards + 1, covWords))).filter { w[$0] != 0 }).sorted()
        result["cov_blocks"] = blocks
        result["cov_guards"] = guards
        lastHits = (1..<max(1, min(blocks + guards + 1, covWords))).filter { w[$0] != 0 }
        if wantReads {
            var r: [Int] = []
            for k in 0..<32768 where w[readBase + k] != 0 {
                let x = w[readBase + k]
                for b in 0..<32 where (x >> UInt32(b)) & 1 != 0 { r.append(k * 32 + b) }
            }
            lastReads = r
        }
        if wantCmplog {
            let n = min((meta["cmp_ids"] as? Int) ?? 0, (covWords - 65536) / 3)
            var c: [[UInt32]] = []
            for i in 0..<n {
                let base: Int = 65536 + 3 * i
                if w[base] != 0 { c.append([UInt32(i), w[base + 1], w[base + 2]]) }
            }
            lastCmp = c
            var keys: [[UInt32]] = []
            let nk = min((meta["key_ids"] as? Int) ?? 0, (covWords - 200000) / 9)
            for i in 0..<nk {
                let base: Int = 200000 + 9 * i
                if w[base] != 0 { keys.append([UInt32(i)] + (1...8).map { w[base + $0] }) }
            }
            lastKeys = keys
        }
    }
}
let wantCoverage = arg("coverage") != nil
// Read tracking (prepare.py SBX_READS): words covWords/2... of the coverage buffer are a bitmap of title memory in
// chunks of 1 << readShift bytes (16 bytes for a 16 MiB window).
let wantReads = arg("reads") != nil
let readBase = 32768
let readShift: Int = { var s = 0; while (window >> s) > 32768 * 32 { s += 1 }; return s }()
let wantCmplog = arg("cmplog") != nil
var lastCmp: [[UInt32]] = []
var lastHits: [Int] = [], lastReads: [Int] = []
var lastLoopA = false, lastLoopB = false   // loop guard of the latest run of A / B (result["loop_guard_*"] is cumulative)
memset(hashTable.contents(), 0, 4096)

struct Bound { var argBuffer: MTLBuffer?; var argIndex = -1; var pushIndex = -1; var used: [(MTLResource, MTLResourceUsage)] = []; var unsupported: [String] = []
               var entries: [TexEntry] = []; var writable: [(String, MTLTexture)] = []
               var encoder: MTLArgumentEncoder? = nil; var members: [MemberInfo] = [] }

// Argument buffers point at process-lifetime textures, so they are built once per library.
var boundCache: [ObjectIdentifier: Bound] = [:]
func boundFor(_ lib: MTLLibrary, _ bindings: [MTLBinding], function: MTLFunction) -> Bound {
    if let b = boundCache[ObjectIdentifier(lib)] { return b }
    var b = bind(bindings, function: function)
    b.entries = bindEntries; b.writable = bindWritable
    if b.unsupported.isEmpty { boundCache[ObjectIdentifier(lib)] = b }
    return b
}
// Per run: the old harness's per-run texture state (writable list in bind order, contents for texSeed).
func prepareTextures(_ b: Bound) {
    for e in b.entries { ensureFilled(e) }
    writable = b.writable
    writableSaved = [:]
    for e in b.entries where e.write { writableSaved[ObjectIdentifier(e.tex)] = e }
}

var fnCache: [ObjectIdentifier: [String: MTLFunction]] = [:]
func function(_ lib: MTLLibrary, _ name: String) -> MTLFunction? {
    if let f = fnCache[ObjectIdentifier(lib)]?[name] { return f }
    let f = lib.makeFunction(name: name)
    if let f { fnCache[ObjectIdentifier(lib), default: [:]][name] = f }
    return f
}

func bind(_ bindings: [MTLBinding], function: MTLFunction) -> Bound {
    bindSeen = [:]; bindEntries = []; bindWritable = []
    var b = Bound()
    for binding in bindings {
        guard let buf = binding as? MTLBufferBinding else { continue }
        if buf.index == covIndex { continue }          // the coverage buffer, bound separately
        let members = buf.bufferPointerType?.elementStructType()?.members ?? []
        let isArgs = members.contains { m in
            if [.texture, .sampler, .pointer].contains(m.dataType) { return true }
            if m.dataType == .array, let a = m.arrayType() { return [.texture, .sampler, .pointer].contains(a.elementType) }
            return false
        }
        if !isArgs {
            if b.pushIndex >= 0 && b.pushIndex != buf.index { b.unsupported.append("two push-constant candidates: buffers \(b.pushIndex) and \(buf.index)") }
            b.pushIndex = buf.index; continue
        }
        let enc = function.makeArgumentEncoder(bufferIndex: buf.index)
        let ab = device.makeBuffer(length: max(enc.encodedLength, 16), options: .storageModeShared)!
        enc.setArgumentBuffer(ab, offset: 0)
        for m in members {
            var count = 1, dtype = m.dataType
            var tref = m.textureReferenceType()
            if dtype == .array, let a = m.arrayType() {
                count = a.arrayLength; dtype = a.elementType
                tref = a.elementTextureReferenceType()
            }
            switch dtype {
            case .texture:
                guard let tr = tref else { continue }
                let write = tr.access != .readOnly
                guard let tex = makeTexture(tr.textureType, tr.textureDataType, write: write, depth: tr.isDepthTexture,
                                            member: m.argumentIndex, name: m.name) else { b.unsupported.append("texture \(m.name)"); continue }
                enc.setTextures([MTLTexture?](repeating: tex, count: count), range: m.argumentIndex..<(m.argumentIndex + count))
                b.used.append((tex, write ? [.read, .write] : .read))
                b.members.append(MemberInfo(name: m.name, id: m.argumentIndex, count: count, sampler: false, type: tr.textureType,
                                            cls: texClass(tr.textureDataType), access: tr.access, defaultTex: tex))
            case .sampler:
                enc.setSamplerStates([MTLSamplerState?](repeating: samplerState, count: count), range: m.argumentIndex..<(m.argumentIndex + count))
                b.members.append(MemberInfo(name: m.name, id: m.argumentIndex, count: count, sampler: true, type: .type2D,
                                            cls: .float, access: .readOnly, defaultTex: nil))
            case .pointer:
                enc.setBuffer(hashTable, offset: 0, index: m.argumentIndex)
                b.used.append((hashTable, .read))
            default:
                b.unsupported.append("member \(m.name) type \(dtype.rawValue)")
            }
        }
        b.argBuffer = ab; b.argIndex = buf.index
        b.encoder = enc
    }
    return b
}

func resetInputs() {
    memory.contents().copyMemory(from: pristine, byteCount: totalWords * 4)
    for (_, t) in writable {
        for (slice, level, bytes) in writableSaved[ObjectIdentifier(t)]?.saved ?? [] {
            let w = max(1, t.width >> level), h = max(1, t.height >> level), z = max(1, t.depth >> level)
            let bpp = bytes.count / (w * h * z)
            bytes.withUnsafeBytes { t.replace(region: MTLRegionMake3D(0, 0, 0, w, h, z), mipmapLevel: level, slice: slice,
                                              withBytes: $0.baseAddress!, bytesPerRow: w * bpp, bytesPerImage: w * h * bpp) }
        }
    }
}

func textureBytes(_ t: MTLTexture) -> Data {
    let bpp = bytesPerPixel(t.pixelFormat)
    var all = Data()
    for slice in 0..<t.arrayLength {
        let n = t.width * t.height * t.depth * bpp
        var d = Data(count: n)
        d.withUnsafeMutableBytes { t.getBytes($0.baseAddress!, bytesPerRow: t.width * bpp, bytesPerImage: t.width * t.height * bpp,
                                             from: MTLRegionMake3D(0, 0, 0, t.width, t.height, t.depth), mipmapLevel: 0, slice: slice) }
        all.append(d)
    }
    return all
}

// MARK: - console descriptors (T#, S#) -> Metal textures and samplers

enum TexClass: String { case float, uint, sint }
func texClass(_ d: MTLDataType) -> TexClass {
    switch d {
    case .uint, .ushort, .uchar: return .uint
    case .int, .short, .char: return .sint
    default: return .float
    }
}
struct MemberInfo { let name: String; let id: Int; let count: Int; let sampler: Bool; let type: MTLTextureType; let cls: TexClass
                    let access: MTLBindingAccess; let defaultTex: MTLTexture? }
var libSource: [ObjectIdentifier: String] = [:]

// GFX10_FORMAT code -> name, from Palmetto's table (tools/gfx10_formats.json). A missing file leaves every code unknown.
let formatsPath: String = arg("formats") ?? URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    .deletingLastPathComponent().appendingPathComponent("../../upstream/palmetto/tools/gfx10_formats.json").standardized.path
let fmtNames: [UInt32: String] = {
    guard let d = FileManager.default.contents(atPath: formatsPath),
          let j = try? JSONSerialization.jsonObject(with: d) as? [String: String] else { return [:] }
    var m: [UInt32: String] = [:]
    for (k, v) in j { if let c = UInt32(k) { m[c] = v.replacingOccurrences(of: "GFX10_FORMAT_", with: "").trimmingCharacters(in: .whitespaces) } }
    return m
}()

// How a texture's bytes are generated. raw: any bits (UNORM/SNORM/sRGB/packed/BC); half/float32: finite values plus the
// special-value mix; int: the per-texture integer styles, truncated to the channel width; scaled: integer values stored
// as floats (Metal has no USCALED/SSCALED formats).
enum Fill { case raw, half, float32, int(Int), scaledU(Int), scaledS(Int) }
struct FmtInfo { let fmt: MTLPixelFormat; let cls: TexClass; let fill: Fill; let channels: Int; let metalChannels: Int; let exact: Bool }

// The shader's rule (rdna2spirv.py, format tables): a name ending _SINT reads through the int view, _UINT through the
// uint view, everything else through the float view. Checked: rebuilds both 512-bit masks of the generated MSL exactly.
func formatClass(_ name: String) -> TexClass { name.hasSuffix("_SINT") ? .sint : (name.hasSuffix("_UINT") ? .uint : .float) }
func nameChannels(_ name: String) -> Int {
    let parts = name.split(separator: "_")
    let n = parts.prefix { $0.allSatisfy { $0.isNumber } || $0.contains("E") && $0.first!.isNumber }.count
    return n == 0 ? 4 : min(n, 4)
}
let exactFormats: [UInt32: (MTLPixelFormat, Fill, Int, Int)] = [   // code: (Metal format, fill, channels, Metal channels)
    1: (.r8Unorm, .raw, 1, 1), 2: (.r8Snorm, .raw, 1, 1), 3: (.r16Float, .scaledU(8), 1, 1), 4: (.r16Float, .scaledS(8), 1, 1),
    5: (.r8Uint, .int(8), 1, 1), 6: (.r8Sint, .int(8), 1, 1), 7: (.r16Unorm, .raw, 1, 1), 8: (.r16Snorm, .raw, 1, 1),
    9: (.r32Float, .scaledU(16), 1, 1), 10: (.r32Float, .scaledS(16), 1, 1), 11: (.r16Uint, .int(16), 1, 1), 12: (.r16Sint, .int(16), 1, 1),
    13: (.r16Float, .half, 1, 1), 14: (.rg8Unorm, .raw, 2, 2), 15: (.rg8Snorm, .raw, 2, 2), 16: (.rg16Float, .scaledU(8), 2, 2),
    17: (.rg16Float, .scaledS(8), 2, 2), 18: (.rg8Uint, .int(8), 2, 2), 19: (.rg8Sint, .int(8), 2, 2), 20: (.r32Uint, .int(32), 1, 1),
    21: (.r32Sint, .int(32), 1, 1), 22: (.r32Float, .float32, 1, 1), 23: (.rg16Unorm, .raw, 2, 2), 24: (.rg16Snorm, .raw, 2, 2),
    25: (.rg32Float, .scaledU(16), 2, 2), 26: (.rg32Float, .scaledS(16), 2, 2), 27: (.rg16Uint, .int(16), 2, 2), 28: (.rg16Sint, .int(16), 2, 2),
    29: (.rg16Float, .half, 2, 2), 36: (.rg11b10Float, .raw, 3, 3), 50: (.rgb10a2Unorm, .raw, 4, 4), 54: (.rgb10a2Uint, .raw, 4, 4),
    56: (.rgba8Unorm, .raw, 4, 4), 57: (.rgba8Snorm, .raw, 4, 4), 58: (.rgba16Float, .scaledU(8), 4, 4), 59: (.rgba16Float, .scaledS(8), 4, 4),
    60: (.rgba8Uint, .int(8), 4, 4), 61: (.rgba8Sint, .int(8), 4, 4), 62: (.rg32Uint, .int(32), 2, 2), 63: (.rg32Sint, .int(32), 2, 2),
    64: (.rg32Float, .float32, 2, 2), 65: (.rgba16Unorm, .raw, 4, 4), 66: (.rgba16Snorm, .raw, 4, 4), 67: (.rgba32Float, .scaledU(16), 4, 4),
    68: (.rgba32Float, .scaledS(16), 4, 4), 69: (.rgba16Uint, .int(16), 4, 4), 70: (.rgba16Sint, .int(16), 4, 4), 71: (.rgba16Float, .half, 4, 4),
    72: (.rgba32Uint, .int(32), 3, 4), 73: (.rgba32Sint, .int(32), 3, 4), 74: (.rgba32Float, .float32, 3, 4),
    75: (.rgba32Uint, .int(32), 4, 4), 76: (.rgba32Sint, .int(32), 4, 4), 77: (.rgba32Float, .float32, 4, 4),
    128: (.r8Unorm_srgb, .raw, 1, 1), 129: (.rg8Unorm_srgb, .raw, 2, 2), 130: (.rgba8Unorm_srgb, .raw, 4, 4),
    132: (.rgb9e5Float, .raw, 3, 3), 140: (.r32Float, .float32, 1, 1),
    169: (.bc1_rgba, .raw, 4, 4), 170: (.bc1_rgba_srgb, .raw, 4, 4), 171: (.bc2_rgba, .raw, 4, 4), 172: (.bc2_rgba_srgb, .raw, 4, 4),
    173: (.bc3_rgba, .raw, 4, 4), 174: (.bc3_rgba_srgb, .raw, 4, 4), 175: (.bc4_rUnorm, .raw, 1, 1), 176: (.bc4_rSnorm, .raw, 1, 1),
    177: (.bc5_rgUnorm, .raw, 2, 2), 178: (.bc5_rgSnorm, .raw, 2, 2), 179: (.bc6H_rgbuFloat, .raw, 3, 3), 180: (.bc6H_rgbFloat, .raw, 3, 3),
    181: (.bc7_rgbaUnorm, .raw, 4, 4), 182: (.bc7_rgbaUnorm_srgb, .raw, 4, 4),
]
func genericFormat(_ cls: TexClass, channels: Int) -> FmtInfo {
    switch cls {
    case .uint: return FmtInfo(fmt: .rgba32Uint, cls: .uint, fill: .int(32), channels: channels, metalChannels: 4, exact: false)
    case .sint: return FmtInfo(fmt: .rgba32Sint, cls: .sint, fill: .int(32), channels: channels, metalChannels: 4, exact: false)
    case .float: return FmtInfo(fmt: .rgba16Float, cls: .float, fill: .half, channels: channels, metalChannels: 4, exact: false)
    }
}
// nil: code 0 (INVALID) or not in the json; the entry keeps today's binding.
func formatInfo(_ code: UInt32) -> FmtInfo? {
    guard code != 0, let name = fmtNames[code] else { return nil }
    let cls = formatClass(name)
    if let (f, fill, ch, mch) = exactFormats[code], !(isBC(f) && !device.supportsBCTextureCompression) {
        return FmtInfo(fmt: f, cls: cls, fill: fill, channels: ch, metalChannels: mch, exact: true)
    }
    return genericFormat(cls, channels: nameChannels(name))
}
func isBC(_ f: MTLPixelFormat) -> Bool { f.rawValue >= MTLPixelFormat.bc1_rgba.rawValue && f.rawValue <= MTLPixelFormat.bc7_rgbaUnorm_srgb.rawValue }
func bcBlockBytes(_ f: MTLPixelFormat) -> Int {
    switch f { case .bc1_rgba, .bc1_rgba_srgb, .bc4_rUnorm, .bc4_rSnorm: return 8; default: return 16 }
}
func bytesPerPixel(_ f: MTLPixelFormat) -> Int {
    switch f {
    case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint, .r8Unorm_srgb: return 1
    case .r16Unorm, .r16Snorm, .r16Uint, .r16Sint, .r16Float, .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint, .rg8Unorm_srgb: return 2
    case .rg32Uint, .rg32Sint, .rg32Float, .rgba16Unorm, .rgba16Snorm, .rgba16Uint, .rgba16Sint, .rgba16Float: return 8
    case .rgba32Uint, .rgba32Sint, .rgba32Float: return 16
    default: return 4
    }
}
// Formats a storage (write / read_write) texture may have on Apple7 without a capability query.
let writableFormats: Set<MTLPixelFormat> = [.r8Unorm, .r8Snorm, .r8Uint, .r8Sint, .r16Unorm, .r16Snorm, .r16Uint, .r16Sint, .r16Float,
    .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint, .r32Uint, .r32Sint, .r32Float, .rg16Unorm, .rg16Snorm, .rg16Uint, .rg16Sint, .rg16Float,
    .rg11b10Float, .rgb10a2Unorm, .rgb10a2Uint, .rgba8Unorm, .rgba8Snorm, .rgba8Uint, .rgba8Sint, .rg32Uint, .rg32Sint, .rg32Float,
    .rgba16Unorm, .rgba16Snorm, .rgba16Uint, .rgba16Sint, .rgba16Float, .rgba32Uint, .rgba32Sint, .rgba32Float]

// Counters for result["descriptors"] (counted on A's runs only, once per variant).
var dcount: [String: Int] = [:]
var dformats: [String: [String: Int]] = ["exact": [:], "emulated": [:], "approx": [:], "unknown": [:]]
var countingOn = false
func dc(_ k: String, _ n: Int = 1) { if countingOn { dcount[k, default: 0] += n } }
func dfmt(_ kind: String, _ code: UInt32) { if countingOn { dformats[kind]![String(code), default: 0] += 1 } }

// Classes each argument-buffer member can be read through: its declared type plus the reinterpret_cast aliases
// SPIRV-Cross emits for the overlapping bindings (array<texture2d<int>, 4096> of m_8, say).
var aliasCache: [ObjectIdentifier: [String: Set<TexClass>]] = [:]
func aliases(_ lib: MTLLibrary) -> [String: Set<TexClass>] {
    if let a = aliasCache[ObjectIdentifier(lib)] { return a }
    var a: [String: Set<TexClass>] = [:]
    let src = libSource[ObjectIdentifier(lib)] ?? ""
    let re = try! NSRegularExpression(pattern: #"array<texture\w+<(float|half|int|uint|short|ushort)[^>]*>, \d+> &>\(spvDescriptorSet0\.(\w+)\)"#)
    for m in re.matches(in: src, range: NSRange(src.startIndex..., in: src)) {
        let t = String(src[Range(m.range(at: 1), in: src)!]), member = String(src[Range(m.range(at: 2), in: src)!])
        a[member, default: []].insert(t == "int" || t == "short" ? .sint : (t == "uint" || t == "ushort" ? .uint : .float))
    }
    aliasCache[ObjectIdentifier(lib)] = a
    return a
}

func floatValue(_ r: UInt64) -> Float {
    let sel = r % 100, x = r >> 8
    return sel < 88 ? Float(Double(x % 1_000_000) / 125_000.0 - 4.0) : Float(bitPattern: specials[Int(x % UInt64(specials.count))])
}
// One mip level / slice of a decoded texture. `style` is the texture's integer style (as the default textures pick it).
func fillDecoded(_ f: FmtInfo, count: Int, style: UInt64, rng: inout SplitMix) -> Data {
    var d = Data(count: count)
    d.withUnsafeMutableBytes { (p: UnsafeMutableRawBufferPointer) in
        let half = [.r16Float, .rg16Float, .rgba16Float].contains(f.fmt)
        switch f.fill {
        case .raw:
            let b = p.bindMemory(to: UInt8.self)
            var i = 0
            while i < b.count { var r = rng.next(); for _ in 0..<8 where i < b.count { b[i] = UInt8(truncatingIfNeeded: r); r >>= 8; i += 1 } }
        case .half:
            let h = p.bindMemory(to: Float16.self)
            for i in 0..<h.count { h[i] = Float16(floatValue(rng.next())) }
        case .float32:
            let w = p.bindMemory(to: Float.self)
            for i in 0..<w.count { w[i] = floatValue(rng.next()) }
        case .int(let bits):
            let mask: UInt64 = bits == 32 ? 0xFFFF_FFFF : (1 << UInt64(bits)) - 1
            func v(_ r: UInt64) -> UInt64 {
                switch style {
                case 0: return UInt64(value(r)) & mask
                case 1: return bits == 32 ? UInt64(Float(Double((r >> 8) % 1000) / 1000).bitPattern) : ((r >> 8) % 1024) & mask
                default: return ((r >> 8) % 1024) & mask
                }
            }
            switch bits {
            case 8: let b = p.bindMemory(to: UInt8.self); for i in 0..<b.count { b[i] = UInt8(v(rng.next())) }
            case 16: let b = p.bindMemory(to: UInt16.self); for i in 0..<b.count { b[i] = UInt16(v(rng.next())) }
            default: let b = p.bindMemory(to: UInt32.self); for i in 0..<b.count { b[i] = UInt32(v(rng.next())) }
            }
        case .scaledU(let bits), .scaledS(let bits):
            let signed: Bool = { if case .scaledS = f.fill { return true }; return false }()
            func v(_ r: UInt64) -> Float {
                let u = Int64((r >> 8) % (1 << UInt64(bits)))
                return Float(signed ? u - (1 << Int64(bits - 1)) : u)
            }
            if half { let h = p.bindMemory(to: Float16.self); for i in 0..<h.count { h[i] = Float16(v(rng.next())) } }
            else { let w = p.bindMemory(to: Float.self); for i in 0..<w.count { w[i] = v(rng.next()) } }
        }
    }
    return d
}

// A texture made for one hash entry. Read-only ones are filled once; writable ones keep their bytes in `entry.saved`
// so resetInputs restores them before every run, like the default writable textures.
final class HashTex { let tex: MTLTexture; let view: MTLTexture; let write: Bool; let entry: TexEntry?
    init(tex: MTLTexture, view: MTLTexture, write: Bool, entry: TexEntry?) { self.tex = tex; self.view = view; self.write = write; self.entry = entry } }
struct HashTexKey: Hashable { let words: [UInt32]; let member: Int; let index: Int; let seed: UInt64 }
var hashTexCache: [HashTexKey: HashTex] = [:]

func swizzleFor(_ w3: UInt32, _ f: FmtInfo) -> MTLTextureSwizzleChannels {
    let sels = (0..<4).map { (w3 >> UInt32(3 * $0)) & 7 }
    if sels.contains(where: { $0 == 2 || $0 == 3 }) { return MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .alpha) }
    func one(_ s: UInt32) -> MTLTextureSwizzle {
        switch s {
        case 0: return .zero
        case 1: return .one
        default:
            let ch = Int(s) - 4   // X, Y, Z, W
            if ch >= f.channels { return ch == 3 ? .one : .zero }   // the format has no such channel: 0, or 1 for W
            return [MTLTextureSwizzle.red, .green, .blue, .alpha][ch]
        }
    }
    return MTLTextureSwizzleChannels(red: one(sels[0]), green: one(sels[1]), blue: one(sels[2]), alpha: one(sels[3]))
}

// Builds the texture for T# `t` at element `index` of member `m` read as class `cls` with format `f`.
func makeHashTexture(_ t: [UInt32], member m: MemberInfo, index: Int, f fIn: FmtInfo) -> HashTex? {
    let write = m.access != .readOnly
    let key = HashTexKey(words: t, member: m.id, index: write ? index : -1, seed: texSeed)
    if let h = hashTexCache[key] { return h }
    var f = fIn
    var type = m.type
    if type == .typeCube { type = .type2DArray }
    if isBC(f.fmt) && !(type == .type2D || type == .type2DArray) { f = genericFormat(f.cls, channels: f.channels); dc("tex_bc_as_generic") }
    if m.access == .readWrite { f = FmtInfo(fmt: m.cls == .sint ? .r32Sint : .r32Uint, cls: m.cls, fill: .int(32), channels: 1, metalChannels: 1, exact: false) }
    else if write && !writableFormats.contains(f.fmt) { f = genericFormat(f.cls, channels: f.channels); dc("tex_storage_as_generic") }
    // T# fields (gfx10_descriptors.h): WIDTH word1[31:30] + word2[11:0], HEIGHT word2[27:14], BASE/LAST_LEVEL word3[15:12]/[19:16],
    // TYPE word3[31:28], DEPTH word4[12:0], BASE_ARRAY word4[28:16], MAX_MIP word5[7:4].
    let tw = Int(((t[1] >> 30) & 3) | ((t[2] & 0xFFF) << 2)) + 1, th = Int((t[2] >> 14) & 0x3FFF) + 1
    var base = Int((t[3] >> 12) & 15), last = Int((t[3] >> 16) & 15)
    let ttype = (t[3] >> 28) & 15, tdepth = Int(t[4] & 0x1FFF), tfirst = Int((t[4] >> 16) & 0x1FFF), maxMip = Int((t[5] >> 4) & 15)
    let cap = type == .type3D ? 64 : (type == .type2DArray ? 128 : 256)
    var w = min(tw, cap), h = type == .type1D ? 1 : min(th, cap)
    if tw > cap || (type != .type1D && th > cap) { dc("tex_size_clamped") }
    var depth = 1, layers = 1
    if type == .type3D {
        if ttype == 10 { depth = min(tdepth + 1, cap); if tdepth + 1 > cap { dc("tex_size_clamped") } } else { depth = 16; dc("tex_depth_default") }
    } else if type == .type2DArray {
        if (ttype == 13 || ttype == 11) && tdepth >= tfirst && (ttype != 11 || (tdepth - tfirst + 1) % 6 == 0) {
            layers = tdepth - tfirst + 1
            if layers > 12 { layers = 12; dc("tex_layers_clamped") }
        } else { layers = 6; dc("tex_layers_default") }
    }
    var levels = 1, viewLevels = 0..<1
    if !write && type != .type1D {
        let chain = Int(log2(Double(max(w, h, depth)))) + 1
        if last < base { base = 0; last = 6 }
        levels = min(max(maxMip + 1, last + 1), chain)
        if last + 1 > chain || maxMip + 1 > chain { dc("tex_levels_clamped") }
        let b = min(base, levels - 1)
        viewLevels = b..<max(b + 1, min(last + 1, levels))
    } else if write {
        // a storage view starts at BASE_LEVEL: give the texture that level's size, one level
        if base > 0 { w = max(1, w >> base); h = max(1, h >> base); if type == .type3D { depth = max(1, depth >> base) } }
    }
    let d = MTLTextureDescriptor()
    d.textureType = type; d.pixelFormat = f.fmt
    d.width = w; d.height = h; d.depth = depth; d.arrayLength = layers; d.mipmapLevelCount = levels
    d.usage = write ? [.shaderRead, .shaderWrite] : [.shaderRead, .pixelFormatView]
    d.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: d) else { dc("tex_create_failed"); return nil }
    var rng = SplitMix(s: texSeed &+ 77 &+ t.enumerated().reduce(UInt64(m.id) &* 0x2545F4914F6CDD1D) { ($0 ^ UInt64($1.element)) &* 0x100000001B3 &+ UInt64($1.offset) }
                       &+ UInt64(write ? index : 0) &* 0x9E3779B1)
    let style = rng.next() % 3
    var saved: [(Int, Int, Data)] = []
    for slice in 0..<layers {
        for level in 0..<levels {
            let lw = max(1, w >> level), lh = max(1, h >> level), lz = max(1, depth >> level)
            let bpr: Int, bytes: Int
            if isBC(f.fmt) { bpr = ((lw + 3) / 4) * bcBlockBytes(f.fmt); bytes = bpr * ((lh + 3) / 4) }
            else { bpr = lw * bytesPerPixel(f.fmt); bytes = bpr * lh * lz }
            let data = fillDecoded(f, count: bytes, style: style, rng: &rng)
            data.withUnsafeBytes { tex.replace(region: MTLRegionMake3D(0, 0, 0, lw, lh, lz), mipmapLevel: level, slice: slice,
                                               withBytes: $0.baseAddress!, bytesPerRow: bpr, bytesPerImage: bytes / lz) }
            if write { saved.append((slice, level, data)) }
        }
    }
    var view: MTLTexture = tex
    if !write {
        let sw = swizzleFor(t[3], f)
        let identity = sw.red == .red && sw.green == .green && sw.blue == .blue && sw.alpha == .alpha
        if !identity || viewLevels != 0..<levels {
            guard let v = tex.makeTextureView(pixelFormat: f.fmt, textureType: type, levels: viewLevels, slices: 0..<layers, swizzle: sw)
            else { dc("tex_view_create_failed"); return nil }
            view = v
        }
    }
    var entry: TexEntry? = nil
    if write {
        entry = TexEntry(tex: tex, key: TexKey(type: type.rawValue, data: 0, write: true, depth: false, member: m.id * 65536 + index), depth: false, write: true)
        entry!.saved = saved; entry!.seed = texSeed
    }
    let ht = HashTex(tex: tex, view: view, write: write, entry: entry)
    hashTexCache[key] = ht
    return ht
}

// S# -> MTLSamplerState, following Palmetto's vulkan_sampler (gfx10_descriptors.h). Not representable on macOS 15
// (lodBias and reductionMode need macOS 26): LOD_BIAS, FILTER_MODE min/max; counted. DEPTH_COMPARE_FUNC is not set:
// the hardware applies it only in sample_c instructions and none of these shaders uses sample_compare.
var samplerCache: [String: MTLSamplerState] = [:]
func samplerFor(_ s: [UInt32]) -> MTLSamplerState? {
    let w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3]
    func am(_ v: UInt32) -> MTLSamplerAddressMode {
        switch v {
        case 0: return .repeat
        case 1: return .mirrorRepeat
        case 6: return .clampToBorderColor
        case 3, 5, 7: return .mirrorClampToEdge
        default: return .clampToEdge   // 2 clamp_last_texel, 4 clamp_half_border
        }
    }
    let d = MTLSamplerDescriptor()
    d.supportArgumentBuffers = true
    d.sAddressMode = am(w0 & 7); d.tAddressMode = am((w0 >> 3) & 7); d.rAddressMode = am((w0 >> 6) & 7)
    let modes: [UInt32] = [w0 & 7, (w0 >> 3) & 7, (w0 >> 6) & 7]
    if modes.contains(where: { (m: UInt32) -> Bool in m == 4 || m == 5 || m == 7 }) { dc("sampler_half_or_once_border_approx") }
    d.maxAnisotropy = 1 << Int(min((w0 >> 9) & 7, 4))
    if (w0 >> 12) & 7 != 0 { dc("sampler_compare_not_applied") }
    if (w0 >> 29) & 3 != 0 { dc("sampler_filter_mode_not_applied") }
    let bias = Int32(bitPattern: ((w2 & 0x3FFF) ^ 0x2000) &- 0x2000)
    if bias != 0 { dc("sampler_lod_bias_not_applied") }
    func filt(_ v: UInt32) -> MTLSamplerMinMagFilter { v == 1 || v == 3 ? .linear : .nearest }
    d.magFilter = filt((w2 >> 20) & 3); d.minFilter = filt((w2 >> 22) & 3)
    let mip = (w2 >> 26) & 3
    d.mipFilter = mip == 2 ? .linear : (mip == 0 ? .notMipmapped : .nearest)
    let minLod = Float(w1 & 0xFFF) / 256, maxLod = Float((w1 >> 12) & 0xFFF) / 256
    d.lodMinClamp = minLod; d.lodMaxClamp = max(minLod, maxLod)
    if maxLod < minLod { dc("sampler_max_lod_below_min") }
    switch (w3 >> 30) & 3 {
    case 1: d.borderColor = .opaqueBlack
    case 2: d.borderColor = .opaqueWhite
    case 3: d.borderColor = .transparentBlack; dc("sampler_border_table_as_transparent")
    default: d.borderColor = .transparentBlack
    }
    if (w0 >> 15) & 1 != 0 {   // FORCE_UNNORMALIZED: what Metal (and Vulkan) require of an unnormalised sampler
        dc("sampler_unnormalized")
        d.normalizedCoordinates = false
        d.sAddressMode = d.sAddressMode == .clampToBorderColor ? .clampToBorderColor : .clampToEdge
        d.tAddressMode = d.tAddressMode == .clampToBorderColor ? .clampToBorderColor : .clampToEdge
        d.minFilter = d.magFilter; d.mipFilter = .notMipmapped; d.maxAnisotropy = 1; d.lodMinClamp = 0; d.lodMaxClamp = 0
    }
    let key = "\(d.sAddressMode.rawValue),\(d.tAddressMode.rawValue),\(d.rAddressMode.rawValue),\(d.minFilter.rawValue),\(d.magFilter.rawValue),\(d.mipFilter.rawValue),\(d.maxAnisotropy),\(d.lodMinClamp),\(d.lodMaxClamp),\(d.borderColor.rawValue),\(d.normalizedCoordinates)"
    if let st = samplerCache[key] { return st }
    if samplerCache.count >= min(512, device.maxArgumentBufferSamplerCount - 16) { dc("sampler_cache_full"); return nil }
    guard let st = device.makeSamplerState(descriptor: d) else { dc("sampler_create_failed"); return nil }
    samplerCache[key] = st
    return st
}

// The variant's hash entries as the table holds them (fillHashTable's winners, index != 0), in the variant's order.
var activeEntries: [[UInt32]] = []
func readBackEntries(_ given: [[UInt32]]) -> [[UInt32]] {
    let w = hashTable.contents().bindMemory(to: UInt32.self, capacity: hashSlots * 10 + 16)
    var inTable: [[UInt32]: UInt32] = [:]
    for slot in 0..<hashSlots where w[slot * 10 + 9] != 0 { inTable[(0..<8).map { w[slot * 10 + 1 + $0] }] = w[slot * 10 + 9] }
    var out: [[UInt32]] = [], seen = Set<[UInt32]>()
    for e in given where e.count == 9 {
        let k = Array(e[0..<8])
        guard let idx = inTable[k], !seen.contains(k) else { continue }
        seen.insert(k); out.append(k + [idx])
    }
    return out
}

// Per library: which elements the current variant changed (restored to the default before the next variant patches).
final class HashBindState { var touched: [(id: Int, member: MemberInfo)] = []
    var used: [(MTLResource, MTLResourceUsage)] = []; var writable: [(String, HashTex)] = [] }
var hashStates: [ObjectIdentifier: HashBindState] = [:]

func applyHashBindings(_ lib: MTLLibrary, _ b: Bound, tag: String) -> HashBindState {
    let st = hashStates[ObjectIdentifier(lib)] ?? HashBindState()
    hashStates[ObjectIdentifier(lib)] = st
    guard let enc = b.encoder else { return st }
    countingOn = tag == "a"
    defer { countingOn = false }
    for (id, m) in st.touched {
        if m.sampler { enc.setSamplerState(samplerState, index: id) } else { enc.setTexture(m.defaultTex, index: id) }
    }
    st.touched = []; st.used = []; st.writable = []
    if activeEntries.isEmpty { return st }
    if tag == "a" && hashTexCache.count > 96 { hashTexCache.removeAll() }   // bound memory; A and B of one variant share
    let al = aliases(lib)
    var tTaken: [Int: [UInt32]] = [:], sTaken: [Int: [UInt32]] = [:]
    for e in activeEntries {
        let words = Array(e[0..<8]), idx = Int(e[8])
        let isS = words[4...7].allSatisfy { $0 == 0xFFFF_FFFF }
        dc(isS ? "ssharp_entries" : "tsharp_entries")
        if isS {
            if sTaken[idx] != nil { dc("index_conflict"); continue }
            sTaken[idx] = words
            let ms = b.members.filter { $0.sampler && idx < $0.count }
            if ms.isEmpty { dc("index_out_of_range"); continue }
            guard let s = samplerFor(words) else { dc("sampler_fallback"); continue }
            for m in ms { enc.setSamplerState(s, index: m.id + idx); st.touched.append((m.id + idx, m)) }
            dc("elements_bound_sampler")
            continue
        }
        if tTaken[idx] != nil { dc("index_conflict"); continue }
        tTaken[idx] = words
        let code = (words[1] >> 20) & 0x1FF
        guard let f = formatInfo(code) else { dc("fallback_unknown_format"); dfmt("unknown", code); continue }
        let emulated: Bool = { if case .scaledU = f.fill { return true }; if case .scaledS = f.fill { return true }; return f.channels == 3 && f.metalChannels == 4 }()
        dfmt(!f.exact ? "approx" : (emulated ? "emulated" : "exact"), code)
        if (words[3] >> 28) & 15 < 8 { dc("tsharp_type_not_image") }
        if (0..<4).contains(where: { (k: Int) -> Bool in let x = (words[3] >> UInt32(3 * k)) & 7; return x == 2 || x == 3 }) { dc("tsharp_reserved_dst_sel_as_identity") }
        if (words[3] >> 16) & 15 < (words[3] >> 12) & 15 { dc("tsharp_last_level_below_base") }
        let tw = Int(((words[1] >> 30) & 3) | ((words[2] & 0xFFF) << 2)) + 1, th = Int((words[2] >> 14) & 0x3FFF) + 1
        if tw > 256 || th > 256 { dc("tsharp_size_over_256") }
        var any = false
        for m in b.members where !m.sampler {
            guard idx < m.count else { dc("element_index_out_of_range"); continue }
            var cls = f.cls, ff = f
            switch m.access {
            case .readOnly:
                let views = (al[m.name] ?? []).union([m.cls])
                if !views.contains(cls) { cls = m.cls; dc("element_class_forced_by_member"); ff = genericFormat(cls, channels: f.channels) }
            case .readWrite:
                cls = m.cls
            default:
                if m.cls != cls { continue }   // the shader picks the storage member of the format's class
            }
            guard let ht = makeHashTexture(words, member: m, index: idx, f: ff) else { continue }
            enc.setTexture(ht.view, index: m.id + idx)
            st.touched.append((m.id + idx, m))
            st.used.append((ht.view, ht.write ? [.read, .write] : .read))
            if ht.view !== ht.tex { st.used.append((ht.tex, ht.write ? [.read, .write] : .read)) }
            if ht.write { st.writable.append(("\(m.name)[\(idx)]", ht)) }
            any = true
            dc(ht.write ? "elements_bound_storage" : "elements_bound_sampled")
        }
        if !any { dc("tsharp_not_bound") }
    }
    st.writable.sort { $0.0 < $1.0 }
    return st
}

// MARK: - run one shader

// The "memory" output keeps its place in `outputs` with an empty Data; its bytes are read from `mem` (window bytes):
// the Metal buffer itself, pristine (A wrote nothing) or snapA (A's copy while B runs).
struct RunOut { var outputs: [(String, Data, String)] = []; var gpuMs = 0.0
                var mem: UnsafeRawPointer? = nil; var memHash: String? = nil; var memWritten = false }

func wait(_ cb: MTLCommandBuffer, tag: String) {
    let sem = DispatchSemaphore(value: 0)
    cb.addCompletedHandler { _ in sem.signal() }
    cb.commit()
    if sem.wait(timeout: .now() + 20) == .timedOut { finish("timeout_\(tag)", code: 3) }
    if cb.status == .error {
        result["gpu_error_\(tag)"] = String(String(describing: cb.error).prefix(3000))
        finish("gpu_\(tag)", code: 4)
    }
}

func runCompute(_ lib: MTLLibrary, tag: String) -> RunOut {
    mark("pipeline_\(tag)")
    guard let fn = function(lib, "main0") else { finish("pipeline_\(tag)") }
    let state: MTLComputePipelineState, refl: MTLComputePipelineReflection?
    if let cached = computePipelines[ObjectIdentifier(lib)] { (state, refl) = cached } else {
        var made: (MTLComputePipelineState?, MTLComputePipelineReflection?, Error?) = (nil, nil, nil)
        let sem = DispatchSemaphore(value: 0)
        device.makeComputePipelineState(function: fn, options: [.bindingInfo]) { s, r, e in made = (s, r, e); sem.signal() }
        if sem.wait(timeout: .now() + stepTimeout) == .timedOut { finish("pipeline_timeout_\(tag)") }
        guard let st = made.0 else {
            result["pipeline_error_\(tag)"] = String(String(describing: made.2).prefix(3000)); finish("pipeline_\(tag)")
        }
        (state, refl) = (st, made.1)
        computePipelines[ObjectIdentifier(lib)] = (state, refl)
    }
    let ls = (meta["local_size"] as? [Int]) ?? [64, 1, 1]
    let threads = MTLSize(width: ls[0], height: ls.count > 1 ? ls[1] : 1, depth: ls.count > 2 ? ls[2] : 1)
    if threads.width * threads.height * threads.depth > state.maxTotalThreadsPerThreadgroup {
        result["max_threads_\(tag)"] = state.maxTotalThreadsPerThreadgroup
        finish("unsupported")
    }
    mark("bind_\(tag)")
    let b = boundFor(lib, refl?.bindings ?? [], function: fn)
    if !b.unsupported.isEmpty { result["unsupported"] = b.unsupported; finish("unsupported") }
    mark("reset_inputs_\(tag)")
    prepareTextures(b)
    let hb = applyHashBindings(lib, b, tag: tag)
    for (n, ht) in hb.writable { writable.append((n, ht.tex)); writableSaved[ObjectIdentifier(ht.tex)] = ht.entry }
    resetInputs()
    snapshotWritable()
    mark("encode_\(tag)")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(state)
    if let ab = b.argBuffer { enc.setBuffer(ab, offset: 0, index: b.argIndex) }
    if b.pushIndex >= 0 { enc.setBytes(push, length: push.count * 4, index: b.pushIndex) }
    result["push_index"] = b.pushIndex
    memset(coverage.contents(), 0, covWords * 4)
    enc.setBuffer(coverage, offset: 0, index: covIndex)
    enc.useResource(memory, usage: [.read, .write])
    for (r, u) in b.used { enc.useResource(r, usage: u) }
    for (r, u) in hb.used { enc.useResource(r, usage: u) }
    let groups = threads.height > 1 ? MTLSize(width: 2, height: 2, depth: 1) : MTLSize(width: 4, height: 1, depth: 1)
    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    enc.endEncoding()
    mark("gpu_\(tag)")
    wait(cb, tag: tag)
    mark("readback_\(tag)")
    readCoverage(tag: tag)
    var out = RunOut(gpuMs: (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    out.outputs.append(("memory", Data(), "u32")); out.mem = UnsafeRawPointer(memory.contents())
    for (n, t) in writable { out.outputs.append(("texture:" + n, textureBytes(t), t.pixelFormat == .rgba32Float ? "f32" : "u32")) }
    return out
}

func snapshotWritable() { for (n, t) in writable { initialBytes["texture:" + n] = textureBytes(t) } }

func clearBytes(_ f32: Bool) -> Data {
    var clear = [UInt32](repeating: 0, count: rtSize * rtSize * 4)
    for i in 0..<clear.count { clear[i] = f32 ? [Float(0.25), 0.5, 0.75, 1][i % 4].bitPattern : 7 }
    return Data(bytes: clear, count: clear.count * 4)
}
let clearF32 = clearBytes(true), clearU32 = clearBytes(false)
let clearDepth: Data = { let c = [Float](repeating: 0.5, count: rtSize * rtSize); return Data(bytes: c, count: c.count * 4) }()
let depthAlways: MTLDepthStencilState = {
    let ds = MTLDepthStencilDescriptor(); ds.depthCompareFunction = .always; ds.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: ds)!
}()

func runFragment(_ lib: MTLLibrary, vs: MTLLibrary, tag: String) -> RunOut {
    mark("pipeline_\(tag)")
    guard let fn = function(lib, "main0"), let vfn = function(vs, "vs_main") else { finish("pipeline_\(tag)") }
    let outs = (meta["outputs"] as? [[String: Any]]) ?? []
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = vfn; d.fragmentFunction = fn
    var targets: [(Int, MTLTexture, String)] = []
    var depthTex: MTLTexture?
    for o in outs {
        let kind = o["kind"] as? String ?? "", type = o["type"] as? String ?? "float"
        if kind == "color", let slot = o["slot"] as? Int {
            let fmt: MTLPixelFormat = type == "uint" || type == "ushort" ? .rgba32Uint : (type == "int" || type == "short" ? .rgba32Sint : .rgba32Float)
            d.colorAttachments[slot].pixelFormat = fmt
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: rtSize, height: rtSize, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
            targets.append((slot, device.makeTexture(descriptor: td)!, fmt == .rgba32Float ? "f32" : "u32"))
        } else if kind == "depth" {
            d.depthAttachmentPixelFormat = .depth32Float
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: rtSize, height: rtSize, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]; td.storageMode = .private
            depthTex = device.makeTexture(descriptor: td)
        }
    }
    let state: MTLRenderPipelineState, refl: MTLRenderPipelineReflection?
    if let cached = renderPipelines[ObjectIdentifier(lib)] { (state, refl) = cached } else {
        var made: (MTLRenderPipelineState?, MTLRenderPipelineReflection?, Error?) = (nil, nil, nil)
        let sem = DispatchSemaphore(value: 0)
        device.makeRenderPipelineState(descriptor: d, options: [.bindingInfo]) { s, r, e in made = (s, r, e); sem.signal() }
        if sem.wait(timeout: .now() + stepTimeout) == .timedOut { finish("pipeline_timeout_\(tag)") }
        guard let st = made.0 else {
            result["pipeline_error_\(tag)"] = String(String(describing: made.2).prefix(3000)); finish("pipeline_\(tag)")
        }
        (state, refl) = (st, made.1)
        renderPipelines[ObjectIdentifier(lib)] = (state, refl)
    }
    mark("bind_\(tag)")
    let b = boundFor(lib, refl?.fragmentBindings ?? [], function: fn)
    if !b.unsupported.isEmpty { result["unsupported"] = b.unsupported; finish("unsupported") }
    mark("reset_inputs_\(tag)")
    prepareTextures(b)
    let hb = applyHashBindings(lib, b, tag: tag)
    for (n, ht) in hb.writable { writable.append((n, ht.tex)); writableSaved[ObjectIdentifier(ht.tex)] = ht.entry }
    resetInputs()
    snapshotWritable()
    mark("encode_\(tag)")
    let rp = MTLRenderPassDescriptor()
    for (slot, t, kind) in targets {
        rp.colorAttachments[slot].texture = t
        rp.colorAttachments[slot].loadAction = .clear
        rp.colorAttachments[slot].clearColor = kind == "f32" ? MTLClearColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1) : MTLClearColor(red: 7, green: 7, blue: 7, alpha: 7)
        rp.colorAttachments[slot].storeAction = .store
    }
    if let dt = depthTex { rp.depthAttachment.texture = dt; rp.depthAttachment.clearDepth = 0.5; rp.depthAttachment.storeAction = .store }
    if targets.isEmpty && depthTex == nil {          // no attachments: a pass with an explicit size
        rp.renderTargetWidth = rtSize; rp.renderTargetHeight = rtSize; rp.defaultRasterSampleCount = 1
    }
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
    enc.setRenderPipelineState(state)
    if depthTex != nil {
        enc.setDepthStencilState(depthAlways)
    }
    if let ab = b.argBuffer { enc.setFragmentBuffer(ab, offset: 0, index: b.argIndex) }
    if b.pushIndex >= 0 { enc.setFragmentBytes(push, length: push.count * 4, index: b.pushIndex) }
    result["push_index"] = b.pushIndex
    memset(coverage.contents(), 0, covWords * 4)
    enc.setFragmentBuffer(coverage, offset: 0, index: covIndex)
    enc.useResource(memory, usage: [.read, .write], stages: .fragment)
    for (r, u) in b.used { enc.useResource(r, usage: u, stages: .fragment) }
    for (r, u) in hb.used { enc.useResource(r, usage: u, stages: .fragment) }
    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    enc.endEncoding()
    // depth is a private texture: copy it to a shared buffer to compare it
    var depthBuf: MTLBuffer?
    if let dt = depthTex {
        depthBuf = device.makeBuffer(length: rtSize * rtSize * 4, options: .storageModeShared)
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: dt, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: rtSize, height: rtSize, depth: 1), to: depthBuf!, destinationOffset: 0,
                  destinationBytesPerRow: rtSize * 4, destinationBytesPerImage: rtSize * rtSize * 4)
        blit.endEncoding()
    }
    mark("gpu_\(tag)")
    wait(cb, tag: tag)
    mark("readback_\(tag)")
    readCoverage(tag: tag)
    var out = RunOut(gpuMs: (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    for (slot, t, kind) in targets {
        out.outputs.append(("color\(slot)", textureBytes(t), kind))
        initialBytes["color\(slot)"] = kind == "f32" ? clearF32 : clearU32
    }
    if let db = depthBuf {
        out.outputs.append(("depth", Data(bytes: db.contents(), count: rtSize * rtSize * 4), "f32"))
        initialBytes["depth"] = clearDepth
    }
    out.outputs.append(("memory", Data(), "u32")); out.mem = UnsafeRawPointer(memory.contents())
    for (n, t) in writable { out.outputs.append(("texture:" + n, textureBytes(t), "u32")) }
    return out
}

// MARK: - main

guard stage == "cs" || stage == "ps" else { result["reason"] = "stage \(stage) not supported yet"; finish("unsupported") }
log("shader \(meta["name"] ?? "?") stage \(stage) seeds \(seedList) a=\(aPath) b=\(bPath)")
mark("compile_a")
guard let libA = compile(aPath, coverage: wantCoverage) else { finish(compileTimedOut ? "compile_timeout_a" : "compile_a") }
let onlyA = bPath == "none"
var libB: MTLLibrary? = nil
if !onlyA {
    mark("compile_b")
    guard let l = compile(bPath) else { finish(compileTimedOut ? "compile_timeout_b" : "compile_b") }
    libB = l
}
var vsLib: MTLLibrary?
if stage == "ps" {
    mark("compile_vs")
    guard let vsPath = arg("vs"), let v = compile(vsPath) else { finish("compile_vs") }
    vsLib = v
}

func hash(_ d: Data) -> String {
    d.withUnsafeBytes { (p: UnsafeRawBufferPointer) in hashRaw(p.baseAddress, p.count) }
}
func hashRaw(_ p: UnsafeRawPointer?, _ count: Int) -> String {
    var h: UInt64 = 0xcbf29ce484222325
    if let p {
        let w = UnsafeBufferPointer(start: p.assumingMemoryBound(to: UInt64.self), count: count / 8)
        for x in w { h = (h ^ x) &* 0x100000001b3 }
    }
    return String(h, radix: 16)
}
func memEqual(_ a: UnsafeRawPointer, _ b: UnsafeRawPointer) -> Bool { a == b || memcmp(a, b, window) == 0 }

// A's title memory: hash it, see whether A wrote it and keep a copy for B only when A wrote it (else it equals pristine).
let snapA = UnsafeMutableRawPointer.allocate(byteCount: window, alignment: 16384)
func settleMemoryA(_ A: inout RunOut, keepForB: Bool) {
    guard let m = A.mem else { return }
    A.memHash = hashRaw(m, window)
    A.memWritten = !memEqual(m, UnsafeRawPointer(pristine))
    if keepForB {
        if A.memWritten { snapA.copyMemory(from: m, byteCount: window); A.mem = UnsafeRawPointer(snapA) }
        else { A.mem = UnsafeRawPointer(pristine) }
    }
}
func outputsEqual(_ A: RunOut, _ B: RunOut) -> Bool {
    for (i, (name, da, _)) in A.outputs.enumerated() {
        guard i < B.outputs.count, B.outputs[i].0 == name else { return false }
        if name == "memory" { if !memEqual(A.mem!, B.mem!) { return false } }
        else if da != B.outputs[i].1 { return false }
    }
    return true
}

// The per-output details (only used for variant 0 and the first difference).
func compare(_ A: RunOut, _ B: RunOut) -> (Bool, [[String: Any]]) {
    var equal = true
    var details: [[String: Any]] = []
    for (i, (name, da0, kind)) in A.outputs.enumerated() {
        guard i < B.outputs.count, B.outputs[i].0 == name else { equal = false; details.append(["output": name, "error": "missing in b"]); continue }
        let db0 = B.outputs[i].1
        let isMem = name == "memory"
        let same = isMem ? memEqual(A.mem!, B.mem!) : da0 == db0
        var row: [String: Any]
        if isMem {
            row = ["output": name, "bytes": window, "hash_a": A.memHash ?? hashRaw(A.mem!, window),
                   "hash_b": same ? (A.memHash ?? hashRaw(A.mem!, window)) : hashRaw(B.mem!, window)]
            row["written_by_a"] = A.memWritten
        } else {
            row = ["output": name, "bytes": da0.count, "hash_a": hash(da0), "hash_b": hash(db0)]
            if let before = initialBytes[name] { row["written_by_a"] = da0 != before }
        }
        if !same {
            equal = false
            var diffWords = 0, maxAbs: Double = 0, first = -1
            withBytes(A, i) { (pa: UnsafeRawBufferPointer) in withBytes(B, i) { (pb: UnsafeRawBufferPointer) in
                let wa = pa.bindMemory(to: UInt32.self), wb = pb.bindMemory(to: UInt32.self)
                for j in 0..<min(wa.count, wb.count) where wa[j] != wb[j] {
                    diffWords += 1; if first < 0 { first = j }
                    if kind == "f32" {
                        let fa = Double(Float(bitPattern: wa[j])), fb = Double(Float(bitPattern: wb[j]))
                        if fa.isFinite && fb.isFinite { maxAbs = max(maxAbs, abs(fa - fb)) } else { maxAbs = .infinity }
                    }
                }
            } }
            row["diff_words"] = diffWords; row["first_diff_word"] = first
            if kind == "f32" { row["max_abs_diff"] = maxAbs.isFinite ? maxAbs : -1 }
        }
        if kind == "f32" {
            var nonfinite = 0
            withBytes(A, i) { (p: UnsafeRawBufferPointer) in for w in p.bindMemory(to: UInt32.self) where (w & 0x7F80_0000) == 0x7F80_0000 { nonfinite += 1 } }
            row["nonfinite_a"] = nonfinite
        }
        details.append(row)
    }
    return (equal, details)
}
func withBytes<R>(_ o: RunOut, _ i: Int, _ body: (UnsafeRawBufferPointer) -> R) -> R {
    if o.outputs[i].0 == "memory", let m = o.mem { return body(UnsafeRawBufferPointer(start: m, count: window)) }
    return o.outputs[i].1.withUnsafeBytes(body)
}

struct Variant { var seed: UInt64; var mem: [(Int, UInt32)] = []; var push: [(Int, UInt32)] = []; var texSeed: UInt64? = nil; var hash: [[UInt32]] = [] }
var variants: [Variant] = seedList.map { Variant(seed: $0) }
if let p = arg("inputs") {
    guard let d = FileManager.default.contents(atPath: p),
          let list = try? JSONSerialization.jsonObject(with: d) as? [[String: Any]] else {
        result["reason"] = "cannot read --inputs \(p)"; finish("bad_inputs", code: 2)
    }
    func pairs(_ x: Any?) -> [(Int, UInt32)] {
        ((x as? [[Any]]) ?? []).compactMap { e in
            guard e.count == 2, let i = (e[0] as? NSNumber)?.intValue, let v = (e[1] as? NSNumber)?.uint32Value else { return nil }
            return (i, v)
        }
    }
    variants = list.map { Variant(seed: (($0["seed"] as? NSNumber)?.uint64Value) ?? 1, mem: pairs($0["mem"]), push: pairs($0["push"]),
                                  texSeed: ($0["tex_seed"] as? NSNumber)?.uint64Value,
                                  hash: (($0["hash"] as? [[Any]]) ?? []).map { $0.compactMap { ($0 as? NSNumber)?.uint32Value } }) }
}
let runAll = arg("all") != nil
var pristineSeed: UInt64? = nil
var undoMem: [(Int, UInt32)] = []   // (word, value before the patch) of the patches applied to pristine
func setupVariant(_ v: Variant) {
    seed = v.seed
    texSeed = v.texSeed ?? v.seed
    if pristineSeed == v.seed { for (i, x) in undoMem.reversed() { pristine[i] = x } }
    else { pristine.update(from: basePointer(v.seed), count: totalWords); pristineSeed = v.seed }
    undoMem.removeAll(keepingCapacity: true)
    setupPush()
    let words = window / 4
    for (i, x) in v.mem where i >= 0 && i < words { undoMem.append((i, pristine[i])); pristine[i] = x }
    for (i, x) in v.push where i >= 0 && i < 32 { push[i] = x }
    fillHashTable(v.hash)
    activeEntries = readBackEntries(v.hash)
}
func writtenAndHashes(_ A: RunOut) -> ([String], [String: String]) {
    var w: [String] = [], h: [String: String] = [:]
    for (name, d, _) in A.outputs {
        if name == "memory" { h[name] = A.memHash!; if A.memWritten { w.append(name) }; continue }
        h[name] = hash(d)
        if let before = initialBytes[name], d != before { w.append(name) }
    }
    return (w, h)
}

var allEqual = true
var perSeed: [[String: Any]] = []
var rows: [[String: Any]] = []
var readsUnion = Set<Int>()
for (k, v) in variants.enumerated() {
    let s = v.seed
    mark("inputs_variant_\(k)")
    setupVariant(v)
    writable.removeAll(); initialBytes.removeAll()
    var A = stage == "cs" ? runCompute(libA, tag: "a") : runFragment(libA, vs: vsLib!, tag: "a")
    settleMemoryA(&A, keepForB: !onlyA)
    let (wr, hs) = writtenAndHashes(A)
    var row: [String: Any] = ["i": k, "seed": s, "hits": lastHits, "written": wr, "hashes": hs,
                              "loop_guard": lastLoopA]
    if wantCmplog { row["cmp"] = lastCmp; row["keys"] = lastKeys }
    readsUnion.formUnion(lastReads)
    if onlyA {
        rows.append(row)
        if k == 0 { result["gpu_ms_a"] = A.gpuMs }
        continue
    }
    writable.removeAll()
    let B = stage == "cs" ? runCompute(libB!, tag: "b") : runFragment(libB!, vs: vsLib!, tag: "b")
    mark("compare_variant_\(k)")
    let equal = outputsEqual(A, B)
    let details: [[String: Any]] = (k == 0 || (!equal && allEqual)) ? compare(A, B).1 : []
    log("variant \(k) seed \(s): \(equal ? "equal" : "DIFFERENT") (gpu a \(String(format: "%.2f", A.gpuMs)) ms, b \(String(format: "%.2f", B.gpuMs)) ms)")
    row["equal"] = equal
    row["loop_guard_b"] = lastLoopB
    rows.append(row)
    perSeed.append(["seed": s, "variant": k, "equal": equal, "gpu_ms_a": A.gpuMs, "gpu_ms_b": B.gpuMs])
    if k == 0 {
        result["gpu_ms_a"] = A.gpuMs
        result["gpu_ms_b"] = B.gpuMs
        if let dump = arg("dump") {
            try? FileManager.default.createDirectory(atPath: dump, withIntermediateDirectories: true)
            for (name, d, _) in A.outputs where name != "memory" { try? d.write(to: URL(fileURLWithPath: dump + "/a_" + name.replacingOccurrences(of: ":", with: "_") + ".bin")) }
        }
    }
    if k == 0 || (!equal && allEqual) { result["outputs"] = details }
    if !equal {
        if allEqual { result["first_diff_seed"] = s; result["first_diff_variant"] = k }
        allEqual = false
        if !runAll { break }
    }
    if !runAll && ((result["loop_guard_a"] as? Bool) == true || (result["loop_guard_b"] as? Bool) == true) { break }
}
if arg("inputs") != nil || runAll || onlyA { result["variants"] = rows }
if wantReads { result["reads"] = readsUnion.sorted(); result["read_shift"] = readShift }
result["per_seed"] = perSeed
result["equal"] = allEqual
if variants.contains(where: { !$0.hash.isEmpty }) {
    var dd: [String: Any] = dcount
    dd["formats"] = dformats
    dd["formats_json"] = fmtNames.isEmpty ? "missing: \(formatsPath)" : formatsPath
    result["descriptors"] = dd
}
finish("ok")
