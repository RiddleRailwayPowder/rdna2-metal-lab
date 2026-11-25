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
               var entries: [TexEntry] = []; var writable: [(String, MTLTexture)] = [] }

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
            case .sampler:
                enc.setSamplerStates([MTLSamplerState?](repeating: samplerState, count: count), range: m.argumentIndex..<(m.argumentIndex + count))
            case .pointer:
                enc.setBuffer(hashTable, offset: 0, index: m.argumentIndex)
                b.used.append((hashTable, .read))
            default:
                b.unsupported.append("member \(m.name) type \(dtype.rawValue)")
            }
        }
        b.argBuffer = ab; b.argIndex = buf.index
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
    let bpp: Int
    switch t.pixelFormat {
    case .rgba32Float, .rgba32Uint, .rgba32Sint: bpp = 16
    case .rgba16Float: bpp = 8
    default: bpp = 4
    }
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
finish("ok")
