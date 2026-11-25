import Metal
let d = MTLCreateSystemDefaultDevice()!
print("GPU:", d.name)
print("supportsRaytracing:", d.supportsRaytracing)
print("supportsRaytracingFromRender:", d.supportsRaytracingFromRender)
print("Apple7 (M1):", d.supportsFamily(.apple7), " Apple8:", d.supportsFamily(.apple8), " Apple9 (M3, HW RT):", d.supportsFamily(.apple9))
print("Metal3:", d.supportsFamily(.metal3))
print("meshShaders(Apple7+ Metal3):", d.supportsFamily(.metal3))
print("recommendedMaxWorkingSet GB:", Double(d.recommendedMaxWorkingSetSize)/1e9)
print("hasUnifiedMemory:", d.hasUnifiedMemory)
