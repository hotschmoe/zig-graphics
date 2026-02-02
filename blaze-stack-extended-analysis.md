# BLAZE/FORGE/FLUX Extended Analysis

## Overview

This document addresses three strategic questions about the Zig graphics stack architecture:

1. **WebAssembly Extension** - How would we add web/WASM support?
2. **GPU Compute** - Can BLAZE serve as a general compute library?
3. **Conformance Testing** - What industry-standard test suites exist?

---

# Question 1: WebAssembly Extension

## The Challenge

BLAZE targets Vulkan directly. WebAssembly in browsers cannot access Vulkan - it must go through WebGPU, which is the browser's GPU abstraction. This creates an architectural fork:

```
Native Path:                    Web Path:
┌──────────────┐               ┌──────────────┐
│    BLAZE     │               │    BLAZE     │
│   (Zig API)  │               │   (Zig API)  │
└──────┬───────┘               └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐               ┌──────────────┐
│   Vulkan     │               │   WebGPU     │
│   Backend    │               │   Backend    │
└──────┬───────┘               └──────┬───────┘
       │                              │
       ▼                              ▼
   GPU Driver                  Browser WebGPU
                               Implementation
```

## WebGPU: The Web Graphics Standard

WebGPU is the successor to WebGL and is now shipping in major browsers. Key facts:

- Chrome enabled WebGPU in version 113 (April 2023)
- Firefox shipped WebGPU support in version 141 (July 2025)
- Safari debuted WebGPU in Safari 26 (June 2025)
- WebGPU uses WGSL (WebGPU Shading Language), which BLAZE already uses
- WebGPU supports both rendering and general-purpose compute (GPGPU)

## Zig + WASM: Current State

Zig has first-class WebAssembly support via LLVM:

```bash
# Compile Zig to WASM
zig build-lib src/main.zig -target wasm32-freestanding -dynamic

# Or with Emscripten for more browser integration
zig build --sysroot [path/to/emsdk]/upstream/emscripten/cache/sysroot
```

There are already working examples of Zig + WebGPU + WASM (e.g., `webgpu-wasm-zig` project on GitHub).

## Architecture for Web Support

### Option A: Dual Backend (Recommended)

Add a WebGPU backend to BLAZE alongside Vulkan:

```zig
// blaze/src/backend.zig

pub const Backend = enum {
    vulkan,
    webgpu,
};

pub const Context = struct {
    backend: Backend,
    
    // Backend-specific handles
    vulkan_ctx: ?vulkan.Context,
    webgpu_ctx: ?webgpu.Context,
    
    pub fn init(allocator: Allocator, config: Config) !Context {
        if (builtin.target.isWasm()) {
            return initWebGPU(allocator, config);
        } else {
            return initVulkan(allocator, config);
        }
    }
};
```

### Option B: WebGPU Everywhere

An alternative is to use wgpu-native (or Dawn) for *all* platforms, not just web. This gives you one backend that works everywhere:

```
┌──────────────┐
│    BLAZE     │
│   (Zig API)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  wgpu-native │  ← C bindings, easy Zig interop
│   or Dawn    │
└──────┬───────┘
       │
   ┌───┴───┬───────┬────────┐
   ▼       ▼       ▼        ▼
Vulkan  D3D12   Metal   WebGPU
(Linux) (Win)   (Mac)   (WASM)
```

**Pros:**
- Single codebase
- Automatic portability
- Well-tested abstraction

**Cons:**
- Dependency on external C library (wgpu-native/Dawn)
- Slightly less control than raw Vulkan
- May limit access to cutting-edge Vulkan extensions

### Recommended Approach: Phased

**Phase 1 (Now):** Build BLAZE with raw Vulkan backend for Windows/Linux
- Full control, maximum learning
- Access to all Vulkan features (mesh shaders, ray tracing, etc.)

**Phase 2 (Later):** Add WebGPU backend for WASM
- Share the BLAZE API surface
- Different implementation underneath
- WGSL shaders work on both (major win!)

**Phase 3 (Optional):** Consider wgpu-native for Metal/macOS
- If you ever care about Mac
- Could also replace Vulkan backend if you want simplicity over control

## What You Lose on Web

| Feature | Vulkan | WebGPU |
|---------|--------|--------|
| Mesh shaders | ✅ | ❌ (not yet) |
| Hardware ray tracing | ✅ | ❌ (not yet) |
| Bindless textures | ✅ Full | ⚠️ Limited |
| Timeline semaphores | ✅ | ⚠️ Different model |
| Memory aliasing | ✅ | ❌ |
| Compute shaders | ✅ | ✅ |
| Storage buffers | ✅ | ✅ |
| Indirect rendering | ✅ | ✅ |
| Multi-draw indirect | ✅ | ⚠️ Limited |

For your use cases (WoW-style games, engineering viz), WebGPU has everything you need.

## Concrete WASM Build Setup

```zig
// build.zig

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    // Conditional backend selection
    if (target.result.isWasm()) {
        // WebGPU backend for WASM
        exe.addModule("blaze_backend", webgpu_module);
        
        // Emscripten-specific settings
        exe.addSystemIncludePath("/path/to/emscripten/cache/sysroot/include");
    } else {
        // Vulkan backend for native
        exe.addModule("blaze_backend", vulkan_module);
        exe.linkSystemLibrary("vulkan");
    }
    
    b.installArtifact(exe);
}
```

## Web-Specific Considerations

### Canvas Integration

On web, you render to an HTML canvas. BLAZE's Surface abstraction needs to handle this:

```zig
// For web
pub const SurfaceDesc = struct {
    native_handle: NativeHandle,
    
    pub const NativeHandle = union(enum) {
        // Native handles...
        xlib: struct { display: *anyopaque, window: u64 },
        win32: struct { hinstance: *anyopaque, hwnd: *anyopaque },
        
        // Web handle
        canvas: struct { 
            canvas_id: []const u8,  // HTML element ID
        },
    };
};
```

### JavaScript Interop

For web, you'll need to call browser APIs via extern functions:

```zig
// Import browser's WebGPU API
extern "webgpu" fn wgpuCreateInstance(desc: *const InstanceDescriptor) Instance;
extern "webgpu" fn wgpuInstanceRequestAdapter(
    instance: Instance,
    options: *const RequestAdapterOptions,
    callback: RequestAdapterCallback,
    userdata: ?*anyopaque,
) void;
```

Or use Emscripten's WebGPU bindings which handle this for you.

---

# Question 2: GPU Compute with BLAZE

## Short Answer: Yes, Absolutely

BLAZE can and should support general GPU compute. Both Vulkan and WebGPU have first-class compute shader support. In fact, the design I outlined already includes compute:

```zig
// Already in the BLAZE spec
pub fn dispatch(self: *Encoder, x: u32, y: u32, z: u32) void {
    self.commands.append(.{ .dispatch = .{ .x = x, .y = y, .z = z } }) catch unreachable;
}

pub fn dispatchIndirect(self: *Encoder, buffer: Buffer, offset: u64) void {
    self.commands.append(.{ .dispatch_indirect = .{ .buffer = buffer, .offset = offset } }) catch unreachable;
}
```

## Compute-Only Mode

For pure compute workloads (no rendering), BLAZE could operate without a window or swapchain:

```zig
pub const ContextConfig = struct {
    // ...existing fields...
    
    mode: Mode = .full,
    
    pub const Mode = enum {
        full,           // Rendering + Compute
        compute_only,   // No swapchain, no rendering pipelines
    };
};

// Usage for pure compute
var ctx = try blaze.Context.init(allocator, .{
    .mode = .compute_only,
    .features = .{
        .timeline_semaphores = true,
        .buffer_device_address = true,
    },
});

// Create compute pipeline
const pipeline = try ctx.createComputePipeline(.{
    .shader = MyComputeShader,
    .entry_point = "main",
});

// Create buffers
const input_buffer = try ctx.createBuffer(.{
    .size = data.len * @sizeOf(f32),
    .usage = .{ .storage = true, .transfer_dst = true },
    .memory = .device_local,
});

const output_buffer = try ctx.createBuffer(.{
    .size = result_size,
    .usage = .{ .storage = true, .transfer_src = true },
    .memory = .device_local,
});

// Upload data
try ctx.uploadBuffer(input_buffer, std.mem.sliceAsBytes(data));

// Dispatch compute
var encoder = blaze.Encoder.init(allocator);
encoder.setPipeline(pipeline);
encoder.setBindGroup(0, bind_group);
encoder.dispatch(data.len / 64, 1, 1);  // 64 threads per workgroup

try ctx.submit(encoder.finish(), null);
ctx.waitIdle();

// Read back results
try ctx.downloadBuffer(output_buffer, result_slice);
```

## Do You Need a Separate Compute Library?

**No.** Here's why:

| Concern | Answer |
|---------|--------|
| API surface | Compute is a subset of graphics - same buffers, pipelines, command encoding |
| Vulkan handles | Same `VkDevice`, `VkQueue`, `VkBuffer` work for compute |
| Shaders | WGSL/SPIR-V compute shaders use the same toolchain |
| Memory | Same allocator, same buffer types |
| Sync | Same timeline semaphores, same fences |

The only difference is you don't create a swapchain or render passes. BLAZE should have a "headless" or "compute-only" mode, not a separate library.

## Compute-Specific Features to Include

### 1. Async Compute Queues

Modern GPUs have separate compute queues that can run in parallel with graphics:

```zig
pub const QueueType = enum {
    graphics,   // Can do everything
    compute,    // Compute only, may run async
    transfer,   // DMA only, highest throughput for copies
};

pub const Context = struct {
    graphics_queue: Queue,
    compute_queue: ?Queue,   // May be separate or same as graphics
    transfer_queue: ?Queue,  // May be separate
    
    pub fn submit(self: *Context, queue_type: QueueType, commands: []const Command, signal: ?*TimelineSemaphore) !void {
        const queue = switch (queue_type) {
            .graphics => self.graphics_queue,
            .compute => self.compute_queue orelse self.graphics_queue,
            .transfer => self.transfer_queue orelse self.graphics_queue,
        };
        // ...
    }
};
```

### 2. Shader Storage Buffer Objects (SSBOs)

```zig
pub const BufferUsage = packed struct {
    vertex: bool = false,
    index: bool = false,
    uniform: bool = false,
    storage: bool = false,      // ← Key for compute
    storage_read: bool = false, // Read-only storage (can be faster)
    indirect: bool = false,
    transfer_src: bool = false,
    transfer_dst: bool = false,
};
```

### 3. Workgroup Shared Memory

In compute shaders, workgroups can share memory:

```wgsl
// Compute shader with shared memory
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    // Each thread in workgroup can access shared_data
    shared_data[local_id.x] = some_computation();
    workgroupBarrier();
    // Now all threads can see each other's results
}
```

BLAZE doesn't need to do anything special for this - it's handled in the shader.

### 4. Subgroup Operations

For advanced compute, Vulkan supports subgroup operations (SIMD-style):

```zig
pub const Features = struct {
    // ...
    subgroup_size_control: bool = false,  // Choose subgroup size
    subgroup_arithmetic: bool = false,    // subgroupAdd, etc.
    subgroup_ballot: bool = false,        // subgroupBallot
    subgroup_shuffle: bool = false,       // subgroupShuffle
};
```

## Example: Parallel Reduction (Sum)

```zig
const std = @import("std");
const blaze = @import("blaze");

const ReductionShader = blaze.Shader.compile(@embedFile("reduction.wgsl"));

pub fn gpuSum(ctx: *blaze.Context, data: []const f32) !f32 {
    const n = data.len;
    const workgroup_size = 256;
    const num_workgroups = (n + workgroup_size - 1) / workgroup_size;
    
    // Create buffers
    const input = try ctx.createBuffer(.{
        .size = n * @sizeOf(f32),
        .usage = .{ .storage = true, .transfer_dst = true },
        .memory = .device_local,
    });
    defer ctx.destroyBuffer(input);
    
    const output = try ctx.createBuffer(.{
        .size = num_workgroups * @sizeOf(f32),
        .usage = .{ .storage = true, .transfer_src = true },
        .memory = .device_local,
    });
    defer ctx.destroyBuffer(output);
    
    // Upload
    try ctx.uploadBuffer(input, std.mem.sliceAsBytes(data));
    
    // Create pipeline and bind group
    const pipeline = try ctx.createComputePipeline(.{ .shader = ReductionShader });
    const bind_group = try ctx.createBindGroup(ReductionShader.BindGroup0, .{
        .input_buffer = input,
        .output_buffer = output,
        .n = @intCast(n),
    });
    
    // Dispatch
    var encoder = blaze.Encoder.init(ctx.allocator);
    defer encoder.deinit();
    
    encoder.setPipeline(pipeline);
    encoder.setBindGroup(0, bind_group);
    encoder.dispatch(@intCast(num_workgroups), 1, 1);
    
    try ctx.submit(encoder.finish(), null);
    ctx.waitIdle();
    
    // Read back partial sums and finish on CPU
    var partial_sums: [num_workgroups]f32 = undefined;
    try ctx.downloadBuffer(output, std.mem.sliceAsBytes(&partial_sums));
    
    var total: f32 = 0;
    for (partial_sums) |s| total += s;
    return total;
}
```

```wgsl
// reduction.wgsl
struct Uniforms {
    n: u32,
}

@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let i = global_id.x;
    
    // Load data (with bounds check)
    if (i < uniforms.n) {
        shared_data[local_id.x] = input_buffer[i];
    } else {
        shared_data[local_id.x] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_id.x < stride) {
            shared_data[local_id.x] += shared_data[local_id.x + stride];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (local_id.x == 0u) {
        output_buffer[workgroup_id.x] = shared_data[0];
    }
}
```

## Verdict

**BLAZE should include compute from day one.** It's the same API surface, same Vulkan primitives, same shader toolchain. The only architectural decision is whether to expose:
- Separate compute queues (for async compute)
- Headless/compute-only context creation

Both are straightforward additions that don't require a separate library.

---

# Question 3: Conformance Testing

## The Gold Standard: Khronos VK-GL-CTS

The Khronos Group maintains the official Vulkan Conformance Test Suite (CTS):

**Repository:** https://github.com/KhronosGroup/VK-GL-CTS

**Scope:**
- Thousands of tests for Vulkan API correctness
- Tests rendering, compute, memory, synchronization, extensions
- Required for Vulkan conformance certification
- Also includes OpenGL/OpenGL ES tests

**However:** This tests *drivers*, not *libraries built on Vulkan*. It's not directly applicable to BLAZE.

## WebGPU CTS

For WebGPU (relevant if you add a WebGPU backend):

**Repository:** https://github.com/gpuweb/cts

**Scope:**
- Tests WebGPU API behavior
- Tests WGSL shader compilation
- Tests memory, synchronization, rendering correctness
- Required for WebGPU conformant browsers

**Categories:**
- **API tests** - JavaScript/API surface coverage
- **Shader tests** - WGSL compilation and execution
- **IDL tests** - Interface correctness
- **Web Platform tests** - Browser integration

## What About Testing BLAZE Itself?

This is the real question. There isn't a "GPU abstraction layer conformance test" in the way there's a TOON format spec. But there are several approaches:

### Approach 1: Port Relevant VK-GL-CTS Tests

The VK-GL-CTS has specific test categories that *could* be adapted:

| CTS Category | Applicability to BLAZE |
|--------------|------------------------|
| `dEQP-VK.api.*` | Not directly (tests raw Vulkan API) |
| `dEQP-VK.memory.*` | Useful for allocator testing |
| `dEQP-VK.pipeline.*` | Useful for pipeline creation |
| `dEQP-VK.binding_model.*` | Useful for bind group testing |
| `dEQP-VK.renderpass.*` | Useful for render pass abstraction |
| `dEQP-VK.compute.*` | **Directly useful** for compute testing |
| `dEQP-VK.image.*` | Useful for texture operations |
| `dEQP-VK.synchronization.*` | **Critical** for sync testing |

You could extract the *logic* of these tests (not the Vulkan calls) and re-implement them against BLAZE's API.

### Approach 2: Graphics Test Frameworks

Several open-source projects provide rendering correctness tests:

**1. Sascha Willems' Vulkan Examples**
- https://github.com/SaschaWillems/Vulkan
- ~100 examples covering most Vulkan features
- Could be ported to BLAZE as functional tests

**2. WebGPU Samples**
- https://webgpu.github.io/webgpu-samples/
- Reference implementations of common rendering techniques
- Already in WGSL (your shader language)

**3. glTF Sample Models**
- https://github.com/KhronosGroup/glTF-Sample-Models
- Standard 3D models for testing loading and rendering
- Good for FORGE testing

### Approach 3: Build Your Own Conformance Suite

Given your experience with TOON conformance testing, you could create a structured test suite:

```
blaze-cts/
├── api/
│   ├── context/
│   │   ├── create_context.zig
│   │   ├── create_context_validation.zig
│   │   ├── create_context_features.zig
│   │   └── ...
│   ├── buffer/
│   │   ├── create_buffer.zig
│   │   ├── buffer_mapping.zig
│   │   ├── buffer_upload.zig
│   │   └── ...
│   ├── texture/
│   ├── pipeline/
│   ├── command/
│   └── sync/
├── compute/
│   ├── dispatch_basic.zig
│   ├── dispatch_indirect.zig
│   ├── workgroup_shared_memory.zig
│   ├── storage_buffer_read_write.zig
│   └── ...
├── render/
│   ├── clear_color.zig
│   ├── triangle.zig
│   ├── textured_quad.zig
│   ├── depth_test.zig
│   ├── blending.zig
│   └── ...
├── golden/
│   ├── triangle.png
│   ├── textured_quad.png
│   └── ...
└── runner.zig
```

**Test categories:**

| Category | # Tests (Target) | Description |
|----------|------------------|-------------|
| Context Creation | 20 | Device selection, feature requirements, error handling |
| Buffer Operations | 40 | Create, map, upload, download, usage flags |
| Texture Operations | 50 | Formats, mipmaps, sampling, storage |
| Pipeline Creation | 60 | Shader compilation, vertex layouts, blend states |
| Command Encoding | 30 | Recording, submission, synchronization |
| Synchronization | 40 | Fences, semaphores, barriers |
| Compute Dispatch | 50 | Basic, indirect, shared memory, workgroup sizes |
| Render Correctness | 100 | Clear, draw, blend, depth, scissor, viewport |
| **Total** | **~400** | |

### Approach 4: Golden Image Testing

For rendering correctness, compare output against known-good reference images:

```zig
// test/render/triangle.zig

const std = @import("std");
const blaze = @import("blaze");
const testing = @import("test_framework");

test "render triangle - basic" {
    var ctx = try testing.createTestContext();
    defer ctx.deinit();
    
    // Render triangle to offscreen texture
    const result = try renderTriangle(&ctx, .{
        .width = 256,
        .height = 256,
        .clear_color = .{ 0.0, 0.0, 0.0, 1.0 },
    });
    
    // Compare against golden image
    try testing.assertImageMatches(
        result,
        "golden/triangle_basic.png",
        .{ .tolerance = 0.01 },  // Allow 1% pixel difference
    );
}

test "render triangle - MSAA 4x" {
    // ...
    try testing.assertImageMatches(result, "golden/triangle_msaa4x.png", .{});
}
```

### Approach 5: Numerical Correctness for Compute

For compute shaders, test mathematical correctness:

```zig
test "compute - parallel sum reduction" {
    var ctx = try testing.createTestContext(.{ .mode = .compute_only });
    defer ctx.deinit();
    
    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const expected: f32 = 55.0;
    
    const result = try gpuSum(&ctx, &input);
    
    try std.testing.expectApproxEqAbs(expected, result, 0.001);
}

test "compute - matrix multiply" {
    // ...
    const expected = cpuMatMul(a, b);
    const result = try gpuMatMul(&ctx, a, b);
    
    try testing.assertMatricesEqual(expected, result, .{ .tolerance = 1e-5 });
}
```

## Recommended Conformance Strategy

### Phase 1: API Smoke Tests (~50 tests)
- Context creation/destruction
- Buffer CRUD
- Texture CRUD  
- Pipeline compilation
- Basic command submission

**Goal:** "Does it work at all?"

### Phase 2: Feature Coverage (~150 tests)
- All buffer usage combinations
- All texture formats you support
- All blend modes
- All depth/stencil modes
- Compute dispatch variations

**Goal:** "Does every feature work?"

### Phase 3: Correctness Testing (~150 tests)
- Golden image comparisons
- Numerical compute results
- Synchronization stress tests
- Multi-frame consistency

**Goal:** "Does it produce correct results?"

### Phase 4: Stress/Edge Cases (~50 tests)
- Resource limits
- Out-of-memory handling
- Invalid input handling
- Race condition detection

**Goal:** "Does it fail gracefully?"

## Example: BLAZE CTS Structure

```zig
// blaze-cts/src/runner.zig

const std = @import("std");
const blaze = @import("blaze");

const TestResult = struct {
    name: []const u8,
    passed: bool,
    duration_ns: u64,
    message: ?[]const u8,
};

const TestSuite = struct {
    name: []const u8,
    tests: []const TestFn,
};

const all_suites = [_]TestSuite{
    .{ .name = "api.context", .tests = @import("api/context.zig").tests },
    .{ .name = "api.buffer", .tests = @import("api/buffer.zig").tests },
    .{ .name = "api.texture", .tests = @import("api/texture.zig").tests },
    .{ .name = "api.pipeline", .tests = @import("api/pipeline.zig").tests },
    .{ .name = "compute.basic", .tests = @import("compute/basic.zig").tests },
    .{ .name = "compute.reduction", .tests = @import("compute/reduction.zig").tests },
    .{ .name = "render.clear", .tests = @import("render/clear.zig").tests },
    .{ .name = "render.triangle", .tests = @import("render/triangle.zig").tests },
    .{ .name = "render.texture", .tests = @import("render/texture.zig").tests },
    // ...
};

pub fn main() !void {
    var passed: usize = 0;
    var failed: usize = 0;
    var skipped: usize = 0;
    
    for (all_suites) |suite| {
        std.debug.print("\n=== {s} ===\n", .{suite.name});
        
        for (suite.tests) |test_fn| {
            const result = runTest(test_fn);
            
            if (result.passed) {
                passed += 1;
                std.debug.print("  ✓ {s}\n", .{result.name});
            } else {
                failed += 1;
                std.debug.print("  ✗ {s}: {s}\n", .{result.name, result.message orelse "unknown error"});
            }
        }
    }
    
    std.debug.print("\n");
    std.debug.print("═══════════════════════════════════════\n", .{});
    std.debug.print("BLAZE Conformance Test Suite v0.1.0\n", .{});
    std.debug.print("═══════════════════════════════════════\n", .{});
    std.debug.print("Passed:  {d}\n", .{passed});
    std.debug.print("Failed:  {d}\n", .{failed});
    std.debug.print("Skipped: {d}\n", .{skipped});
    std.debug.print("Total:   {d}\n", .{passed + failed + skipped});
    std.debug.print("═══════════════════════════════════════\n", .{});
    
    if (failed > 0) {
        std.process.exit(1);
    }
}
```

Output:
```
=== api.context ===
  ✓ create_default_context
  ✓ create_with_validation
  ✓ create_compute_only
  ✓ destroy_context

=== api.buffer ===
  ✓ create_vertex_buffer
  ✓ create_index_buffer
  ✓ create_uniform_buffer
  ✓ create_storage_buffer
  ✓ map_host_visible_buffer
  ✓ upload_to_device_local

=== compute.basic ===
  ✓ dispatch_1d
  ✓ dispatch_2d
  ✓ dispatch_3d
  ✓ dispatch_indirect

=== render.triangle ===
  ✓ triangle_basic
  ✓ triangle_colored
  ✓ triangle_msaa_4x
  ✗ triangle_with_depth: golden image mismatch (diff: 2.3%)

═══════════════════════════════════════
BLAZE Conformance Test Suite v0.1.0
═══════════════════════════════════════
Passed:  391
Failed:  1
Skipped: 0
Total:   392
═══════════════════════════════════════
```

---

# Summary

| Question | Answer |
|----------|--------|
| **WebAssembly?** | Add WebGPU backend; WGSL shaders already compatible. Phase it after Vulkan is solid. |
| **GPU Compute?** | Yes, BLAZE should include compute. Same API, same primitives. Add headless mode. |
| **Conformance?** | No exact equivalent to TOON spec. Build your own ~400 test suite covering API, compute, render correctness. |

## Recommended Priorities

1. **Now:** Build BLAZE Vulkan backend + basic CTS (~50 smoke tests)
2. **Soon:** Add compute support + compute CTS (~50 tests)
3. **Later:** Add WebGPU backend for WASM
4. **Ongoing:** Grow CTS to ~400 tests as features mature

The conformance suite will be as valuable as the library itself - it proves correctness and catches regressions as you iterate.
