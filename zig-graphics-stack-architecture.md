# Zig Graphics Stack Architecture

## Overview

This document specifies the architecture for a Zig-native graphics stack consisting of three libraries:

| Library | Purpose | Inspiration |
|---------|---------|-------------|
| **BLAZE** | Lean GPU abstraction over Vulkan | BLADE (kvark) |
| **FORGE** | 3D scene rendering, GPU-driven | BLADE-render + modern techniques |
| **FLUX** | UI framework, hybrid immediate/retained | GPUI concepts, Zig-native design |

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│         (Structural Engineering Software, Games, Tools)         │
├─────────────────────────────────────────────────────────────────┤
│                           FLUX                                  │
│    Declarative UI · Layout · Styling · Input · Text Rendering   │
├─────────────────────────────────────────────────────────────────┤
│                           FORGE                                 │
│   Scene Graph · Meshlets · GPU Culling · Materials · Lighting   │
├─────────────────────────────────────────────────────────────────┤
│                           BLAZE                                 │
│  Command Encoding · Pipelines · Memory · Sync · Shader Reflect  │
├─────────────────────────────────────────────────────────────────┤
│                     Vulkan / GLES (future)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles (All Libraries)

1. **Comptime over runtime** - Push decisions to compile time wherever possible
2. **Explicit over implicit** - No hidden allocations, no magic lifetimes
3. **Data-oriented** - Structs of arrays, cache-friendly layouts, minimal indirection
4. **Zero-cost abstractions** - Abstractions should compile away entirely
5. **Incremental complexity** - Simple things simple, complex things possible

---

# BLAZE

## Purpose

BLAZE is a lean, Zig-native GPU abstraction layer targeting Vulkan 1.3+ on Windows and Linux. It prioritizes ergonomics and modern GPU features over broad hardware compatibility.

**Non-goals:**
- Legacy hardware support (no Vulkan 1.0/1.1 fallbacks)
- Metal/D3D12 backends (may add later, not initial focus)
- Safety rails that prevent advanced usage

## Zig Strengths Leveraged

### 1. Comptime Shader Reflection

BLADE uses naga for runtime shader reflection. BLAZE goes further: parse WGSL at comptime to generate type-safe bind group layouts.

```zig
// shaders/quad.wgsl (embedded via @embedFile)
// @group(0) @binding(0) var<uniform> transform: mat4x4<f32>;
// @group(0) @binding(1) var texture: texture_2d<f32>;
// @group(0) @binding(2) var tex_sampler: sampler;

const QuadShader = blaze.Shader.compile(@embedFile("shaders/quad.wgsl"));

// Comptime-generated from shader reflection:
// QuadShader.BindGroup0 is a struct type
// QuadShader.bindings is known at comptime
// Type errors if you pass wrong uniform type

var bind_group = try ctx.createBindGroup(QuadShader.BindGroup0, .{
    .transform = &transform_buffer,  // Compile error if wrong type
    .texture = texture_view,
    .tex_sampler = sampler,
});
```

**Differentiation:** Rust BLADE does runtime reflection. BLAZE does comptime reflection, catching binding mismatches at compile time with zero runtime cost.

### 2. Comptime Pipeline Configuration

Pipeline state is verbose in Vulkan. BLAZE uses comptime to flatten configuration:

```zig
const pipeline = try ctx.createPipeline(.{
    .shader = QuadShader,
    .vertex_layout = VertexPCT,  // Comptime: extracts attributes from struct fields
    .topology = .triangle_list,
    .cull_mode = .back,
    .depth = .{ .test = true, .write = true, .compare = .less },
    .blend = .alpha,  // Preset, or specify full blend state
    .render_targets = &.{.rgba8},
});

// VertexPCT is just a struct - BLAZE derives vertex attributes at comptime
const VertexPCT = struct {
    position: [3]f32,  // location 0, format R32G32B32_SFLOAT
    color: [4]u8,      // location 1, format R8G8B8A8_UNORM (normalized via type)
    texcoord: [2]f32,  // location 2, format R32G32_SFLOAT
};
```

### 3. Tagged Unions for Command Encoding

Zig's tagged unions enable type-safe command recording without inheritance:

```zig
const Command = union(enum) {
    begin_render_pass: RenderPassBegin,
    end_render_pass: void,
    set_pipeline: Pipeline,
    set_bind_group: struct { index: u32, group: BindGroup },
    set_vertex_buffer: struct { slot: u32, buffer: Buffer, offset: u64 },
    set_index_buffer: struct { buffer: Buffer, offset: u64, index_type: IndexType },
    draw: struct { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 },
    draw_indexed: struct { index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32 },
    draw_indirect: struct { buffer: Buffer, offset: u64, draw_count: u32, stride: u32 },
    dispatch: struct { x: u32, y: u32, z: u32 },
    copy_buffer: struct { src: Buffer, dst: Buffer, regions: []const BufferCopy },
    // ...
};
```

### 4. Error Sets for Precise Failure Handling

```zig
pub const CreateBufferError = error{
    OutOfDeviceMemory,
    OutOfHostMemory,
    InvalidSize,
    UnsupportedUsage,
};

pub const SubmitError = error{
    DeviceLost,
    OutOfHostMemory,
    OutOfDeviceMemory,
};

// Callers know exactly what can fail
fn uploadMesh(ctx: *Context, data: []const u8) CreateBufferError!Buffer {
    return ctx.createBuffer(.{
        .size = data.len,
        .usage = .{ .vertex = true, .transfer_dst = true },
        .memory = .device_local,
    });
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BLAZE Public API                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Context    │  Resources   │   Commands   │     Sync       │
│  ─────────   │  ──────────  │  ──────────  │  ──────────    │
│  init/deinit │  Buffer      │  Encoder     │  Fence         │
│  createXxx   │  Texture     │  submit      │  Semaphore     │
│  submit      │  Sampler     │  Commands    │  Timeline      │
│              │  Pipeline    │              │                │
│              │  BindGroup   │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Vulkan Backend                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Instance    │  Allocator   │  Cmd Pool    │  Sync Pool     │
│  Device      │  (VMA-style) │  Cmd Buffers │  Query Pool    │
│  Queues      │              │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

## Core Types

### Context

The entry point. Owns Vulkan instance, device, queues, and internal allocators.

```zig
pub const Context = struct {
    // Opaque - users don't access internals
    
    pub const Config = struct {
        app_name: []const u8 = "BLAZE App",
        validation: bool = builtin.mode == .Debug,
        preferred_device: ?[]const u8 = null,  // Substring match on device name
        features: Features = .{},
    };
    
    pub const Features = struct {
        timeline_semaphores: bool = true,      // Required for modern sync
        dynamic_rendering: bool = true,        // No render pass objects
        descriptor_indexing: bool = true,      // Bindless-style access
        buffer_device_address: bool = true,    // GPU pointers
        mesh_shaders: bool = false,            // Optional, for FORGE
        ray_tracing: bool = false,             // Optional, future
    };
    
    pub fn init(allocator: Allocator, config: Config) !Context;
    pub fn deinit(self: *Context) void;
    
    // Resource creation
    pub fn createBuffer(self: *Context, desc: BufferDesc) CreateBufferError!Buffer;
    pub fn createTexture(self: *Context, desc: TextureDesc) CreateTextureError!Texture;
    pub fn createSampler(self: *Context, desc: SamplerDesc) CreateSamplerError!Sampler;
    pub fn createPipeline(self: *Context, desc: anytype) CreatePipelineError!Pipeline;
    pub fn createBindGroup(self: *Context, comptime Layout: type, bindings: Layout) !BindGroup;
    
    // Destruction (or use defer patterns)
    pub fn destroyBuffer(self: *Context, buffer: Buffer) void;
    // ... etc
    
    // Command submission
    pub fn submit(self: *Context, commands: []const Command, signal: ?*TimelineSemaphore) SubmitError!void;
    pub fn present(self: *Context, swapchain: Swapchain) PresentError!void;
    
    // Sync
    pub fn waitIdle(self: *Context) void;
    pub fn waitTimeline(self: *Context, semaphore: *TimelineSemaphore, value: u64) void;
};
```

### Buffer

```zig
pub const Buffer = struct {
    handle: u64,  // Opaque
    size: u64,
    mapped: ?[*]u8,  // Non-null if host-visible and mapped
    
    pub fn slice(self: Buffer, offset: u64, len: u64) BufferSlice {
        return .{ .buffer = self, .offset = offset, .size = len };
    }
    
    // Convenience for mapped buffers
    pub fn write(self: Buffer, comptime T: type, data: []const T) void {
        std.debug.assert(self.mapped != null);
        const bytes = std.mem.sliceAsBytes(data);
        @memcpy(self.mapped.?[0..bytes.len], bytes);
    }
};

pub const BufferDesc = struct {
    size: u64,
    usage: BufferUsage,
    memory: MemoryLocation,
    label: ?[]const u8 = null,
};

pub const BufferUsage = packed struct {
    vertex: bool = false,
    index: bool = false,
    uniform: bool = false,
    storage: bool = false,
    indirect: bool = false,
    transfer_src: bool = false,
    transfer_dst: bool = false,
};

pub const MemoryLocation = enum {
    device_local,      // Fast GPU memory, not CPU accessible
    host_visible,      // CPU accessible, coherent
    host_cached,       // CPU accessible, cached (good for readback)
    auto,              // Let allocator decide based on usage
};
```

### Texture

```zig
pub const Texture = struct {
    handle: u64,
    extent: Extent3D,
    format: Format,
    mip_levels: u32,
    array_layers: u32,
    
    pub fn view(self: Texture, desc: TextureViewDesc) TextureView;
};

pub const TextureDesc = struct {
    extent: Extent3D,
    format: Format,
    usage: TextureUsage,
    mip_levels: u32 = 1,
    array_layers: u32 = 1,
    samples: u32 = 1,
    label: ?[]const u8 = null,
};

pub const Format = enum(u32) {
    rgba8_unorm,
    rgba8_srgb,
    bgra8_unorm,
    bgra8_srgb,
    r8_unorm,
    rg8_unorm,
    rgba16_float,
    rgba32_float,
    r32_float,
    rg32_float,
    depth32_float,
    depth24_stencil8,
    // ... etc
    
    pub fn bytesPerPixel(self: Format) u32 { ... }
    pub fn isDepth(self: Format) bool { ... }
    pub fn isSrgb(self: Format) bool { ... }
};
```

### Command Encoding

BLAZE uses a simple command list model rather than Vulkan's complex command buffer management:

```zig
pub const Encoder = struct {
    allocator: Allocator,
    commands: std.ArrayList(Command),
    
    pub fn init(allocator: Allocator) Encoder {
        return .{
            .allocator = allocator,
            .commands = std.ArrayList(Command).init(allocator),
        };
    }
    
    pub fn deinit(self: *Encoder) void {
        self.commands.deinit();
    }
    
    pub fn reset(self: *Encoder) void {
        self.commands.clearRetainingCapacity();
    }
    
    pub fn finish(self: *Encoder) []const Command {
        return self.commands.items;
    }
    
    // Render pass (dynamic rendering - no render pass objects)
    pub fn beginRenderPass(self: *Encoder, desc: RenderPassDesc) void {
        self.commands.append(.{ .begin_render_pass = desc }) catch unreachable;
    }
    
    pub fn endRenderPass(self: *Encoder) void {
        self.commands.append(.end_render_pass) catch unreachable;
    }
    
    // State
    pub fn setPipeline(self: *Encoder, pipeline: Pipeline) void {
        self.commands.append(.{ .set_pipeline = pipeline }) catch unreachable;
    }
    
    pub fn setBindGroup(self: *Encoder, index: u32, group: BindGroup) void {
        self.commands.append(.{ .set_bind_group = .{ .index = index, .group = group } }) catch unreachable;
    }
    
    pub fn setVertexBuffer(self: *Encoder, slot: u32, buffer: Buffer, offset: u64) void {
        self.commands.append(.{ .set_vertex_buffer = .{ .slot = slot, .buffer = buffer, .offset = offset } }) catch unreachable;
    }
    
    pub fn setIndexBuffer(self: *Encoder, buffer: Buffer, offset: u64, index_type: IndexType) void {
        self.commands.append(.{ .set_index_buffer = .{ .buffer = buffer, .offset = offset, .index_type = index_type } }) catch unreachable;
    }
    
    pub fn setScissor(self: *Encoder, rect: Rect2D) void {
        self.commands.append(.{ .set_scissor = rect }) catch unreachable;
    }
    
    pub fn setViewport(self: *Encoder, viewport: Viewport) void {
        self.commands.append(.{ .set_viewport = viewport }) catch unreachable;
    }
    
    // Draw
    pub fn draw(self: *Encoder, vertex_count: u32, instance_count: u32) void {
        self.commands.append(.{ .draw = .{
            .vertex_count = vertex_count,
            .instance_count = instance_count,
            .first_vertex = 0,
            .first_instance = 0,
        } }) catch unreachable;
    }
    
    pub fn drawIndexed(self: *Encoder, index_count: u32, instance_count: u32) void {
        self.commands.append(.{ .draw_indexed = .{
            .index_count = index_count,
            .instance_count = instance_count,
            .first_index = 0,
            .vertex_offset = 0,
            .first_instance = 0,
        } }) catch unreachable;
    }
    
    pub fn drawIndirect(self: *Encoder, buffer: Buffer, offset: u64, draw_count: u32) void {
        self.commands.append(.{ .draw_indirect = .{
            .buffer = buffer,
            .offset = offset,
            .draw_count = draw_count,
            .stride = @sizeOf(DrawIndirectCommand),
        } }) catch unreachable;
    }
    
    pub fn drawIndexedIndirect(self: *Encoder, buffer: Buffer, offset: u64, draw_count: u32) void {
        self.commands.append(.{ .draw_indexed_indirect = .{
            .buffer = buffer,
            .offset = offset,
            .draw_count = draw_count,
            .stride = @sizeOf(DrawIndexedIndirectCommand),
        } }) catch unreachable;
    }
    
    // Compute
    pub fn dispatch(self: *Encoder, x: u32, y: u32, z: u32) void {
        self.commands.append(.{ .dispatch = .{ .x = x, .y = y, .z = z } }) catch unreachable;
    }
    
    pub fn dispatchIndirect(self: *Encoder, buffer: Buffer, offset: u64) void {
        self.commands.append(.{ .dispatch_indirect = .{ .buffer = buffer, .offset = offset } }) catch unreachable;
    }
    
    // Transfers
    pub fn copyBuffer(self: *Encoder, src: Buffer, dst: Buffer, regions: []const BufferCopy) void {
        self.commands.append(.{ .copy_buffer = .{ .src = src, .dst = dst, .regions = regions } }) catch unreachable;
    }
    
    pub fn copyBufferToTexture(self: *Encoder, src: Buffer, dst: Texture, regions: []const BufferTextureCopy) void {
        self.commands.append(.{ .copy_buffer_to_texture = .{ .src = src, .dst = dst, .regions = regions } }) catch unreachable;
    }
};
```

### Synchronization

BLAZE uses timeline semaphores as the primary sync primitive (Vulkan 1.2+):

```zig
pub const TimelineSemaphore = struct {
    handle: u64,
    value: u64,  // Current signaled value
    
    pub fn signal(self: *TimelineSemaphore) u64 {
        self.value += 1;
        return self.value;
    }
};

// Usage pattern:
var timeline = try ctx.createTimelineSemaphore(0);

// Frame N
encoder.beginRenderPass(...);
// ... record commands
encoder.endRenderPass();
const signal_value = timeline.signal();
try ctx.submit(encoder.finish(), &timeline);

// Frame N+1 can wait on frame N
ctx.waitTimeline(&timeline, signal_value);
```

## Shader Compilation

BLAZE compiles WGSL to SPIR-V at build time using naga (via C bindings or a build step).

```zig
// build.zig
const blaze = @import("blaze");

pub fn build(b: *std.Build) void {
    // ...
    
    // Compile shaders at build time
    const shader_step = blaze.addShaderCompilation(b, .{
        .sources = &.{
            "shaders/quad.wgsl",
            "shaders/mesh.wgsl",
            "shaders/compute_cull.wgsl",
        },
        .output_dir = "zig-cache/shaders",
    });
    
    exe.step.dependOn(&shader_step.step);
}
```

```zig
// In application code
const QuadShader = blaze.Shader.load(@embedFile("zig-cache/shaders/quad.spv"));

// QuadShader now has comptime-known:
// - BindGroup layouts (derived from reflection data embedded in SPIR-V)
// - Push constant layout
// - Vertex input layout (if vertex shader)
```

## Memory Management

BLAZE includes a VMA-style sub-allocator:

```zig
// Internal - users don't interact directly
const Allocator = struct {
    // Pools for different memory types
    device_local: MemoryPool,
    host_visible: MemoryPool,
    host_cached: MemoryPool,
    
    // Large allocations get dedicated memory
    dedicated_threshold: u64 = 256 * 1024 * 1024,  // 256 MB
    
    pub fn allocate(self: *Allocator, requirements: MemoryRequirements, location: MemoryLocation) !Allocation;
    pub fn free(self: *Allocator, allocation: Allocation) void;
};
```

## Swapchain / Windowing

BLAZE is windowing-agnostic. It accepts platform handles:

```zig
pub const SurfaceDesc = struct {
    // Platform-specific window handle
    native_handle: NativeHandle,
    
    pub const NativeHandle = union(enum) {
        xlib: struct { display: *anyopaque, window: u64 },
        xcb: struct { connection: *anyopaque, window: u32 },
        wayland: struct { display: *anyopaque, surface: *anyopaque },
        win32: struct { hinstance: *anyopaque, hwnd: *anyopaque },
    };
};

pub const Swapchain = struct {
    handle: u64,
    format: Format,
    extent: Extent2D,
    images: []Texture,
    
    pub fn acquireNextImage(self: *Swapchain, timeout_ns: u64) !u32;
    pub fn currentImage(self: *Swapchain) Texture;
};

pub const SwapchainDesc = struct {
    surface: Surface,
    extent: Extent2D,
    format: Format = .bgra8_srgb,
    present_mode: PresentMode = .fifo,  // vsync
    image_count: u32 = 3,               // triple buffer
};
```

## Example: Triangle

```zig
const std = @import("std");
const blaze = @import("blaze");

const TriangleShader = blaze.Shader.compile(@embedFile("triangle.wgsl"));

const Vertex = struct {
    position: [2]f32,
    color: [3]f32,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize
    var ctx = try blaze.Context.init(allocator, .{});
    defer ctx.deinit();
    
    // Create window surface (pseudo-code, actual windowing is external)
    const surface = try ctx.createSurface(.{ .native_handle = getWindowHandle() });
    var swapchain = try ctx.createSwapchain(.{ .surface = surface, .extent = .{ .width = 800, .height = 600 } });
    
    // Create pipeline
    const pipeline = try ctx.createPipeline(.{
        .shader = TriangleShader,
        .vertex_layout = Vertex,
        .render_targets = &.{swapchain.format},
    });
    
    // Create vertex buffer
    const vertices = [_]Vertex{
        .{ .position = .{ 0.0, -0.5 }, .color = .{ 1, 0, 0 } },
        .{ .position = .{ 0.5, 0.5 }, .color = .{ 0, 1, 0 } },
        .{ .position = .{ -0.5, 0.5 }, .color = .{ 0, 0, 1 } },
    };
    
    const vbo = try ctx.createBuffer(.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .vertex = true, .transfer_dst = true },
        .memory = .device_local,
    });
    
    // Upload (staging handled internally for device_local)
    try ctx.uploadBuffer(vbo, std.mem.asBytes(&vertices));
    
    // Main loop
    var encoder = blaze.Encoder.init(allocator);
    defer encoder.deinit();
    
    while (running) {
        const image_index = try swapchain.acquireNextImage(std.math.maxInt(u64));
        
        encoder.reset();
        encoder.beginRenderPass(.{
            .color_attachments = &.{.{
                .view = swapchain.images[image_index].view(.{}),
                .load_op = .clear,
                .store_op = .store,
                .clear_value = .{ .color = .{ 0.1, 0.1, 0.1, 1.0 } },
            }},
        });
        
        encoder.setPipeline(pipeline);
        encoder.setVertexBuffer(0, vbo, 0);
        encoder.draw(3, 1);
        
        encoder.endRenderPass();
        
        try ctx.submit(encoder.finish(), null);
        try ctx.present(&swapchain);
    }
}
```

---

# FORGE

## Purpose

FORGE is a GPU-driven 3D scene renderer built on BLAZE. It implements modern rendering techniques: meshlets, GPU culling, indirect rendering, and material batching.

**Target use cases:**
- WoW-style game rendering (many objects, stylized art, performance over realism)
- Structural engineering visualization (CAD-like, interactive manipulation)
- General 3D applications

## Zig Strengths Leveraged

### 1. Comptime Scene Configuration

Define what your scene supports at compile time:

```zig
const MyScene = forge.Scene(.{
    .max_objects = 100_000,
    .max_meshlets = 1_000_000,
    .max_materials = 1024,
    .max_lights = 256,
    .features = .{
        .shadows = true,
        .ambient_occlusion = false,
        .bloom = true,
    },
});

// MyScene is a concrete type with:
// - Appropriately sized buffers
// - Only the shader permutations you need
// - No runtime feature checks in hot paths
```

### 2. Data-Oriented Entity Storage

Zig's explicit memory layout enables cache-efficient entity storage:

```zig
// Struct of Arrays for cache efficiency during culling
pub fn ObjectStorage(comptime capacity: u32) type {
    return struct {
        // Transform data - accessed together during culling
        positions: [capacity][3]f32,
        rotations: [capacity][4]f32,  // quaternion
        scales: [capacity][3]f32,
        
        // Bounding data - accessed during culling
        bounding_spheres: [capacity][4]f32,  // xyz = center, w = radius
        
        // Render data - accessed during draw
        mesh_ids: [capacity]u16,
        material_ids: [capacity]u16,
        flags: [capacity]ObjectFlags,
        
        // Active count
        count: u32,
        
        pub fn add(self: *@This(), desc: ObjectDesc) ?ObjectHandle { ... }
        pub fn remove(self: *@This(), handle: ObjectHandle) void { ... }
    };
}
```

### 3. Inline GPU Structs

Zig's `extern struct` and `packed struct` give exact control over GPU buffer layouts:

```zig
// These match shader structs exactly - no padding surprises
pub const GpuObjectData = extern struct {
    world_matrix: [4][4]f32,
    prev_world_matrix: [4][4]f32,  // For motion vectors
    bounding_sphere: [4]f32,
    mesh_id: u32,
    material_id: u32,
    flags: u32,
    _padding: u32,
};

comptime {
    std.debug.assert(@sizeOf(GpuObjectData) == 160);
    std.debug.assert(@alignOf(GpuObjectData) == 4);
}

pub const DrawIndirectCommand = extern struct {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
};
```

### 4. Comptime Material System

Materials are comptime-configurable shader programs:

```zig
pub const Material = struct {
    // Base properties every material has
    base_color: [4]f32 = .{ 1, 1, 1, 1 },
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    
    // Texture bindings (null = use default)
    albedo_texture: ?TextureHandle = null,
    normal_texture: ?TextureHandle = null,
    metallic_roughness_texture: ?TextureHandle = null,
    
    // Shader variant
    shader: ShaderVariant = .standard,
    
    pub const ShaderVariant = enum {
        standard,       // PBR
        unlit,          // No lighting
        stylized,       // Toon/cel shading
        terrain,        // Multi-texture blending
        vegetation,     // Wind animation
    };
};

// At comptime, FORGE generates only the shader permutations you use
const scene = forge.Scene(.{
    .shader_variants = .{ .standard, .unlit, .stylized },  // Only these 3
});
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scene (User-Facing)                       │
│  ─────────────────────────────────────────────────────────────  │
│  addObject() / removeObject()     Camera management              │
│  addLight() / removeLight()       Material management            │
│  render(encoder, target)          Mesh/meshlet management        │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │   Culler     │ │   Drawer     │ │   Lighter    │
         │  ──────────  │ │  ──────────  │ │  ──────────  │
         │  Frustum     │ │  Indirect    │ │  Clustering  │
         │  Occlusion   │ │  Draws       │ │  Shadows     │
         │  LOD         │ │  Multi-draw  │ │  Probes      │
         └──────────────┘ └──────────────┘ └──────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │        GPU Buffers           │
                    │  ──────────────────────────  │
                    │  Object buffer (transforms)  │
                    │  Meshlet buffer              │
                    │  Draw indirect buffer        │
                    │  Material buffer             │
                    │  Light buffer                │
                    └──────────────────────────────┘
                                   │
                                   ▼
                              BLAZE Submit
```

## Core Types

### Scene

```zig
pub fn Scene(comptime config: SceneConfig) type {
    return struct {
        const Self = @This();
        
        // GPU resources
        ctx: *blaze.Context,
        object_buffer: blaze.Buffer,
        meshlet_buffer: blaze.Buffer,
        draw_buffer: blaze.Buffer,
        material_buffer: blaze.Buffer,
        light_buffer: blaze.Buffer,
        
        // CPU-side storage (mirrors GPU for updates)
        objects: ObjectStorage(config.max_objects),
        meshlets: MeshletStorage(config.max_meshlets),
        materials: MaterialStorage(config.max_materials),
        lights: LightStorage(config.max_lights),
        
        // Pipelines (generated at init based on config)
        cull_pipeline: blaze.Pipeline,
        draw_pipelines: [config.shader_variants.len]blaze.Pipeline,
        
        // Camera
        camera: Camera,
        
        pub fn init(ctx: *blaze.Context) !Self { ... }
        pub fn deinit(self: *Self) void { ... }
        
        // Object management
        pub fn addObject(self: *Self, desc: ObjectDesc) ?ObjectHandle {
            const handle = self.objects.add(desc) orelse return null;
            self.markObjectsDirty();
            return handle;
        }
        
        pub fn removeObject(self: *Self, handle: ObjectHandle) void {
            self.objects.remove(handle);
            self.markObjectsDirty();
        }
        
        pub fn setObjectTransform(self: *Self, handle: ObjectHandle, transform: Transform) void {
            self.objects.setTransform(handle, transform);
            self.markObjectsDirty();
        }
        
        // Mesh management
        pub fn loadMesh(self: *Self, path: []const u8) !MeshHandle { ... }
        pub fn uploadMesh(self: *Self, vertices: []const Vertex, indices: []const u32) !MeshHandle { ... }
        
        // Material management  
        pub fn createMaterial(self: *Self, desc: Material) !MaterialHandle { ... }
        pub fn updateMaterial(self: *Self, handle: MaterialHandle, desc: Material) void { ... }
        
        // Rendering
        pub fn render(self: *Self, encoder: *blaze.Encoder, target: RenderTarget) void {
            self.uploadDirtyData();
            self.cullPass(encoder);
            self.drawPass(encoder, target);
        }
        
        fn cullPass(self: *Self, encoder: *blaze.Encoder) void {
            // GPU compute shader culls objects, writes to draw_buffer
            encoder.setPipeline(self.cull_pipeline);
            encoder.setBindGroup(0, self.cull_bind_group);
            encoder.dispatch(
                (self.objects.count + 63) / 64,  // 64 threads per workgroup
                1,
                1,
            );
            
            // Barrier before drawing
            encoder.memoryBarrier(.{
                .src = .{ .compute_write = true },
                .dst = .{ .indirect_read = true, .vertex_read = true },
            });
        }
        
        fn drawPass(self: *Self, encoder: *blaze.Encoder, target: RenderTarget) void {
            encoder.beginRenderPass(.{
                .color_attachments = &.{.{
                    .view = target.color,
                    .load_op = .clear,
                    .store_op = .store,
                    .clear_value = .{ .color = .{ 0.1, 0.1, 0.15, 1.0 } },
                }},
                .depth_attachment = .{
                    .view = target.depth,
                    .load_op = .clear,
                    .store_op = .store,
                    .clear_value = .{ .depth = 1.0 },
                },
            });
            
            // One indirect draw per shader variant (materials batched)
            for (self.draw_pipelines, 0..) |pipeline, variant_idx| {
                encoder.setPipeline(pipeline);
                encoder.setBindGroup(0, self.scene_bind_group);
                encoder.drawIndexedIndirect(
                    self.draw_buffer,
                    variant_idx * @sizeOf(DrawIndirectCommand),
                    1,  // Single multi-draw per variant
                );
            }
            
            encoder.endRenderPass();
        }
    };
}

pub const SceneConfig = struct {
    max_objects: u32 = 100_000,
    max_meshlets: u32 = 1_000_000,
    max_materials: u32 = 1024,
    max_lights: u32 = 256,
    max_textures: u32 = 4096,
    
    shader_variants: []const Material.ShaderVariant = &.{ .standard },
    
    features: Features = .{},
    
    pub const Features = struct {
        shadows: bool = true,
        shadow_cascades: u32 = 4,
        ambient_occlusion: bool = false,
        bloom: bool = false,
        motion_blur: bool = false,
    };
};
```

### Meshlets

Meshlets are the core of GPU-driven rendering - small chunks of a mesh that can be independently culled:

```zig
pub const Meshlet = extern struct {
    // Indices into the mesh's vertex/index buffers
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    
    // Bounding for culling
    bounding_sphere: [4]f32,  // xyz = center (object space), w = radius
    cone_axis: [3]f32,        // Normal cone for backface culling
    cone_cutoff: f32,         // cos(angle) - if view dot axis > cutoff, cull
};

pub const Mesh = struct {
    vertices: blaze.Buffer,
    indices: blaze.Buffer,
    meshlets: []Meshlet,
    
    // Metadata
    vertex_count: u32,
    index_count: u32,
    meshlet_count: u32,
    bounding_sphere: [4]f32,
};

// Meshletization happens at load time
pub fn meshletize(
    vertices: []const Vertex,
    indices: []const u32,
    max_vertices_per_meshlet: u32,
    max_triangles_per_meshlet: u32,
) []Meshlet {
    // Implementation uses meshoptimizer algorithm
    // Groups triangles into meshlets with good locality
    // Computes bounding spheres and normal cones
    ...
}
```

### GPU Culling Shader

```wgsl
// compute_cull.wgsl

struct ObjectData {
    world_matrix: mat4x4<f32>,
    bounding_sphere: vec4<f32>,  // xyz = center, w = radius
    mesh_id: u32,
    material_id: u32,
    flags: u32,
    pad: u32,
}

struct DrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

struct CullUniforms {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_position: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> objects: array<ObjectData>;
@group(0) @binding(2) var<storage, read_write> draw_commands: array<DrawCommand>;
@group(0) @binding(3) var<storage, read_write> draw_count: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let object_idx = id.x;
    if (object_idx >= arrayLength(&objects)) {
        return;
    }
    
    let obj = objects[object_idx];
    
    // Transform bounding sphere to world space
    let world_center = (obj.world_matrix * vec4<f32>(obj.bounding_sphere.xyz, 1.0)).xyz;
    let world_radius = obj.bounding_sphere.w * length(obj.world_matrix[0].xyz);  // Assumes uniform scale
    
    // Frustum culling
    for (var i = 0u; i < 6u; i++) {
        let plane = uniforms.frustum_planes[i];
        let dist = dot(plane.xyz, world_center) + plane.w;
        if (dist < -world_radius) {
            return;  // Outside frustum
        }
    }
    
    // Object is visible - emit draw command
    let draw_idx = atomicAdd(&draw_count, 1u);
    
    // Look up mesh data and emit draw
    // (In practice, you'd have a meshlet buffer and emit per-meshlet)
    draw_commands[draw_idx] = DrawCommand(
        get_mesh_index_count(obj.mesh_id),
        1u,
        get_mesh_first_index(obj.mesh_id),
        i32(get_mesh_vertex_offset(obj.mesh_id)),
        object_idx,
    );
}
```

### Camera

```zig
pub const Camera = struct {
    position: [3]f32,
    rotation: [4]f32,  // Quaternion
    
    // Projection
    fov_y: f32 = std.math.degreesToRadians(60.0),
    aspect: f32 = 16.0 / 9.0,
    near: f32 = 0.1,
    far: f32 = 1000.0,
    
    pub fn viewMatrix(self: Camera) [4][4]f32 { ... }
    pub fn projectionMatrix(self: Camera) [4][4]f32 { ... }
    pub fn viewProjectionMatrix(self: Camera) [4][4]f32 { ... }
    pub fn frustumPlanes(self: Camera) [6][4]f32 { ... }
    
    // Helpers
    pub fn lookAt(self: *Camera, target: [3]f32) void { ... }
    pub fn orbit(self: *Camera, target: [3]f32, yaw: f32, pitch: f32, distance: f32) void { ... }
};
```

## Example: Structural Visualization

```zig
const std = @import("std");
const blaze = @import("blaze");
const forge = @import("forge");

const Scene = forge.Scene(.{
    .max_objects = 10_000,
    .shader_variants = &.{ .standard, .unlit },
    .features = .{ .shadows = true },
});

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var ctx = try blaze.Context.init(allocator, .{});
    defer ctx.deinit();
    
    var scene = try Scene.init(&ctx);
    defer scene.deinit();
    
    // Load beam mesh
    const beam_mesh = try scene.loadMesh("assets/beam.gltf");
    
    // Create materials
    const steel_mat = try scene.createMaterial(.{
        .base_color = .{ 0.8, 0.8, 0.85, 1.0 },
        .metallic = 0.9,
        .roughness = 0.3,
    });
    
    const concrete_mat = try scene.createMaterial(.{
        .base_color = .{ 0.7, 0.7, 0.65, 1.0 },
        .metallic = 0.0,
        .roughness = 0.9,
    });
    
    // Add beams from structural model
    for (structural_model.beams) |beam| {
        _ = scene.addObject(.{
            .mesh = beam_mesh,
            .material = steel_mat,
            .transform = .{
                .position = beam.start_point,
                .rotation = beam.orientation,
                .scale = .{ beam.length, beam.depth, beam.width },
            },
        });
    }
    
    // Setup camera
    scene.camera = .{
        .position = .{ 20, 15, 20 },
        .fov_y = std.math.degreesToRadians(45.0),
    };
    scene.camera.lookAt(.{ 0, 5, 0 });
    
    // Render loop
    var encoder = blaze.Encoder.init(allocator);
    defer encoder.deinit();
    
    while (running) {
        encoder.reset();
        scene.render(&encoder, .{
            .color = swapchain.currentImage().view(.{}),
            .depth = depth_texture.view(.{}),
        });
        try ctx.submit(encoder.finish(), null);
        try ctx.present(&swapchain);
    }
}
```

---

# FLUX

## Purpose

FLUX is a UI framework for building high-performance desktop applications. It combines declarative UI definition with GPU-accelerated rendering via BLAZE.

**Target use cases:**
- Engineering software (forms, data tables, visualizations)
- Game UI (HUD, menus, debug tools)
- Desktop tools and utilities

## Design Philosophy

FLUX is **hybrid immediate/retained**, similar to GPUI, but with Zig-native patterns:

1. **Declarative views** - You describe what UI should look like given state
2. **Retained structure** - FLUX maintains element tree, diffs on updates
3. **Immediate rendering** - Each frame, visible elements emit draw commands
4. **GPU-batched drawing** - All UI primitives batched into minimal draw calls

This gives you:
- React-like developer experience (declare UI from state)
- Retained mode efficiency (only update what changed)
- Immediate mode simplicity (no complex retained tree management)
- GPU performance (everything is instanced rendering)

## Zig Strengths Leveraged

### 1. Comptime Element Trees

Elements are comptime-known types, not heap-allocated trait objects:

```zig
// FLUX builds element trees at comptime
fn view(state: *AppState) flux.Element {
    return flux.column(.{}, .{
        flux.text("Hello", .{ .size = 24 }),
        flux.button("Click me", .{ .on_click = state.increment }),
        flux.text(state.counter_text(), .{}),
    });
}

// At comptime, this generates a concrete tree type:
// struct {
//     kind: .column,
//     children: struct {
//         @"0": struct { kind: .text, props: ... },
//         @"1": struct { kind: .button, props: ... },
//         @"2": struct { kind: .text, props: ... },
//     },
// }
```

**Why this matters:** No allocations for UI tree, perfect cache locality, compiler can inline element code.

### 2. Function Pointers Over Closures

Rust closures capture state implicitly. FLUX uses explicit function pointers with context:

```zig
// Zig style: explicit context
const Button = struct {
    label: []const u8,
    on_click: ?*const fn (*anyopaque) void = null,
    on_click_ctx: ?*anyopaque = null,
    
    pub fn click(self: Button) void {
        if (self.on_click) |handler| {
            handler(self.on_click_ctx.?);
        }
    }
};

// Usage - wrap method in helper
const state: *AppState = ...;
flux.button("Click", .{
    .on_click = AppState.increment,
    .on_click_ctx = state,
});

// Or use a comptime helper to generate wrapper
flux.button("Click", .{
    .on_click = flux.callback(state, AppState.increment),
});
```

### 3. Arena Allocator for Frame Data

FLUX uses frame-scoped arena allocation for temporary UI data:

```zig
pub const Frame = struct {
    arena: std.heap.ArenaAllocator,
    
    pub fn begin(backing: Allocator) Frame {
        return .{ .arena = std.heap.ArenaAllocator.init(backing) };
    }
    
    pub fn end(self: *Frame) void {
        self.arena.deinit();
    }
    
    // All UI allocations use frame arena - freed in bulk at frame end
    pub fn alloc(self: *Frame) Allocator {
        return self.arena.allocator();
    }
};

// Typical frame:
var frame = Frame.begin(gpa);
defer frame.end();

const root = view(&state);
const layout = computeLayout(root, window_size, frame.alloc());
const draws = paint(root, layout, frame.alloc());
// ... submit draws
// Frame ends - all allocations freed instantly
```

### 4. Inline Styles (No CSS Runtime)

Styles are comptime structs, not runtime-parsed strings:

```zig
pub const Style = struct {
    // Sizing
    width: Dimension = .auto,
    height: Dimension = .auto,
    min_width: ?f32 = null,
    max_width: ?f32 = null,
    min_height: ?f32 = null,
    max_height: ?f32 = null,
    
    // Spacing
    padding: Edges = .{},
    margin: Edges = .{},
    
    // Flexbox
    flex_direction: FlexDirection = .row,
    justify_content: JustifyContent = .start,
    align_items: AlignItems = .stretch,
    flex_grow: f32 = 0,
    flex_shrink: f32 = 1,
    gap: f32 = 0,
    
    // Visual
    background: ?Color = null,
    border: ?Border = null,
    border_radius: f32 = 0,
    
    // Text
    font_size: f32 = 14,
    font_weight: FontWeight = .normal,
    text_color: Color = .{ .r = 0, .g = 0, .b = 0, .a = 255 },
    text_align: TextAlign = .left,
};

// Comptime style presets
pub const styles = struct {
    pub const card = Style{
        .padding = .all(16),
        .background = .{ .r = 255, .g = 255, .b = 255, .a = 255 },
        .border_radius = 8,
        .border = .{ .width = 1, .color = .{ .r = 200, .g = 200, .b = 200, .a = 255 } },
    };
    
    pub const primary_button = Style{
        .padding = .{ .horizontal = 16, .vertical = 8 },
        .background = .{ .r = 59, .g = 130, .b = 246, .a = 255 },
        .text_color = .{ .r = 255, .g = 255, .b = 255, .a = 255 },
        .border_radius = 4,
    };
};
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application                                 │
│  ──────────────────────────────────────────────────────────────  │
│  State struct       view(state) -> Element       Event handlers  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Element Tree (Comptime)                      │
│  ──────────────────────────────────────────────────────────────  │
│  Declarative description of UI structure and properties          │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌────────────┐      ┌────────────┐      ┌────────────┐
     │   Layout   │      │   Input    │      │   Paint    │
     │  ────────  │      │  ────────  │      │  ────────  │
     │  Flexbox   │      │  Hit test  │      │  Quads     │
     │  Measure   │      │  Focus     │      │  Text      │
     │  Position  │      │  Events    │      │  Images    │
     └────────────┘      └────────────┘      └────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │      Draw List         │
                    │  ────────────────────  │
                    │  Quad instances        │
                    │  Glyph instances       │
                    │  Clip rects            │
                    └────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      GPU Renderer      │
                    │  ────────────────────  │
                    │  Batched instancing    │
                    │  Texture atlases       │
                    │  SDF text              │
                    └────────────────────────┘
                                │
                                ▼
                             BLAZE
```

## Core Types

### App

```zig
pub fn App(comptime State: type, comptime view: fn (*State) Element) type {
    return struct {
        const Self = @This();
        
        state: State,
        
        // Internal
        ctx: *blaze.Context,
        renderer: Renderer,
        window: Window,
        prev_tree: ?ElementTree = null,
        
        pub fn init(allocator: Allocator, initial_state: State, window_config: WindowConfig) !Self {
            var ctx = try blaze.Context.init(allocator, .{});
            var renderer = try Renderer.init(&ctx);
            var window = try Window.init(window_config);
            
            return .{
                .state = initial_state,
                .ctx = ctx,
                .renderer = renderer,
                .window = window,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.renderer.deinit();
            self.window.deinit();
            self.ctx.deinit();
        }
        
        pub fn run(self: *Self) !void {
            while (!self.window.shouldClose()) {
                // Poll events
                const events = self.window.pollEvents();
                
                // Rebuild element tree
                const tree = view(&self.state);
                
                // Compute layout
                const layout = computeLayout(tree, self.window.size());
                
                // Handle input
                for (events) |event| {
                    self.handleInput(event, tree, layout);
                }
                
                // Paint
                var draw_list = DrawList.init(self.frame_arena.allocator());
                paint(tree, layout, &draw_list);
                
                // Render
                self.renderer.render(&draw_list, self.window.swapchain());
                
                // Store for diffing
                self.prev_tree = tree;
            }
        }
    };
}

// Usage:
const app = flux.App(MyState, myView);

pub fn main() !void {
    var my_app = try app.init(allocator, .{}, .{ .title = "My App", .width = 1280, .height = 720 });
    defer my_app.deinit();
    try my_app.run();
}
```

### Elements

```zig
pub const Element = union(enum) {
    empty: void,
    text: Text,
    container: Container,
    button: Button,
    input: Input,
    scroll: Scroll,
    custom: Custom,
    
    pub const Text = struct {
        content: []const u8,
        style: TextStyle = .{},
    };
    
    pub const Container = struct {
        style: Style = .{},
        children: []const Element,
    };
    
    pub const Button = struct {
        label: []const u8,
        style: Style = styles.button,
        on_click: ?Callback = null,
        disabled: bool = false,
    };
    
    pub const Input = struct {
        value: []const u8,
        placeholder: []const u8 = "",
        style: Style = styles.input,
        on_change: ?fn ([]const u8) void = null,
    };
    
    pub const Scroll = struct {
        style: Style = .{},
        child: *const Element,
        scroll_x: bool = false,
        scroll_y: bool = true,
    };
    
    pub const Custom = struct {
        data: *anyopaque,
        layout_fn: *const fn (*anyopaque, Constraints) Size,
        paint_fn: *const fn (*anyopaque, Rect, *DrawList) void,
    };
};

// Builder functions (return Element, enable nice syntax)
pub fn text(content: []const u8, style: TextStyle) Element {
    return .{ .text = .{ .content = content, .style = style } };
}

pub fn column(style: Style, children: anytype) Element {
    var merged_style = style;
    merged_style.flex_direction = .column;
    return container(merged_style, children);
}

pub fn row(style: Style, children: anytype) Element {
    var merged_style = style;
    merged_style.flex_direction = .row;
    return container(merged_style, children);
}

pub fn container(style: Style, children: anytype) Element {
    // Comptime: convert tuple to slice
    const Children = @TypeOf(children);
    const fields = @typeInfo(Children).Struct.fields;
    
    var child_array: [fields.len]Element = undefined;
    inline for (fields, 0..) |field, i| {
        child_array[i] = @field(children, field.name);
    }
    
    return .{ .container = .{ .style = style, .children = &child_array } };
}

pub fn button(label: []const u8, props: Button) Element {
    var btn = props;
    btn.label = label;
    return .{ .button = btn };
}
```

### Layout

FLUX uses a flexbox-inspired layout algorithm:

```zig
pub const Layout = struct {
    // Computed values
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    
    // For containers: child layouts
    children: []Layout,
};

pub const Constraints = struct {
    min_width: f32 = 0,
    max_width: f32 = std.math.inf(f32),
    min_height: f32 = 0,
    max_height: f32 = std.math.inf(f32),
};

pub fn computeLayout(element: Element, available_size: Size, allocator: Allocator) Layout {
    return layoutElement(element, .{
        .min_width = 0,
        .max_width = available_size.width,
        .min_height = 0,
        .max_height = available_size.height,
    }, allocator);
}

fn layoutElement(element: Element, constraints: Constraints, allocator: Allocator) Layout {
    return switch (element) {
        .text => |t| layoutText(t, constraints),
        .container => |c| layoutContainer(c, constraints, allocator),
        .button => |b| layoutButton(b, constraints),
        .input => |i| layoutInput(i, constraints),
        // ...
    };
}

fn layoutContainer(container: Container, constraints: Constraints, allocator: Allocator) Layout {
    const style = container.style;
    
    // Compute available space for children
    const padding = style.padding;
    const content_constraints = Constraints{
        .min_width = constraints.min_width - padding.horizontal(),
        .max_width = constraints.max_width - padding.horizontal(),
        .min_height = constraints.min_height - padding.vertical(),
        .max_height = constraints.max_height - padding.vertical(),
    };
    
    // Layout children based on flex direction
    var child_layouts = allocator.alloc(Layout, container.children.len) catch unreachable;
    
    if (style.flex_direction == .column) {
        layoutFlexColumn(container.children, content_constraints, style, child_layouts);
    } else {
        layoutFlexRow(container.children, content_constraints, style, child_layouts);
    }
    
    // Compute total size
    var total_width: f32 = 0;
    var total_height: f32 = 0;
    
    for (child_layouts) |child| {
        if (style.flex_direction == .column) {
            total_width = @max(total_width, child.width);
            total_height += child.height;
        } else {
            total_width += child.width;
            total_height = @max(total_height, child.height);
        }
    }
    
    // Add gaps
    if (container.children.len > 1) {
        const gap_count: f32 = @floatFromInt(container.children.len - 1);
        if (style.flex_direction == .column) {
            total_height += style.gap * gap_count;
        } else {
            total_width += style.gap * gap_count;
        }
    }
    
    return .{
        .x = 0,
        .y = 0,
        .width = total_width + padding.horizontal(),
        .height = total_height + padding.vertical(),
        .children = child_layouts,
    };
}
```

### Drawing

```zig
pub const DrawList = struct {
    quads: std.ArrayList(QuadInstance),
    glyphs: std.ArrayList(GlyphInstance),
    clips: std.ArrayList(ClipRect),
    
    current_clip: ?u32 = null,
    
    pub fn init(allocator: Allocator) DrawList {
        return .{
            .quads = std.ArrayList(QuadInstance).init(allocator),
            .glyphs = std.ArrayList(GlyphInstance).init(allocator),
            .clips = std.ArrayList(ClipRect).init(allocator),
        };
    }
    
    pub fn pushClip(self: *DrawList, rect: Rect) void {
        const clip_idx = @as(u32, @intCast(self.clips.items.len));
        self.clips.append(.{ .rect = rect, .parent = self.current_clip }) catch unreachable;
        self.current_clip = clip_idx;
    }
    
    pub fn popClip(self: *DrawList) void {
        if (self.current_clip) |idx| {
            self.current_clip = self.clips.items[idx].parent;
        }
    }
    
    pub fn addQuad(self: *DrawList, rect: Rect, style: QuadStyle) void {
        self.quads.append(.{
            .position = .{ rect.x, rect.y },
            .size = .{ rect.width, rect.height },
            .color = style.color,
            .border_radius = style.border_radius,
            .border_width = style.border_width,
            .border_color = style.border_color,
            .clip_idx = self.current_clip orelse 0xFFFF,
        }) catch unreachable;
    }
    
    pub fn addText(self: *DrawList, text: []const u8, pos: [2]f32, style: TextStyle, font: *Font) void {
        var x = pos[0];
        const y = pos[1];
        
        for (text) |char| {
            const glyph = font.getGlyph(char) orelse continue;
            
            self.glyphs.append(.{
                .position = .{ x + glyph.bearing_x, y - glyph.bearing_y },
                .size = .{ glyph.width, glyph.height },
                .uv_min = glyph.uv_min,
                .uv_max = glyph.uv_max,
                .color = style.color,
                .clip_idx = self.current_clip orelse 0xFFFF,
            }) catch unreachable;
            
            x += glyph.advance;
        }
    }
};

pub const QuadInstance = extern struct {
    position: [2]f32,
    size: [2]f32,
    color: [4]u8,
    border_radius: f32,
    border_width: f32,
    border_color: [4]u8,
    clip_idx: u16,
    _pad: u16 = 0,
};

pub const GlyphInstance = extern struct {
    position: [2]f32,
    size: [2]f32,
    uv_min: [2]f32,
    uv_max: [2]f32,
    color: [4]u8,
    clip_idx: u16,
    _pad: u16 = 0,
};
```

### GPU Renderer

```zig
pub const Renderer = struct {
    ctx: *blaze.Context,
    
    // Pipelines
    quad_pipeline: blaze.Pipeline,
    glyph_pipeline: blaze.Pipeline,
    
    // Buffers (resized as needed)
    quad_buffer: blaze.Buffer,
    glyph_buffer: blaze.Buffer,
    clip_buffer: blaze.Buffer,
    
    // Font atlas
    font_atlas: blaze.Texture,
    font_sampler: blaze.Sampler,
    font: Font,
    
    // Uniforms
    uniform_buffer: blaze.Buffer,
    
    pub fn init(ctx: *blaze.Context) !Renderer { ... }
    
    pub fn render(self: *Renderer, draw_list: *DrawList, target: blaze.TextureView) void {
        // Upload instance data
        self.uploadQuads(draw_list.quads.items);
        self.uploadGlyphs(draw_list.glyphs.items);
        self.uploadClips(draw_list.clips.items);
        
        var encoder = blaze.Encoder.init(self.ctx.allocator);
        defer encoder.deinit();
        
        encoder.beginRenderPass(.{
            .color_attachments = &.{.{
                .view = target,
                .load_op = .clear,
                .store_op = .store,
                .clear_value = .{ .color = .{ 0.95, 0.95, 0.95, 1.0 } },
            }},
        });
        
        // Draw quads (backgrounds, borders)
        if (draw_list.quads.items.len > 0) {
            encoder.setPipeline(self.quad_pipeline);
            encoder.setBindGroup(0, self.quad_bind_group);
            encoder.draw(4, @intCast(draw_list.quads.items.len));  // Quad = 4 vertices
        }
        
        // Draw glyphs (text)
        if (draw_list.glyphs.items.len > 0) {
            encoder.setPipeline(self.glyph_pipeline);
            encoder.setBindGroup(0, self.glyph_bind_group);
            encoder.draw(4, @intCast(draw_list.glyphs.items.len));
        }
        
        encoder.endRenderPass();
        
        self.ctx.submit(encoder.finish(), null) catch {};
    }
};
```

### Text Rendering

FLUX uses SDF (Signed Distance Field) text rendering for crisp text at any scale:

```zig
pub const Font = struct {
    atlas: blaze.Texture,
    glyphs: std.AutoHashMap(u32, Glyph),  // codepoint -> glyph
    
    line_height: f32,
    ascender: f32,
    descender: f32,
    
    pub const Glyph = struct {
        // Position in atlas (UV coordinates)
        uv_min: [2]f32,
        uv_max: [2]f32,
        
        // Metrics (in pixels at base size)
        width: f32,
        height: f32,
        bearing_x: f32,
        bearing_y: f32,
        advance: f32,
    };
    
    pub fn load(ctx: *blaze.Context, path: []const u8, size: f32) !Font {
        // Load TTF via stb_truetype or freetype
        // Generate SDF atlas
        // Extract metrics
        ...
    }
    
    pub fn getGlyph(self: *Font, codepoint: u32) ?Glyph {
        return self.glyphs.get(codepoint);
    }
    
    pub fn measureText(self: *Font, text: []const u8) Size {
        var width: f32 = 0;
        for (text) |char| {
            if (self.getGlyph(char)) |glyph| {
                width += glyph.advance;
            }
        }
        return .{ .width = width, .height = self.line_height };
    }
};
```

## Example: Engineering Calculator

```zig
const std = @import("std");
const flux = @import("flux");

const State = struct {
    // Inputs
    span_ft: f32 = 20.0,
    load_plf: f32 = 100.0,
    fy_ksi: f32 = 50.0,
    
    // Results
    moment_ftlb: ?f32 = null,
    required_sx: ?f32 = null,
    selected_beam: ?[]const u8 = null,
    
    fn calculate(self: *State) void {
        self.moment_ftlb = self.load_plf * self.span_ft * self.span_ft / 8.0;
        self.required_sx = self.moment_ftlb.? * 12.0 / (0.66 * self.fy_ksi * 1000.0);
        self.selected_beam = selectBeam(self.required_sx.?);
    }
    
    fn fmt(comptime format: []const u8, args: anytype) []const u8 {
        // Use frame allocator for temporary strings
        return std.fmt.allocPrint(flux.frameAllocator(), format, args) catch "";
    }
};

fn view(state: *State) flux.Element {
    return flux.column(.{ .padding = .all(24), .gap = 16 }, .{
        // Header
        flux.text("Simple Beam Calculator", .{ .size = 24, .weight = .bold }),
        
        // Input section
        flux.container(.{ .style = flux.styles.card, .gap = 12 }, .{
            flux.text("Inputs", .{ .size = 18, .weight = .semibold }),
            
            inputRow("Span", "ft", &state.span_ft),
            inputRow("Uniform Load", "plf", &state.load_plf),
            inputRow("Fy", "ksi", &state.fy_ksi),
            
            flux.button("Calculate", .{
                .style = flux.styles.primary_button,
                .on_click = flux.callback(state, State.calculate),
            }),
        }),
        
        // Results section
        if (state.moment_ftlb != null) resultsSection(state) else flux.empty(),
    });
}

fn inputRow(label: []const u8, unit: []const u8, value: *f32) flux.Element {
    return flux.row(.{ .align_items = .center, .gap = 8 }, .{
        flux.text(label, .{ .width = 120 }),
        flux.inputFloat(value, .{ .width = 100 }),
        flux.text(unit, .{ .color = .gray }),
    });
}

fn resultsSection(state: *State) flux.Element {
    return flux.container(.{ .style = flux.styles.card, .gap = 8 }, .{
        flux.text("Results", .{ .size = 18, .weight = .semibold }),
        
        resultRow("Max Moment", State.fmt("{d:.1} ft-lb", .{state.moment_ftlb.?})),
        resultRow("Required Sx", State.fmt("{d:.2} in³", .{state.required_sx.?})),
        resultRow("Selected Beam", state.selected_beam orelse "None"),
    });
}

fn resultRow(label: []const u8, value: []const u8) flux.Element {
    return flux.row(.{ .justify_content = .space_between }, .{
        flux.text(label, .{}),
        flux.text(value, .{ .weight = .semibold }),
    });
}

pub fn main() !void {
    var app = try flux.App(State, view).init(std.heap.page_allocator, .{}, .{
        .title = "Beam Calculator",
        .width = 400,
        .height = 500,
    });
    defer app.deinit();
    
    try app.run();
}
```

---

# Integration: FLUX + FORGE

For applications that need both UI and 3D (like structural engineering software), FLUX and FORGE integrate naturally:

```zig
const State = struct {
    // UI state
    selected_member: ?MemberHandle = null,
    show_loads: bool = true,
    
    // Scene
    scene: *forge.Scene,
    
    // Model data
    model: StructuralModel,
};

fn view(state: *State) flux.Element {
    return flux.row(.{ .flex = 1 }, .{
        // Left panel: controls
        flux.container(.{ .width = 300, .padding = .all(16) }, .{
            controlsPanel(state),
        }),
        
        // Main area: 3D viewport
        flux.custom(.{
            .flex = 1,
            .render = struct {
                fn render(s: *State, rect: flux.Rect, draw_list: *flux.DrawList) void {
                    // FORGE renders to a texture, FLUX composites it
                    s.scene.render(draw_list.get3DEncoder(), .{
                        .viewport = rect,
                    });
                }
            }.render,
            .context = state,
        }),
        
        // Right panel: properties
        if (state.selected_member) |member|
            propertiesPanel(state.model.getMember(member))
        else
            flux.empty(),
    });
}
```

---

# Appendix A: Dependency Graph

```
┌─────────────────────────────────────────────────┐
│                 Application                      │
│         (Your structural software, game)         │
└────────────────────┬───────────────────────────┬┘
                     │                           │
         ┌───────────┴───────────┐   ┌──────────┴───────────┐
         ▼                       ▼   ▼                      │
    ┌─────────┐             ┌─────────┐                     │
    │  FLUX   │             │  FORGE  │                     │
    │  (UI)   │             │ (Scene) │                     │
    └────┬────┘             └────┬────┘                     │
         │                       │                          │
         └───────────┬───────────┘                          │
                     ▼                                      │
              ┌───────────┐                                 │
              │   BLAZE   │◄────────────────────────────────┘
              │   (GPU)   │    (Direct BLAZE usage for
              └─────┬─────┘     custom rendering also OK)
                    │
                    ▼
            ┌───────────────┐
            │    Vulkan     │
            └───────────────┘
```

# Appendix B: File Structure

```
zig-graphics/
├── blaze/
│   ├── src/
│   │   ├── blaze.zig           # Public API
│   │   ├── context.zig         # Context implementation
│   │   ├── encoder.zig         # Command encoding
│   │   ├── resources.zig       # Buffer, Texture, etc.
│   │   ├── pipeline.zig        # Pipeline creation
│   │   ├── shader.zig          # Shader compilation/reflection
│   │   ├── sync.zig            # Synchronization primitives
│   │   ├── allocator.zig       # Memory allocator
│   │   └── vulkan/
│   │       ├── instance.zig
│   │       ├── device.zig
│   │       ├── swapchain.zig
│   │       └── bindings.zig    # Vulkan function pointers
│   ├── shaders/
│   │   └── *.wgsl
│   └── build.zig
│
├── forge/
│   ├── src/
│   │   ├── forge.zig           # Public API
│   │   ├── scene.zig           # Scene management
│   │   ├── mesh.zig            # Mesh and meshlet handling
│   │   ├── material.zig        # Material system
│   │   ├── camera.zig          # Camera utilities
│   │   ├── culling.zig         # GPU culling
│   │   ├── lighting.zig        # Light management
│   │   └── loaders/
│   │       ├── gltf.zig
│   │       └── obj.zig
│   ├── shaders/
│   │   ├── cull.wgsl
│   │   ├── mesh.wgsl
│   │   └── material_*.wgsl
│   └── build.zig
│
├── flux/
│   ├── src/
│   │   ├── flux.zig            # Public API
│   │   ├── app.zig             # Application framework
│   │   ├── element.zig         # Element types
│   │   ├── layout.zig          # Layout algorithm
│   │   ├── paint.zig           # Paint/draw list
│   │   ├── input.zig           # Input handling
│   │   ├── renderer.zig        # GPU rendering
│   │   ├── text.zig            # Text shaping/rendering
│   │   ├── style.zig           # Style definitions
│   │   └── widgets/
│   │       ├── button.zig
│   │       ├── input.zig
│   │       ├── scroll.zig
│   │       └── ...
│   ├── shaders/
│   │   ├── quad.wgsl
│   │   └── glyph.wgsl
│   └── build.zig
│
└── examples/
    ├── triangle/               # BLAZE basics
    ├── mesh_viewer/            # FORGE basics
    ├── calculator/             # FLUX basics
    └── structural_viz/         # Full integration
```

# Appendix C: Zig Differentiation Summary

| Aspect | Rust (BLADE/GPUI) | Zig (BLAZE/FORGE/FLUX) |
|--------|-------------------|------------------------|
| Shader binding | Runtime reflection | Comptime reflection, type-safe |
| Vertex layout | Derive macro | Comptime from struct fields |
| Element trees | Trait objects, heap alloc | Comptime types, stack/arena |
| Event handlers | Closures with captures | Function pointers + context |
| Style system | Runtime parsing | Comptime structs |
| Memory management | RAII, hidden Drop | Explicit allocators, arenas |
| Error handling | Result<T, E>, ? operator | Error sets, errdefer |
| Generic containers | Generics + trait bounds | Comptime type parameters |
| Build integration | Cargo, proc macros | build.zig, comptime |
| GPU struct layout | repr(C) + bytemuck | extern struct, comptime assert |
| Debug | dbg!, println! | std.debug.print, @breakpoint |

**Key Zig advantages leveraged:**
1. **Comptime** - Shader reflection, element trees, style resolution at compile time
2. **Explicit allocators** - Frame arenas for zero-cost temporary allocations
3. **No hidden control flow** - No Drop, no implicit copies, predictable performance
4. **C interop** - Direct Vulkan calls, easy freetype/harfbuzz integration
5. **Build system** - Shader compilation integrated into build.zig
