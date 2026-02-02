# Zig Graphics Stack Conformance Testing

## North Star Vision

This document defines the conformance testing strategy for BLAZE, FORGE, and FLUX. The goal is not just "does it work?" but **provable correctness** across all platforms, configurations, and edge cases.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONFORMANCE PYRAMID                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                           ┌─────────┐                                    │
│                          ╱ Integration ╲     ← Full stack tests          │
│                         ╱   (FLUX+FORGE  ╲      (~50 tests)              │
│                        ╱     +BLAZE)      ╲                              │
│                       ├───────────────────┤                              │
│                      ╱    FLUX (UI)        ╲    ← Layout, input,         │
│                     ╱      (~200 tests)     ╲     rendering (~200)       │
│                    ├─────────────────────────┤                           │
│                   ╱      FORGE (3D)          ╲  ← Scene, culling,        │
│                  ╱        (~300 tests)        ╲   materials (~300)       │
│                 ├─────────────────────────────┤                          │
│                ╱        BLAZE (GPU)            ╲ ← API, compute,         │
│               ╱          (~500 tests)           ╲  render (~500)         │
│              └───────────────────────────────────┘                       │
│                                                                          │
│                    Total Target: ~1050 tests                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Philosophy

### 1. Tests Are Specification

Every public API function should have corresponding tests that serve as:
- **Documentation** - Show how to use the API correctly
- **Contract** - Define expected behavior precisely
- **Guard Rails** - Catch regressions immediately

### 2. Reproducibility Is Non-Negotiable

Tests must produce identical results:
- Across runs on the same machine
- Across different GPU vendors (NVIDIA, AMD, Intel)
- Across driver versions (within reason)
- In CI environments

### 3. Failure Modes Matter

Test not just success paths but:
- Invalid inputs (should error, not crash)
- Resource exhaustion (OOM, descriptor limits)
- Edge cases (zero-size buffers, empty draws)
- Concurrent access patterns

### 4. Performance Is Correctness

A GPU library that's correct but slow is incorrect for its purpose:
- Benchmark critical paths
- Track performance regressions
- Set performance budgets

---

# BLAZE Conformance Test Suite (BLAZE-CTS)

## Test Categories

### Category 1: Context & Device (`blaze.context.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| CTX-001 | `create_default` | Create context with default config | P0 |
| CTX-002 | `create_with_validation` | Create with validation layers enabled | P0 |
| CTX-003 | `create_compute_only` | Create headless compute context | P0 |
| CTX-004 | `create_with_features` | Request specific Vulkan features | P1 |
| CTX-005 | `create_preferred_device` | Select GPU by name substring | P1 |
| CTX-006 | `create_missing_features` | Request unavailable features → error | P1 |
| CTX-007 | `destroy_with_pending_work` | Destroy while GPU busy → clean shutdown | P1 |
| CTX-008 | `multiple_contexts` | Create multiple contexts simultaneously | P2 |
| CTX-009 | `context_device_info` | Query device name, limits, features | P1 |
| CTX-010 | `context_memory_budget` | Query available VRAM | P2 |

**Verification Method:** API returns, no crashes, validation layer silence

---

### Category 2: Buffer Operations (`blaze.buffer.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| BUF-001 | `create_vertex_buffer` | Create with vertex usage | P0 |
| BUF-002 | `create_index_buffer` | Create with index usage | P0 |
| BUF-003 | `create_uniform_buffer` | Create with uniform usage | P0 |
| BUF-004 | `create_storage_buffer` | Create with storage usage | P0 |
| BUF-005 | `create_indirect_buffer` | Create with indirect usage | P1 |
| BUF-006 | `create_combined_usage` | Multiple usage flags | P1 |
| BUF-007 | `create_zero_size` | Zero size → error | P1 |
| BUF-008 | `create_huge_buffer` | Near-limit size allocation | P2 |
| BUF-009 | `map_host_visible` | Map and write directly | P0 |
| BUF-010 | `map_write_read_back` | Write via map, verify contents | P0 |
| BUF-011 | `upload_device_local` | Staging upload to device memory | P0 |
| BUF-012 | `download_device_local` | Read back from device memory | P0 |
| BUF-013 | `buffer_slice` | Create slice, use in binding | P1 |
| BUF-014 | `destroy_while_mapped` | Destroy mapped buffer → clean | P1 |
| BUF-015 | `destroy_while_in_use` | Destroy during GPU use → defer | P2 |

**Verification Method:**
- Map and verify byte contents
- Round-trip: upload → download → compare
- Validation layer silence

---

### Category 3: Texture Operations (`blaze.texture.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| TEX-001 | `create_rgba8` | Create RGBA8 texture | P0 |
| TEX-002 | `create_depth32` | Create depth texture | P0 |
| TEX-003 | `create_formats` | All supported formats | P1 |
| TEX-004 | `create_with_mipmaps` | Multi-level mipmap chain | P1 |
| TEX-005 | `create_array` | Texture array | P1 |
| TEX-006 | `create_cube` | Cubemap texture | P2 |
| TEX-007 | `create_3d` | 3D/volume texture | P2 |
| TEX-008 | `create_multisampled` | MSAA texture | P1 |
| TEX-009 | `upload_texture` | Upload image data | P0 |
| TEX-010 | `upload_subregion` | Upload to texture subregion | P1 |
| TEX-011 | `download_texture` | Read back texture data | P1 |
| TEX-012 | `generate_mipmaps` | Auto-generate mip chain | P1 |
| TEX-013 | `texture_view_default` | Create view with defaults | P0 |
| TEX-014 | `texture_view_mip_range` | View specific mip levels | P1 |
| TEX-015 | `texture_view_array_slice` | View specific array layers | P1 |

**Verification Method:**
- Golden image comparison (upload → render → compare)
- Pixel readback verification
- Format conversion correctness

---

### Category 4: Sampler Operations (`blaze.sampler.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| SMP-001 | `create_linear` | Linear filtering | P0 |
| SMP-002 | `create_nearest` | Nearest filtering | P0 |
| SMP-003 | `create_mipmap_linear` | Trilinear mipmapping | P1 |
| SMP-004 | `address_repeat` | Repeat addressing | P0 |
| SMP-005 | `address_clamp` | Clamp to edge | P1 |
| SMP-006 | `address_mirror` | Mirror repeat | P1 |
| SMP-007 | `address_border` | Border color | P2 |
| SMP-008 | `anisotropic` | Anisotropic filtering | P1 |
| SMP-009 | `compare_sampler` | Depth comparison sampler | P1 |

**Verification Method:** Render textured quad, compare to golden image

---

### Category 5: Pipeline Creation (`blaze.pipeline.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| PIP-001 | `create_graphics_minimal` | Minimal vertex+fragment | P0 |
| PIP-002 | `create_graphics_full` | All stages configured | P1 |
| PIP-003 | `create_compute` | Compute pipeline | P0 |
| PIP-004 | `vertex_layout_basic` | Position-only vertex | P0 |
| PIP-005 | `vertex_layout_interleaved` | Multiple attributes | P0 |
| PIP-006 | `vertex_layout_instanced` | Per-instance attributes | P1 |
| PIP-007 | `blend_opaque` | No blending | P0 |
| PIP-008 | `blend_alpha` | Standard alpha blend | P0 |
| PIP-009 | `blend_additive` | Additive blend | P1 |
| PIP-010 | `blend_premultiplied` | Premultiplied alpha | P1 |
| PIP-011 | `depth_test_less` | Standard depth test | P0 |
| PIP-012 | `depth_test_modes` | All compare functions | P1 |
| PIP-013 | `depth_write_disabled` | Read-only depth | P1 |
| PIP-014 | `stencil_basic` | Simple stencil test | P2 |
| PIP-015 | `cull_back` | Backface culling | P0 |
| PIP-016 | `cull_front` | Frontface culling | P1 |
| PIP-017 | `cull_none` | No culling | P1 |
| PIP-018 | `topology_triangles` | Triangle list | P0 |
| PIP-019 | `topology_lines` | Line list | P1 |
| PIP-020 | `topology_points` | Point list | P2 |
| PIP-021 | `shader_compile_error` | Invalid shader → error | P0 |
| PIP-022 | `pipeline_cache` | Pipeline caching | P2 |

**Verification Method:**
- Successful creation
- Render and verify output
- Shader compilation error messages

---

### Category 6: Shader Reflection (`blaze.shader.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| SHD-001 | `reflect_uniforms` | Extract uniform bindings | P0 |
| SHD-002 | `reflect_storage` | Extract storage bindings | P0 |
| SHD-003 | `reflect_textures` | Extract texture bindings | P0 |
| SHD-004 | `reflect_samplers` | Extract sampler bindings | P0 |
| SHD-005 | `reflect_push_constants` | Extract push constant layout | P1 |
| SHD-006 | `reflect_vertex_inputs` | Extract vertex attributes | P0 |
| SHD-007 | `bind_group_type_safety` | Wrong type → compile error | P0 |
| SHD-008 | `bind_group_missing` | Missing binding → compile error | P0 |
| SHD-009 | `wgsl_to_spirv` | Compile WGSL shader | P0 |
| SHD-010 | `spirv_validation` | Generated SPIR-V is valid | P0 |

**Verification Method:**
- Comptime error on type mismatch
- SPIR-V validator (spirv-val)
- Successful pipeline creation

---

### Category 7: Command Encoding (`blaze.encoder.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| CMD-001 | `encode_empty` | Empty command list | P1 |
| CMD-002 | `encode_single_draw` | One draw call | P0 |
| CMD-003 | `encode_multi_draw` | Multiple draws | P0 |
| CMD-004 | `encode_indexed_draw` | Draw indexed | P0 |
| CMD-005 | `encode_indirect_draw` | Draw indirect | P1 |
| CMD-006 | `encode_dispatch` | Compute dispatch | P0 |
| CMD-007 | `encode_dispatch_indirect` | Compute indirect | P1 |
| CMD-008 | `encode_copy_buffer` | Buffer to buffer copy | P0 |
| CMD-009 | `encode_copy_texture` | Texture copy | P1 |
| CMD-010 | `encode_copy_buf_to_tex` | Buffer to texture | P0 |
| CMD-011 | `encode_scissor` | Set scissor rect | P1 |
| CMD-012 | `encode_viewport` | Set viewport | P1 |
| CMD-013 | `encode_reset_reuse` | Reset and reuse encoder | P0 |
| CMD-014 | `render_pass_single` | Single render pass | P0 |
| CMD-015 | `render_pass_multi_target` | MRT rendering | P1 |
| CMD-016 | `render_pass_clear_load` | Clear vs load attachment | P0 |
| CMD-017 | `render_pass_resolve_msaa` | MSAA resolve | P1 |

**Verification Method:**
- Validation layer silence
- Correct visual output
- Buffer content verification

---

### Category 8: Synchronization (`blaze.sync.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| SYN-001 | `timeline_create` | Create timeline semaphore | P0 |
| SYN-002 | `timeline_signal_wait` | Signal and wait | P0 |
| SYN-003 | `timeline_cpu_wait` | CPU waits on GPU | P0 |
| SYN-004 | `timeline_gpu_wait` | GPU waits on GPU | P1 |
| SYN-005 | `timeline_multi_wait` | Wait on multiple values | P1 |
| SYN-006 | `fence_create` | Create fence | P0 |
| SYN-007 | `fence_wait` | Wait on fence | P0 |
| SYN-008 | `fence_reset` | Reset and reuse fence | P1 |
| SYN-009 | `barrier_compute_to_draw` | Compute → graphics barrier | P0 |
| SYN-010 | `barrier_draw_to_compute` | Graphics → compute barrier | P1 |
| SYN-011 | `barrier_transfer` | Transfer barriers | P0 |
| SYN-012 | `wait_idle` | Full device idle | P0 |
| SYN-013 | `submit_ordering` | Submits execute in order | P0 |
| SYN-014 | `async_compute_overlap` | Async compute with graphics | P2 |

**Verification Method:**
- Race condition detection (multiple runs)
- Data hazard verification
- Validation layer checks

---

### Category 9: Compute Correctness (`blaze.compute.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| CMP-001 | `sum_reduction` | Parallel sum | P0 |
| CMP-002 | `prefix_sum` | Parallel prefix sum | P1 |
| CMP-003 | `matrix_multiply` | Matrix multiplication | P1 |
| CMP-004 | `vector_add` | Element-wise add | P0 |
| CMP-005 | `sort_bitonic` | Bitonic sort | P2 |
| CMP-006 | `histogram` | Histogram computation | P1 |
| CMP-007 | `workgroup_shared` | Shared memory usage | P0 |
| CMP-008 | `workgroup_barrier` | Barrier correctness | P0 |
| CMP-009 | `atomic_add` | Atomic addition | P0 |
| CMP-010 | `atomic_min_max` | Atomic min/max | P1 |
| CMP-011 | `indirect_dispatch` | Indirect dispatch | P1 |
| CMP-012 | `large_dispatch` | Max workgroup count | P2 |
| CMP-013 | `subgroup_add` | Subgroup reduction | P2 |
| CMP-014 | `subgroup_ballot` | Subgroup ballot | P2 |

**Verification Method:**
- Numerical comparison with CPU reference
- Tolerance-based floating point comparison
- Statistical verification for probabilistic algorithms

---

### Category 10: Render Correctness (`blaze.render.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| RND-001 | `clear_color` | Clear to solid color | P0 |
| RND-002 | `clear_depth` | Clear depth buffer | P0 |
| RND-003 | `triangle_solid` | Solid color triangle | P0 |
| RND-004 | `triangle_vertex_color` | Vertex-colored triangle | P0 |
| RND-005 | `triangle_textured` | Textured triangle | P0 |
| RND-006 | `quad_textured` | Textured quad | P0 |
| RND-007 | `cube_depth` | Cube with depth test | P0 |
| RND-008 | `cube_textured` | Textured cube | P0 |
| RND-009 | `instancing` | Instanced rendering | P1 |
| RND-010 | `multi_draw` | Multiple objects | P1 |
| RND-011 | `alpha_blend` | Alpha blended sprites | P0 |
| RND-012 | `scissor_clip` | Scissor clipping | P1 |
| RND-013 | `viewport_transform` | Non-default viewport | P1 |
| RND-014 | `wireframe` | Wireframe mode | P2 |
| RND-015 | `point_rendering` | Point sprites | P2 |
| RND-016 | `line_rendering` | Line rendering | P1 |
| RND-017 | `msaa_4x` | 4x MSAA | P1 |
| RND-018 | `msaa_8x` | 8x MSAA | P2 |
| RND-019 | `mrt_output` | Multiple render targets | P1 |
| RND-020 | `offscreen_render` | Render to texture | P0 |

**Verification Method:** Golden image comparison with configurable tolerance

---

## Golden Image Testing Infrastructure

### Directory Structure

```
blaze/
└── test/
    ├── golden/
    │   ├── render/
    │   │   ├── triangle_solid.png
    │   │   ├── triangle_solid_nvidia.png    # Vendor-specific if needed
    │   │   ├── triangle_solid_amd.png
    │   │   ├── cube_textured.png
    │   │   └── ...
    │   └── compute/
    │       └── (numerical reference data)
    ├── src/
    │   ├── runner.zig
    │   ├── framework.zig
    │   ├── golden.zig
    │   ├── context/
    │   ├── buffer/
    │   ├── texture/
    │   ├── pipeline/
    │   ├── compute/
    │   └── render/
    └── build.zig
```

### Image Comparison Algorithm

```zig
pub const ImageCompareOptions = struct {
    /// Per-pixel color tolerance (0-255 per channel)
    color_tolerance: u8 = 2,

    /// Maximum percentage of pixels allowed to differ
    max_diff_percent: f32 = 0.1,

    /// Enable perceptual comparison (LAB color space)
    perceptual: bool = true,

    /// Ignore anti-aliasing edge pixels
    ignore_aa: bool = true,
};

pub fn compareImages(
    actual: Image,
    expected: Image,
    options: ImageCompareOptions,
) ImageCompareResult {
    if (actual.width != expected.width or actual.height != expected.height) {
        return .{ .match = false, .reason = .dimension_mismatch };
    }

    var diff_count: u32 = 0;
    var max_diff: u32 = 0;

    for (0..actual.height) |y| {
        for (0..actual.width) |x| {
            const actual_pixel = actual.getPixel(x, y);
            const expected_pixel = expected.getPixel(x, y);

            const diff = if (options.perceptual)
                perceptualDiff(actual_pixel, expected_pixel)
            else
                colorDiff(actual_pixel, expected_pixel);

            if (diff > options.color_tolerance) {
                if (options.ignore_aa and isEdgePixel(expected, x, y)) {
                    continue;
                }
                diff_count += 1;
                max_diff = @max(max_diff, diff);
            }
        }
    }

    const diff_percent = @as(f32, @floatFromInt(diff_count)) /
                         @as(f32, @floatFromInt(actual.width * actual.height)) * 100.0;

    return .{
        .match = diff_percent <= options.max_diff_percent,
        .diff_percent = diff_percent,
        .diff_count = diff_count,
        .max_diff = max_diff,
    };
}
```

### Golden Image Generation Mode

```bash
# Normal test run
zig build test

# Generate/update golden images
zig build test -- --generate-golden

# Compare and show diff images
zig build test -- --save-diffs
```

---

# FORGE Conformance Test Suite (FORGE-CTS)

## Test Categories

### Category 1: Scene Management (`forge.scene.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| SCN-001 | `create_scene` | Create with default config | P0 |
| SCN-002 | `create_configured` | Custom limits and features | P1 |
| SCN-003 | `add_object` | Add single object | P0 |
| SCN-004 | `add_many_objects` | Add 10,000 objects | P0 |
| SCN-005 | `remove_object` | Remove object | P0 |
| SCN-006 | `update_transform` | Update object transform | P0 |
| SCN-007 | `object_visibility` | Toggle visibility | P1 |
| SCN-008 | `scene_clear` | Clear all objects | P1 |
| SCN-009 | `scene_stats` | Query object counts | P1 |

---

### Category 2: Mesh & Meshlets (`forge.mesh.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| MSH-001 | `load_gltf` | Load glTF mesh | P0 |
| MSH-002 | `load_obj` | Load OBJ mesh | P1 |
| MSH-003 | `upload_mesh` | Upload from memory | P0 |
| MSH-004 | `meshletize_basic` | Create meshlets | P0 |
| MSH-005 | `meshletize_large` | Meshletize 1M triangles | P1 |
| MSH-006 | `meshlet_bounds` | Bounding sphere accuracy | P0 |
| MSH-007 | `meshlet_cones` | Normal cone accuracy | P1 |
| MSH-008 | `mesh_lod` | LOD chain generation | P2 |

---

### Category 3: Materials (`forge.material.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| MAT-001 | `create_standard` | PBR material | P0 |
| MAT-002 | `create_unlit` | Unlit material | P0 |
| MAT-003 | `create_with_textures` | Textured material | P0 |
| MAT-004 | `material_variants` | All shader variants | P1 |
| MAT-005 | `material_update` | Update material properties | P1 |
| MAT-006 | `material_batching` | Same material = 1 draw | P1 |

---

### Category 4: GPU Culling (`forge.culling.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| CUL-001 | `frustum_visible` | Object in frustum → drawn | P0 |
| CUL-002 | `frustum_outside` | Object outside → culled | P0 |
| CUL-003 | `frustum_partial` | Partial intersection | P0 |
| CUL-004 | `frustum_many` | Cull 100,000 objects | P0 |
| CUL-005 | `backface_culled` | Backfacing meshlet culled | P1 |
| CUL-006 | `occlusion_hidden` | Occluded object culled | P2 |
| CUL-007 | `lod_selection` | Distance-based LOD | P2 |
| CUL-008 | `culling_stats` | Verify cull counts | P1 |

---

### Category 5: Camera (`forge.camera.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| CAM-001 | `perspective_matrix` | Projection correctness | P0 |
| CAM-002 | `view_matrix` | View matrix correctness | P0 |
| CAM-003 | `frustum_planes` | Plane extraction | P0 |
| CAM-004 | `look_at` | Look-at targeting | P0 |
| CAM-005 | `orbit_camera` | Orbit around point | P1 |
| CAM-006 | `fly_camera` | Free-fly movement | P1 |

---

### Category 6: Lighting (`forge.lighting.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| LIT-001 | `directional_light` | Sun light | P0 |
| LIT-002 | `point_light` | Point light falloff | P1 |
| LIT-003 | `spot_light` | Spotlight cone | P2 |
| LIT-004 | `many_lights` | 256 lights | P1 |
| LIT-005 | `shadow_directional` | Directional shadow map | P1 |
| LIT-006 | `shadow_cascades` | CSM shadows | P2 |

---

### Category 7: Render Correctness (`forge.render.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| FRN-001 | `single_object` | Render one object | P0 |
| FRN-002 | `many_objects` | Render 10,000 objects | P0 |
| FRN-003 | `textured_objects` | Textured rendering | P0 |
| FRN-004 | `pbr_materials` | PBR material correctness | P1 |
| FRN-005 | `stylized_shading` | Toon/cel shading | P2 |
| FRN-006 | `transparency` | Alpha-blended objects | P1 |
| FRN-007 | `indirect_draw` | Indirect rendering | P0 |

---

# FLUX Conformance Test Suite (FLUX-CTS)

## Test Categories

### Category 1: Layout (`flux.layout.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| LAY-001 | `single_element` | Single element sizing | P0 |
| LAY-002 | `row_layout` | Horizontal flex | P0 |
| LAY-003 | `column_layout` | Vertical flex | P0 |
| LAY-004 | `nested_layout` | Nested containers | P0 |
| LAY-005 | `flex_grow` | Flex grow distribution | P0 |
| LAY-006 | `flex_shrink` | Flex shrink behavior | P1 |
| LAY-007 | `justify_start` | Justify content start | P0 |
| LAY-008 | `justify_center` | Justify content center | P0 |
| LAY-009 | `justify_end` | Justify content end | P0 |
| LAY-010 | `justify_between` | Space between | P1 |
| LAY-011 | `align_start` | Align items start | P0 |
| LAY-012 | `align_center` | Align items center | P0 |
| LAY-013 | `align_stretch` | Align items stretch | P0 |
| LAY-014 | `padding` | Padding applied correctly | P0 |
| LAY-015 | `margin` | Margin applied correctly | P0 |
| LAY-016 | `gap` | Gap between children | P0 |
| LAY-017 | `min_max_size` | Min/max constraints | P1 |
| LAY-018 | `fixed_size` | Fixed width/height | P0 |
| LAY-019 | `percentage_size` | Percentage sizing | P1 |
| LAY-020 | `text_measurement` | Text size calculation | P0 |

**Verification Method:** Compare computed layout rects against expected values

---

### Category 2: Rendering (`flux.render.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| FLR-001 | `solid_rect` | Solid color rectangle | P0 |
| FLR-002 | `rounded_rect` | Rounded corners | P0 |
| FLR-003 | `border` | Border rendering | P0 |
| FLR-004 | `gradient_linear` | Linear gradient | P2 |
| FLR-005 | `text_basic` | Single line text | P0 |
| FLR-006 | `text_multiline` | Multi-line text | P1 |
| FLR-007 | `text_alignment` | Text alignment | P1 |
| FLR-008 | `text_styles` | Bold, italic, etc. | P1 |
| FLR-009 | `image_render` | Image display | P1 |
| FLR-010 | `image_scaled` | Image scaling modes | P1 |
| FLR-011 | `clip_rect` | Clipping to bounds | P0 |
| FLR-012 | `nested_clip` | Nested clipping | P1 |
| FLR-013 | `z_order` | Z-ordering correctness | P0 |
| FLR-014 | `opacity` | Element opacity | P1 |
| FLR-015 | `batching` | Draw call batching | P1 |

**Verification Method:** Golden image comparison

---

### Category 3: Input Handling (`flux.input.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| INP-001 | `click_button` | Button click detection | P0 |
| INP-002 | `click_nested` | Click nested element | P0 |
| INP-003 | `hover_enter_exit` | Hover state changes | P0 |
| INP-004 | `focus_tab` | Tab navigation | P1 |
| INP-005 | `focus_click` | Click to focus | P0 |
| INP-006 | `keyboard_text` | Text input | P0 |
| INP-007 | `keyboard_shortcuts` | Key combinations | P1 |
| INP-008 | `scroll_wheel` | Mouse wheel scroll | P1 |
| INP-009 | `drag_basic` | Drag detection | P1 |
| INP-010 | `hit_test_clip` | Clipped elements not hit | P0 |

**Verification Method:** Event sequence verification, state assertions

---

### Category 4: Widgets (`flux.widgets.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| WID-001 | `button_states` | Normal/hover/pressed/disabled | P0 |
| WID-002 | `text_input` | Text input field | P0 |
| WID-003 | `text_input_selection` | Text selection | P1 |
| WID-004 | `checkbox` | Checkbox toggle | P0 |
| WID-005 | `radio_group` | Radio button group | P1 |
| WID-006 | `slider` | Value slider | P1 |
| WID-007 | `dropdown` | Dropdown/select | P1 |
| WID-008 | `scroll_view` | Scrollable container | P0 |
| WID-009 | `scroll_indicators` | Scrollbar rendering | P1 |
| WID-010 | `modal` | Modal dialog | P2 |
| WID-011 | `tooltip` | Tooltip display | P2 |

**Verification Method:** Interaction simulation + visual verification

---

### Category 5: Text Rendering (`flux.text.*`)

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| TXT-001 | `font_loading` | Load TTF font | P0 |
| TXT-002 | `glyph_cache` | Glyph caching | P0 |
| TXT-003 | `sdf_quality` | SDF rendering quality | P0 |
| TXT-004 | `text_scaling` | Text at various sizes | P0 |
| TXT-005 | `unicode_basic` | Basic Unicode support | P0 |
| TXT-006 | `unicode_emoji` | Emoji rendering | P2 |
| TXT-007 | `text_shaping` | Complex script shaping | P2 |
| TXT-008 | `kerning` | Kerning applied | P1 |
| TXT-009 | `line_height` | Line height calculation | P0 |
| TXT-010 | `word_wrap` | Word wrapping | P0 |

**Verification Method:** Golden image at multiple DPI scales

---

# Integration Tests

## FLUX + FORGE Integration

| Test ID | Name | Description | Priority |
|---------|------|-------------|----------|
| INT-001 | `ui_over_3d` | UI rendered over 3D scene | P0 |
| INT-002 | `3d_viewport_widget` | 3D viewport as UI element | P0 |
| INT-003 | `ui_updates_scene` | UI controls modify scene | P1 |
| INT-004 | `scene_events_to_ui` | 3D picking updates UI | P1 |
| INT-005 | `resize_both` | Window resize affects both | P0 |

---

# Performance Benchmarks

Not just correctness—track performance regressions:

```
benchmarks/
├── blaze/
│   ├── buffer_upload.zig      # MB/s throughput
│   ├── texture_upload.zig     # MB/s throughput
│   ├── draw_calls.zig         # Draws/second
│   ├── compute_dispatch.zig   # Dispatches/second
│   └── pipeline_create.zig    # Pipelines/second
├── forge/
│   ├── culling_throughput.zig # Objects culled/ms
│   ├── render_many.zig        # FPS at object counts
│   └── scene_update.zig       # Transform updates/ms
└── flux/
    ├── layout_compute.zig     # Elements laid out/ms
    ├── text_shaping.zig       # Glyphs shaped/ms
    └── render_ui.zig          # UI elements/ms
```

### Benchmark Output Format

```
═══════════════════════════════════════════════════════════════
BLAZE Performance Benchmarks
═══════════════════════════════════════════════════════════════
buffer_upload_1mb        │ 4,523 MB/s │ baseline: 4,200 │ ✓ +7.7%
buffer_upload_64mb       │ 12,847 MB/s │ baseline: 12,500 │ ✓ +2.8%
draw_calls_simple        │ 847,293/s │ baseline: 800,000 │ ✓ +5.9%
draw_calls_textured      │ 412,847/s │ baseline: 400,000 │ ✓ +3.2%
compute_dispatch_64      │ 1,293,847/s │ baseline: 1,200,000 │ ✓ +7.8%
───────────────────────────────────────────────────────────────
All benchmarks passed baseline thresholds
═══════════════════════════════════════════════════════════════
```

---

# CI Integration

## GitHub Actions Workflow

```yaml
# .github/workflows/conformance.yml

name: Conformance Tests

on:
  push:
    branches: [master]
  pull_request:

jobs:
  blaze-cts:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        gpu: [nvidia, amd, intel, software]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Zig
        uses: goto-bus-stop/setup-zig@v2
        with:
          version: 0.13.0

      - name: Setup GPU (Software Renderer)
        if: matrix.gpu == 'software'
        run: |
          sudo apt-get install -y mesa-vulkan-drivers
          export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

      - name: Run BLAZE CTS
        run: |
          cd blaze
          zig build test-cts -- --junit-output=results.xml

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: blaze-cts-${{ matrix.gpu }}
          path: blaze/results.xml

      - name: Upload Diff Images (on failure)
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: golden-diffs-${{ matrix.gpu }}
          path: blaze/test/diffs/

  forge-cts:
    needs: blaze-cts
    runs-on: ubuntu-latest
    steps:
      # Similar structure...

  flux-cts:
    needs: blaze-cts
    runs-on: ubuntu-latest
    steps:
      # Similar structure...

  integration:
    needs: [forge-cts, flux-cts]
    runs-on: ubuntu-latest
    steps:
      # Full stack integration tests...

  benchmarks:
    needs: integration
    runs-on: [self-hosted, gpu]  # Dedicated GPU runner for consistent results
    steps:
      - name: Run Benchmarks
        run: zig build benchmark

      - name: Compare to Baseline
        run: python scripts/compare_benchmarks.py

      - name: Fail on Regression
        run: |
          if grep -q "REGRESSION" benchmark_report.txt; then
            exit 1
          fi
```

---

# Test Runner Implementation

```zig
// test/runner.zig

const std = @import("std");

pub const TestResult = struct {
    name: []const u8,
    category: []const u8,
    status: Status,
    duration_ns: u64,
    message: ?[]const u8 = null,
    diff_image: ?[]const u8 = null,

    pub const Status = enum {
        passed,
        failed,
        skipped,
        timeout,
    };
};

pub const TestRunner = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayList(TestResult),
    filter: ?[]const u8 = null,
    generate_golden: bool = false,
    save_diffs: bool = false,
    junit_output: ?[]const u8 = null,

    pub fn run(self: *TestRunner) !void {
        const suites = [_]TestSuite{
            @import("context/tests.zig").suite,
            @import("buffer/tests.zig").suite,
            @import("texture/tests.zig").suite,
            @import("pipeline/tests.zig").suite,
            @import("compute/tests.zig").suite,
            @import("render/tests.zig").suite,
            @import("sync/tests.zig").suite,
        };

        for (suites) |suite| {
            if (self.filter) |f| {
                if (std.mem.indexOf(u8, suite.name, f) == null) continue;
            }

            try self.runSuite(suite);
        }

        self.printSummary();

        if (self.junit_output) |path| {
            try self.writeJunitXml(path);
        }
    }

    fn runSuite(self: *TestRunner, suite: TestSuite) !void {
        std.debug.print("\n═══ {s} ═══\n", .{suite.name});

        for (suite.tests) |test_case| {
            const start = std.time.nanoTimestamp();

            const result = if (test_case.run()) |_| blk: {
                break :blk TestResult{
                    .name = test_case.name,
                    .category = suite.name,
                    .status = .passed,
                    .duration_ns = @intCast(std.time.nanoTimestamp() - start),
                };
            } else |err| blk: {
                break :blk TestResult{
                    .name = test_case.name,
                    .category = suite.name,
                    .status = .failed,
                    .duration_ns = @intCast(std.time.nanoTimestamp() - start),
                    .message = @errorName(err),
                };
            };

            self.results.append(result) catch unreachable;
            self.printResult(result);
        }
    }

    fn printResult(self: *TestRunner, result: TestResult) void {
        _ = self;
        const symbol = switch (result.status) {
            .passed => "✓",
            .failed => "✗",
            .skipped => "○",
            .timeout => "⏱",
        };

        const color = switch (result.status) {
            .passed => "\x1b[32m",
            .failed => "\x1b[31m",
            .skipped => "\x1b[33m",
            .timeout => "\x1b[33m",
        };

        std.debug.print("  {s}{s}\x1b[0m {s}", .{ color, symbol, result.name });

        if (result.message) |msg| {
            std.debug.print(" - {s}", .{msg});
        }

        std.debug.print(" ({d:.2}ms)\n", .{
            @as(f64, @floatFromInt(result.duration_ns)) / 1_000_000.0,
        });
    }

    fn printSummary(self: *TestRunner) void {
        var passed: usize = 0;
        var failed: usize = 0;
        var skipped: usize = 0;

        for (self.results.items) |r| {
            switch (r.status) {
                .passed => passed += 1,
                .failed => failed += 1,
                .skipped => skipped += 1,
                .timeout => failed += 1,
            }
        }

        std.debug.print("\n", .{});
        std.debug.print("═══════════════════════════════════════\n", .{});
        std.debug.print("         CONFORMANCE RESULTS           \n", .{});
        std.debug.print("═══════════════════════════════════════\n", .{});
        std.debug.print("  Passed:  \x1b[32m{d}\x1b[0m\n", .{passed});
        std.debug.print("  Failed:  \x1b[31m{d}\x1b[0m\n", .{failed});
        std.debug.print("  Skipped: \x1b[33m{d}\x1b[0m\n", .{skipped});
        std.debug.print("  Total:   {d}\n", .{passed + failed + skipped});
        std.debug.print("═══════════════════════════════════════\n", .{});

        if (failed > 0) {
            std.debug.print("\n\x1b[31mFailed tests:\x1b[0m\n", .{});
            for (self.results.items) |r| {
                if (r.status == .failed) {
                    std.debug.print("  • {s}.{s}", .{ r.category, r.name });
                    if (r.message) |msg| {
                        std.debug.print(": {s}", .{msg});
                    }
                    std.debug.print("\n", .{});
                }
            }
        }
    }
};
```

---

# Summary

| Library | Test Count | Categories | Verification |
|---------|------------|------------|--------------|
| **BLAZE** | ~500 | 10 categories | API, numerical, golden images |
| **FORGE** | ~300 | 7 categories | Golden images, culling stats |
| **FLUX** | ~200 | 5 categories | Layout rects, golden images, events |
| **Integration** | ~50 | 1 category | Full stack golden images |
| **Total** | **~1050** | | |

## Priority Distribution

| Priority | Description | Count |
|----------|-------------|-------|
| P0 | Critical - Must pass for release | ~400 |
| P1 | Important - Should pass for release | ~400 |
| P2 | Nice to have - May be deferred | ~250 |

## Milestone Targets

| Milestone | Tests Passing | Description |
|-----------|---------------|-------------|
| **v0.1** | 100 P0 tests | Core functionality works |
| **v0.2** | All P0 tests | Release candidate quality |
| **v0.3** | All P0+P1 tests | Production ready |
| **v1.0** | All tests | Full conformance |

---

*This document is the north star for conformance testing. Update it as new features are added.*
