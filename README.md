# Zig Graphics Stack

A Zig-native graphics stack for building high-performance applications.

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

## Libraries

| Library | Description | Repository |
|---------|-------------|------------|
| **BLAZE** | Lean GPU abstraction over Vulkan | [blaze](https://github.com/hotschmoe/blaze) |
| **FORGE** | GPU-driven 3D scene renderer | [forge](https://github.com/hotschmoe/forge) |
| **FLUX** | GPU-accelerated UI framework | [flux](https://github.com/hotschmoe/flux) |

## Design Principles

1. **Comptime over runtime** - Push decisions to compile time wherever possible
2. **Explicit over implicit** - No hidden allocations, no magic lifetimes
3. **Data-oriented** - Structs of arrays, cache-friendly layouts, minimal indirection
4. **Zero-cost abstractions** - Abstractions should compile away entirely
5. **Incremental complexity** - Simple things simple, complex things possible

## Getting Started

Clone with submodules:

```bash
git clone --recursive git@github.com:hotschmoe/zig-graphics.git
```

Update all submodules:

```bash
git pull && git submodule update --remote
```

## Status

🚧 **Work in Progress** - Not ready for production use.

## License

MIT © hotschmoe
