# xmx-bench

A tiny, self-contained Vulkan benchmark that answers one question:

> **Can a Linux Vulkan compute shader actually reach Intel's XMX matrix units,
> and how much faster is that than the shader-core / DP4A paths?**

It runs the same GEMM several ways and compares throughput, checking every
kernel against a CPU reference first.

| kernel | how | unit |
|---|---|---|
| `XMX-i8`   | `VK_KHR_cooperative_matrix`, int8 `M8 N16 K32`, subgroup scope | **XMX / DPAS** |
| `DP4A-i8`  | `GL_EXT_integer_dot_product` `dotPacked4x8EXT` (`OpSDot`) on shared tiles | DP4A |
| `SIMT-i8`  | classic shared-memory-tiled int8 MAC | shader cores (scalar) |
| `XMX-f16`  | cooperative matrix, fp16 `M8 N16 K16`, fp32 accumulate | **XMX / DPAS** |
| `SIMT-f16` | shared-memory-tiled fp16 MAC | shader cores |

## Headline result (Intel Arc 140V / Lunar Lake, Mesa 26.2.4, 2048³)

```
XMX-i8    5228 GOPS    1.00x
SIMT-i8    368 GOPS    0.07x     (14.2x slower)
DP4A-i8   1671 GOPS    0.32x     ( 3.1x slower)
XMX-f16   4565 GOPS    0.87x
SIMT-f16   580 GOPS    0.11x
```

The cooperative-matrix path is **~14× the scalar shader-core path** and **~3× a
hand-written DP4A (`OpSDot`) kernel** — the signature of XMX being used. See
[RESULTS.md](RESULTS.md) for the full tables and caveats.

## Why this matters

`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` reporting an int8
`M8 N16 K32` config proves the *extension* is advertised; it does not prove a
shader actually executes on XMX. This benchmark executes it, verifies the
result, and measures the gap.

## Build

Requires `glslangValidator`, `gcc`, and the Vulkan loader. On Arch/CachyOS:

    sudo pacman -S --needed glslang gcc vulkan-icd-loader vulkan-intel

The Khronos Vulkan headers are vendored under `include/` (Apache-2.0), so no
`vulkan-headers` package is needed.

    ./build.sh
    ./bin/caps                    # list device features + cooperative-matrix configs
    ./bin/bench 2048 2048 2048    # run the GEMMs (M,N,K must be multiples of 32)

## Design notes

- **No descriptor sets.** A/B/C buffers are handed to the shaders as device
  addresses via `GL_EXT_buffer_reference`, passed in push constants.
- **Timing** uses `VkQueryPool` GPU timestamps, not host fences.
- `XMX-*`: one workgroup = one subgroup = a 16×32 output tile built from a 2×2
  grid of 8×16 cooperative-matrix MMAs per K-step.
- `DP4A-i8`: reads A (M×K) and Bᵀ (N×K) as packed 4×int8 words, stages tiles in
  shared memory, and accumulates with `dotPacked4x8EXT`.
- Nothing here is a tuned BLAS; the point is reachability and a reproducible
  gap, not a peak-throughput record.

## Files

    src/caps.c             enumerate cooperative-matrix configs / features
    src/bench.c            host: buffers, pipelines, correctness, GPU timing
    shaders/gemm_xmx.comp       int8 cooperative-matrix (XMX)
    shaders/gemm_dp4a.comp      int8 OpSDot DP4A baseline
    shaders/gemm_simt.comp      int8 scalar shader-core baseline
    shaders/gemm_f16_xmx.comp   fp16 cooperative-matrix (XMX)
    shaders/gemm_f16_simt.comp  fp16 shader-core baseline
    include/               vendored Khronos Vulkan headers

## License

MIT for this project's code; the vendored `include/vulkan` and
`include/vk_video` headers are Apache-2.0 (Khronos Group).
