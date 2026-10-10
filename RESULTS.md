# xmx-bench — results

## Environment

| | |
|---|---|
| GPU | Intel Arc 130V/140V (Lunar Lake / Xe2), `8086:64a0`, `Intel(R) Graphics (LNL)` |
| Kernel | `7.1.8-1-cachyos-custom` (`xe` KMD) |
| Mesa / Vulkan | Mesa **26.2.4** (ANV), device API 1.4.354, loader 1.4.363 |
| Tools | glslang 16.6.0, gcc, `VK_EXT`… (no proprietary driver) |

## Device capabilities (`./bin/caps`)

```
ext: cooperative_matrix=1 integer_dot_product=1 float16_int8=1
features: shaderFloat16=1 shaderInt8=1 integerDotProduct=1 cooperativeMatrix=1
subgroup: size=32
cooperative matrix configs: count=6
  [0] M=8 N=16 K=16  f16 x f16 -> f16     scope=SUBGROUP
  [1] M=8 N=16 K=16  f16 x f16 -> f32     scope=SUBGROUP
  [4] M=8 N=16 K=32  s8  x s8  -> s32     scope=SUBGROUP
  [5] M=8 N=16 K=32  u8  x u8  -> u32     scope=SUBGROUP
```

The `M8 N16 K32` int8 config (#4) is the shape XeSS's XMX/DPAS path uses.

## Throughput (`./bin/bench N N N`)

GEMM `C = A(MxK) * B(KxN)`, ops = 2·M·N·K. **GPU timestamps** (VkQueryPool),
best of 5 samples. Correctness is checked against a CPU reference before timing
(200 sampled outputs; all five kernels pass in every run).

### int8 (int32 accumulate)

| M=N=K | XMX (coop matrix) | shader cores (scalar) | **DP4A** (`OpSDot`) |
|---|---:|---:|---:|
| 1024 | 3718 GOPS | 355 GOPS | 1624 GOPS |
| 2048 | **5228 GOPS** | 368 GOPS | 1671 GOPS |
| 4096 | 5186 GOPS | 354 GOPS | 1649 GOPS |

### fp16 (fp32 accumulate)

| M=N=K | XMX (coop matrix) | shader cores |
|---|---:|---:|
| 1024 | 4796 GOPS | 570 GOPS |
| 2048 | 4565 GOPS | 580 GOPS |
| 4096 | 2419 GOPS | 530 GOPS |

### Headline ratios (2048³, int8)

- **XMX vs scalar shader cores: 14.2×**  (5228 vs 368 GOPS)
- **XMX vs DP4A (`OpSDot`): 3.1×**    (5228 vs 1671 GOPS)
- **DP4A vs scalar shader cores: 4.5×** (1671 vs 368 GOPS)

Run-to-run variance is roughly ±15% (iGPU clock/thermal); the ratios are stable.

## Interpretation

- The `VK_KHR_cooperative_matrix` path **executes on Linux** on this hardware:
  ~5.2 TOPS int8 and ~4.6 TOPS fp16, with correct results.
- It is **~14× the scalar shader-core path** and **~3× the DP4A path** at the
  same problem. A gap of this size against a hand-written `OpSDot` DP4A kernel
  is the signature of the matrix engine (XMX/DPAS) being used — if the
  cooperative-matrix path were lowered onto the EUs, the `XMX` and `DP4A`
  columns would be within a small factor.
- Conclusion: **XMX hardware + Mesa/ANV are ready.** What is missing for XeSS
  on Linux is purely the XeSS runtime side: the shipped Windows `libxess.dll`
  D3D12 path still selects the cross-vendor implementation under Proton
  (`xessIsOptimalDriver -> WARNING_OLD_DRIVER`), so games run the DP4A model.

## Caveats (read before quoting)

- These are **demonstrator kernels, not tuned BLAS**. `SIMT-*` is a plain
  16×16-tiled shared-memory GEMM, `DP4A-i8` stages packed words in shared
  memory and uses `dotPacked4x8EXT`, `XMX-*` is a 16×32-tile cooperative-matrix
  GEMM. The **ratios** matter more than the absolute numbers.
- `SIMT-i8` is "scalar shader-core int8"; `DP4A-i8` is the hardware dot-product
  path. They are deliberately different baselines.
- Timed with GPU timestamps around batched dispatches (`reps` chosen so a single
  submit stays short enough to avoid the GPU watchdog).
- The f16 configs `M8 N16 K16` were tested; the `u8` config was not.

## Reproduce

```
./build.sh
./bin/caps
./bin/bench 2048 2048 2048
```
