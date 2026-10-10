#!/bin/sh
# Build the XMX-vs-shader-core int8 GEMM benchmark. Requires: glslangValidator, gcc, libvulkan.
set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "compiling shaders..."
for s in gemm_xmx gemm_simt gemm_dp4a gemm_f16_xmx gemm_f16_simt; do
    glslangValidator -S comp -V --target-env vulkan1.3 -o "bin/$s.spv" "shaders/$s.comp" >/dev/null
done

echo "building caps + bench..."
gcc -O2 -Iinclude src/caps.c  -o bin/caps  -lvulkan
gcc -O2 -Iinclude src/bench.c -o bin/bench -lvulkan

echo "done."
echo "  ./bin/caps                       # list cooperative-matrix configs"
echo "  ./bin/bench 2048 2048 2048       # XMX vs shader-core int8 GEMM"
