cmake_minimum_required(VERSION 3.27)
project(transformer CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit)


add_compile_options(-Wno-changes-meaning)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

add_library(transformer_lib
        src/vendor/argparse.hpp
        src/vendor/safetensors.hh
        src/vendor/safetensors_impl.cpp
        src/CudaBuffer.cuh
        src/CudaBuffer.cu
        src/ErrorCheck.h
        src/qwen2/Qwen2Config.h
        src/qwen2/Qwen2Layer.cuh
        src/qwen2/Qwen2Model.cuh
        src/qwen2/Qwen2Loader.h
        src/qwen2/Qwen2Loader.cu
        src/HostBuffer.cpp
        src/HostBuffer.h
        src/HostBuffer.cpp
        src/HostBuffer.h
        src/BPE.cpp
        src/BPE.h
        src/gpu_ops/ArgMax.cu
        src/gpu_ops/ArgMax.cuh
        src/gpu_ops/LayerNorm.cu
        src/gpu_ops/LayerNorm.cuh
        src/gpu_ops/RoPE.cu
        src/gpu_ops/RoPE.cuh
        src/gpu_ops/MatrixVectorMultiply.cu
        src/gpu_ops/MatrixVectorMultiply.cuh
        src/gpu_ops/SiLUMult.cu
        src/gpu_ops/SiLUMult.cuh
        src/gpu_ops/GroupQueryAttention.cuh)

target_link_libraries(transformer_lib PRIVATE CUDA::cudart CUDA::nvtx3)

add_executable(transformer
        src/vendor/argparse.hpp
        src/main.cu
)
target_link_libraries(transformer PRIVATE transformer_lib)

# tests
enable_testing()

add_executable(bpetest ./src/tests/BPETest.cu)
target_link_libraries(bpetest PRIVATE transformer_lib)
add_test(NAME bpetest COMMAND bpetest)

add_executable(argmaxtest ./src/tests/ArgMaxTest.cu)
target_link_libraries(argmaxtest PRIVATE transformer_lib)
add_test(NAME argmaxtest COMMAND argmaxtest)

add_executable(matvecmulttest ./src/tests/MatVecMultTest.cu)
target_link_libraries(matvecmulttest PRIVATE transformer_lib)
add_test(NAME matvecmulttest COMMAND matvecmulttest)

add_executable(layernormtest ./src/tests/LayerNormTest.cu)
target_link_libraries(layernormtest PRIVATE transformer_lib)
add_test(NAME layernormtest COMMAND layernormtest)

add_executable(ropetest ./src/tests/RoPETest.cu)
target_link_libraries(ropetest PRIVATE transformer_lib)
add_test(NAME ropetest COMMAND ropetest)

add_executable(silumulttest ./src/tests/SiLUMultTest.cu)
target_link_libraries(silumulttest PRIVATE transformer_lib)
add_test(NAME silumulttest COMMAND silumulttest)

add_executable(gqatest ./src/tests/GQATest.cu)
target_link_libraries(gqatest PRIVATE transformer_lib)
add_test(NAME gqatest COMMAND gqatest)

# CPU attention reference -- pure C++, no CUDA dependency.
add_executable(attentioncputest ./src/tests/AttentionCPUTest.cpp)
target_compile_options(attentioncputest PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
add_test(NAME attentioncputest COMMAND attentioncputest)

# Naive CUDA prefill-attention baseline, verified against the CPU reference.
add_executable(naiveprefillattentiontest ./src/tests/NaivePrefillAttentionTest.cu)
target_link_libraries(naiveprefillattentiontest PRIVATE transformer_lib)
# Optimize the host code: the CPU reference (attention_materialized) is timed here, and
# nvcc defaults host code to -O0 -- without this the CPU-vs-GPU speedup is ~10x inflated.
target_compile_options(naiveprefillattentiontest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
add_test(NAME naiveprefillattentiontest COMMAND naiveprefillattentiontest)

# Naive N-query prefill of the full Qwen2 model (GPU), vs the decode loop.
add_executable(prefillmodeltest ./src/tests/PrefillModelTest.cu)
target_link_libraries(prefillmodeltest PRIVATE transformer_lib)
add_test(NAME prefillmodeltest COMMAND prefillmodeltest)

# Fused FlashAttention kernel, verified vs CPU reference + naive GPU, timed vs naive.
add_executable(flashattentiontest ./src/tests/FlashAttentionTest.cu)
target_link_libraries(flashattentiontest PRIVATE transformer_lib)
target_compile_options(flashattentiontest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
add_test(NAME flashattentiontest COMMAND flashattentiontest)

# FlashAttention tile-size sweep (RPB x BC) -- benchmark only, not a ctest.
add_executable(flashattentionsweep ./src/tests/FlashAttentionSweep.cu)
target_link_libraries(flashattentionsweep PRIVATE transformer_lib)
target_compile_options(flashattentionsweep PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)

# FlashAttention wired into the full Qwen2 prefill (bf16) -- end-to-end correctness + tok/s.
add_executable(flashprefilltest ./src/tests/FlashPrefillTest.cu)
target_link_libraries(flashprefilltest PRIVATE transformer_lib)
target_compile_options(flashprefilltest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
add_test(NAME flashprefilltest COMMAND flashprefilltest)

# ncu profiling runners (not ctests), one per src/<name>.cu:
#   profile_kernels          -- decode-path kernels (ArgMax, LayerNorm, MatVec, RoPE, SiLUMult)
#   profile_naive_attention  -- naive prefill-attention baseline kernels
#   profile_flash_attention  -- FlashAttention kernel (same protocol as naive)
foreach(runner IN ITEMS profile_kernels profile_naive_attention profile_flash_attention)
    add_executable(${runner} ./src/${runner}.cu)
    target_link_libraries(${runner} PRIVATE transformer_lib)
    target_compile_options(${runner} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
endforeach()
