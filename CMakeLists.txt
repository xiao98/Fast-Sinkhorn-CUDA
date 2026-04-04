cmake_minimum_required(VERSION 3.18)
project(FastSinkhornCUDA LANGUAGES CXX CUDA)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architecture (auto-detect or set manually)
# Common architectures: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86, 89, 90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# Compiler Flags
# ============================================================================
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -DDEBUG")

# ============================================================================
# Core Library
# ============================================================================
add_library(fast_sinkhorn STATIC
    src/sinkhorn_kernel.cu
    src/cost_matrix.cu
    src/sinkhorn_solver.cpp
    src/memory_profiler.cpp
)

target_include_directories(fast_sinkhorn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fast_sinkhorn PUBLIC
    CUDA::cudart
)

# ============================================================================
# Examples
# ============================================================================
add_executable(basic_ot examples/basic_ot.cpp)
target_link_libraries(basic_ot PRIVATE fast_sinkhorn)

add_executable(benchmark examples/benchmark.cpp)
target_link_libraries(benchmark PRIVATE fast_sinkhorn)

# ============================================================================
# Tests
# ============================================================================
add_executable(test_sinkhorn tests/test_sinkhorn.cpp)
target_link_libraries(test_sinkhorn PRIVATE fast_sinkhorn)

# ============================================================================
# Experiments (for paper)
# ============================================================================

# Baseline benchmark (outputs CSV)
add_executable(bench_ours experiments/baselines/bench_ours.cpp)
target_link_libraries(bench_ours PRIVATE fast_sinkhorn)

# Scaling experiment
add_executable(scaling_experiment experiments/scaling/scaling_experiment.cpp)
target_link_libraries(scaling_experiment PRIVATE fast_sinkhorn)

# Ablation: warp shuffle vs shared memory
add_executable(ablation_warp_shuffle experiments/ablation/ablation_warp_shuffle.cu)
target_link_libraries(ablation_warp_shuffle PRIVATE fast_sinkhorn)

# Ablation: log-domain vs standard-domain
add_executable(ablation_log_domain experiments/ablation/ablation_log_domain.cpp)
target_link_libraries(ablation_log_domain PRIVATE fast_sinkhorn)

# Ablation: block size
add_executable(ablation_block_size experiments/ablation/ablation_block_size.cu)
target_link_libraries(ablation_block_size PRIVATE fast_sinkhorn)

# Ablation: convergence check interval
add_executable(ablation_check_interval experiments/ablation/ablation_check_interval.cpp)
target_link_libraries(ablation_check_interval PRIVATE fast_sinkhorn)

# Stability experiment
add_executable(stability_epsilon experiments/stability/stability_epsilon.cpp)
target_link_libraries(stability_epsilon PRIVATE fast_sinkhorn)

# Convergence profiling
add_executable(convergence_profile experiments/convergence/convergence_profile.cpp)
target_link_libraries(convergence_profile PRIVATE fast_sinkhorn)

# ============================================================================
# Install
# ============================================================================
install(TARGETS fast_sinkhorn ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS basic_ot benchmark test_sinkhorn RUNTIME DESTINATION bin)
