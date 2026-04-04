/**
 * @file memory_profiler.cpp
 * @brief GPU memory and performance profiling implementation.
 *
 * Uses CUDA runtime API for:
 *   - cudaMemGetInfo: GPU memory usage snapshots
 *   - cudaEvent: Precise kernel execution timing
 *
 * Provides formatted profiling reports for analyzing GPU resource usage
 * across different stages of the Sinkhorn computation.
 */

#include "sinkhorn/memory_profiler.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

namespace fastsinkhorn {

// ============================================================================
// PIMPL Implementation
// ============================================================================

struct MemoryProfiler::Impl {
    std::vector<ProfileRecord> records;
    cudaEvent_t current_start = nullptr;
    cudaEvent_t current_stop  = nullptr;
    std::string current_label;

    size_t initial_free  = 0;
    size_t initial_total = 0;

    Impl() {
        cudaEventCreate(&current_start);
        cudaEventCreate(&current_stop);

        // Record initial GPU memory state
        cudaMemGetInfo(&initial_free, &initial_total);
    }

    ~Impl() {
        if (current_start) cudaEventDestroy(current_start);
        if (current_stop)  cudaEventDestroy(current_stop);
    }
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

MemoryProfiler::MemoryProfiler() : pImpl_(new Impl()) {}

MemoryProfiler::~MemoryProfiler() { delete pImpl_; }

// ============================================================================
// Memory Snapshots
// ============================================================================

void MemoryProfiler::snapshot(const std::string& label) {
    ProfileRecord record;
    record.label = label;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    record.memory_free = free_mem;
    record.memory_used = total_mem - free_mem;
    record.elapsed_ms = 0.0f;

    pImpl_->records.push_back(record);
}

// ============================================================================
// Timing Events
// ============================================================================

void MemoryProfiler::startEvent(const std::string& label) {
    pImpl_->current_label = label;
    cudaEventRecord(pImpl_->current_start);
}

void MemoryProfiler::stopEvent() {
    cudaEventRecord(pImpl_->current_stop);
    cudaEventSynchronize(pImpl_->current_stop);

    ProfileRecord record;
    record.label = pImpl_->current_label;
    cudaEventElapsedTime(&record.elapsed_ms, pImpl_->current_start, pImpl_->current_stop);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    record.memory_free = free_mem;
    record.memory_used = total_mem - free_mem;

    pImpl_->records.push_back(record);
}

// ============================================================================
// Reporting
// ============================================================================

size_t MemoryProfiler::getPeakMemoryUsage() const {
    size_t peak = 0;
    for (const auto& r : pImpl_->records) {
        peak = std::max(peak, r.memory_used);
    }
    return peak;
}

const std::vector<ProfileRecord>& MemoryProfiler::getRecords() const {
    return pImpl_->records;
}

void MemoryProfiler::printReport() const {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                  GPU Memory Profiling Report                ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ GPU Total Memory: %8.1f MB                               ║\n",
           pImpl_->initial_total / (1024.0 * 1024.0));
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ %-25s │ %8s │ %8s │ %8s ║\n",
           "Stage", "Time(ms)", "Used(MB)", "Free(MB)");
    printf("╠══════════════════════════════════════════════════════════════╣\n");

    for (const auto& r : pImpl_->records) {
        printf("║ %-25.25s │ %8.2f │ %8.1f │ %8.1f ║\n",
               r.label.c_str(),
               r.elapsed_ms,
               r.memory_used / (1024.0 * 1024.0),
               r.memory_free / (1024.0 * 1024.0));
    }

    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Peak Memory Usage: %8.1f MB                              ║\n",
           getPeakMemoryUsage() / (1024.0 * 1024.0));
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

void MemoryProfiler::reset() {
    pImpl_->records.clear();
}

} // namespace fastsinkhorn
