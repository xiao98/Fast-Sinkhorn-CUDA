#pragma once
/**
 * @file memory_profiler.h
 * @brief GPU memory and timing profiler for Sinkhorn solver.
 *
 * Provides detailed tracking of GPU memory consumption and kernel execution
 * times. Uses cudaMemGetInfo for memory snapshots and cudaEvent for precise
 * kernel timing.
 */

#include <string>
#include <vector>
#include <cstddef>

namespace fastsinkhorn {

/**
 * @brief A single profiling record for a GPU operation.
 */
struct ProfileRecord {
    std::string label;           ///< Operation name
    float elapsed_ms = 0.0f;     ///< Kernel execution time (ms)
    size_t memory_used = 0;      ///< GPU memory used after operation (bytes)
    size_t memory_free = 0;      ///< GPU memory free after operation (bytes)
};

/**
 * @brief GPU memory and performance profiler.
 *
 * Usage:
 * @code
 *   MemoryProfiler profiler;
 *   profiler.startEvent("SinkhornIteration");
 *   // ... launch kernel ...
 *   profiler.stopEvent();
 *   profiler.printReport();
 * @endcode
 */
class MemoryProfiler {
public:
    MemoryProfiler();
    ~MemoryProfiler();

    /// Record a memory snapshot with a label
    void snapshot(const std::string& label);

    /// Start timing a GPU event
    void startEvent(const std::string& label);

    /// Stop timing and record the elapsed time
    void stopEvent();

    /// Get peak memory usage across all snapshots (bytes)
    size_t getPeakMemoryUsage() const;

    /// Get all profiling records
    const std::vector<ProfileRecord>& getRecords() const;

    /// Print a formatted profiling report to stdout
    void printReport() const;

    /// Reset all records
    void reset();

private:
    struct Impl;
    Impl* pImpl_;  ///< PIMPL pattern to hide CUDA types from host code
};

} // namespace fastsinkhorn
