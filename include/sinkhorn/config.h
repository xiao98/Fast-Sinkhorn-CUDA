#pragma once
/**
 * @file config.h
 * @brief Configuration structures and constants for the Sinkhorn solver.
 *
 * This header defines all tunable parameters for the regularized optimal
 * transport computation, including regularization strength, convergence
 * criteria, and CUDA execution parameters.
 */

#include <cstddef>
#include <limits>

namespace fastsinkhorn {

// ============================================================================
// Numerical Constants
// ============================================================================

/// Machine epsilon for float, used in convergence checks
constexpr float kFloatEps = 1e-7f;

/// Default regularization parameter epsilon
constexpr float kDefaultEpsilon = 1e-2f;

/// Maximum safe value for exponential to avoid overflow
constexpr float kMaxExpArg = 80.0f;

// ============================================================================
// CUDA Execution Parameters
// ============================================================================

/// Number of threads per block for 1D kernels
constexpr int kBlockSize = 256;

/// Tile size for shared memory tiling in Sinkhorn kernel
constexpr int kTileSize = 32;

/// Warp size (NVIDIA GPUs)
constexpr int kWarpSize = 32;

// ============================================================================
// Solver Configuration
// ============================================================================

/**
 * @brief Configuration parameters for the Sinkhorn solver.
 *
 * Controls the behavior of the entropic regularized optimal transport solver.
 * The regularization parameter ε controls the trade-off between approximation
 * quality and convergence speed:
 *   - Small ε → closer to exact OT, but slower convergence
 *   - Large ε → faster convergence, but blurrier transport plan
 */
struct SinkhornConfig {
    /// Regularization parameter ε for entropic regularization.
    /// The regularized problem is: min_{π ∈ Π(μ,ν)} <C, π> + ε·KL(π | μ⊗ν)
    float epsilon = kDefaultEpsilon;

    /// Maximum number of Sinkhorn iterations
    int max_iterations = 1000;

    /// Convergence threshold on marginal constraint violation.
    /// Stops when ||π·1 - μ||₁ < threshold
    float convergence_threshold = 1e-6f;

    /// Check convergence every N iterations (avoid frequent GPU→CPU transfers)
    int convergence_check_interval = 10;

    /// Whether to use log-domain stabilization (strongly recommended).
    /// Standard domain may cause numerical overflow for small ε.
    bool use_log_domain = true;

    /// Enable detailed GPU memory profiling
    bool enable_profiling = false;

    /// Verbose output (iteration-level convergence info)
    bool verbose = false;
};

/**
 * @brief Result structure returned by the Sinkhorn solver.
 */
struct SinkhornResult {
    /// Regularized optimal transport cost: <C, π*>
    float transport_cost = 0.0f;

    /// Regularized OT distance (including entropy term):
    /// <C, π*> + ε·KL(π* | μ⊗ν)
    float sinkhorn_distance = 0.0f;

    /// Number of iterations until convergence
    int iterations = 0;

    /// Final marginal constraint violation
    float marginal_error = 0.0f;

    /// Whether the solver converged within max_iterations
    bool converged = false;

    /// Total GPU computation time in milliseconds
    float elapsed_ms = 0.0f;

    /// Peak GPU memory usage in bytes
    size_t peak_memory_bytes = 0;
};

} // namespace fastsinkhorn
