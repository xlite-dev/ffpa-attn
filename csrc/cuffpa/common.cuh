#pragma once

// Project-wide numeric constants shared across cuffpa kernels. Centralized
// here so the softmax / rescale helpers (prefill.cuh) and the CuTe TMA kernel
// (cute/fwd_sm120.cuh) share a single source of truth. Add further
// shared macros / constants here as needed.

// exp2f softmax optimization: expf(x) == exp2f(x * FFPA_M_LOG2E).
#define FFPA_M_LOG2E 1.44269504088896340736f
// Inverse of FFPA_M_LOG2E: convert a log2-domain value back to natural-log.
#define FFPA_M_LN2 0.69314718055994530942f
