#pragma once
#include <atomic>

namespace ffpa {

enum class CudaBackendImpl : int {
  AUTO = 0,
  NATIVE = 1,
  TMA = 2,
  CUTE = 3,
  CUTE_TMA = 4,
  // TMA kernel families; ffpa-attn ships sm_120 variants only. Future
  // Hopper/Blackwell-datacenter specialisations get their own entries
  // (e.g. CUTE_TMA_FP8_SM_90).
  CUTE_TMA_FP8_SM_120 = 5,
  CUTE_TMA_FP4_SM_120 = 6,
  // TMA-free cp.async fp8 persist-D family (D<=224), sm_89 target.
  CUTE_FP8_SM_89 = 7,
};

inline std::atomic<CudaBackendImpl>& backend_impl_hint() {
  static std::atomic<CudaBackendImpl> hint{CudaBackendImpl::AUTO};
  return hint;
}

inline void set_backend_impl_hint(CudaBackendImpl impl) {
  backend_impl_hint().store(impl, std::memory_order_relaxed);
}

inline CudaBackendImpl get_backend_impl_hint() {
  return backend_impl_hint().load(std::memory_order_relaxed);
}

}  // namespace ffpa
