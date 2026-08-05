#pragma once
#include <atomic>

namespace ffpa {

enum class CudaBackendImpl : int {
  AUTO = 0,
  NATIVE = 1,
  TMA = 2,
  CUTE = 3,
  CUTE_TMA = 4,
  CUTE_TMA_W8A8 = 5,
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
