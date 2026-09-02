"""FP4 M4N2 attn-bias determinism regression (PC-0-5 race).

The fp4 split-D M4N2 kernel reuses one smem region for both the O epilogue
staging and the P softmax round-trip tile. Repeated calls of the bias
template back-to-back behind a no-bias prelude used to race: the tile-0 P
scatter could overtake the previous work's epilogue staging read, showing
as non-deterministic stripe-shaped O errors (~5e-3..3e-2) with a stable
lse. The fix orders every kv tile with a full-CTA barrier.

This test pins the trigger sequence (no-bias prelude, then the row-broadcast
bias template) and requires bit-identical outputs across back-to-back calls.
"""

import math

import pytest
import torch

from ffpa_attn.functional import CUDABackend

try:
  from ffpa_attn.cuda import set_cuda_backend_impl, CudaBackendImpl
  from ffpa_attn.cuda._ffpa_fwd import _ffpa_attn_forward_cuda
  FFPA_CUDA_EXT_BUILT = True
except Exception:  # pragma: no cover
  FFPA_CUDA_EXT_BUILT = False


def _fp4_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


pytestmark = [
  pytest.mark.skipif(not _fp4_available(), reason="fp4 path requires sm_120"),
  pytest.mark.skipif(not FFPA_CUDA_EXT_BUILT, reason="ffpa CUDA ext required"),
]


def _run(q, k, v, backend, bias):
  set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_FP4)
  o, _ = _ffpa_attn_forward_cuda(
    q, k, v, None, bias, backend.stages, backend.acc_code, 0,
    1.0 / math.sqrt(q.size(-1)), 0.0, 0, 0, backend.fp8_smooth_k,
    backend.fp8_smooth_v, backend.fp8_q_quant_method_code,
    backend.fp8_k_quant_method_code, backend.fp8_v_quant_method_code,
    backend.fp8_pv_acc_code, backend.fp8_qk_mm_type_code, backend.fp8_hybrid,
    backend.fp8_hybrid_n_early, backend.fp4_hybrid, backend.fp4_hybrid_n_early,
    backend.fp8_hadamard, backend.fp4_hadamard, backend.fp4_pv_mm_type_code,
    backend.fp4_smooth_v, backend.tensor_layout_code
  )
  return o


@pytest.mark.parametrize("D", [768])
@pytest.mark.xfail(
  strict=False,
  reason="PC-0-5: fp4 m4n2 bias output is not bit-deterministic after a "
  "no-bias prelude (cross-template timing-sensitive race; both necessary "
  "ingredients confirmed: no-bias kernel prelude + the mode-3 bias "
  "load-section stores. See rfc-future-optimizations.md PC-0-5)",
)
def test_fp4_m4n2_bias_determinism_after_no_bias_prelude(D):
  torch.manual_seed(0)
  B, H, N = 1, 4, 2048
  q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
  k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
  v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
  bias = torch.randn(1, 1, 1, N, device="cuda", dtype=torch.float16) * 0.25

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=True,
    enable_tma=True,
    enable_cute=True,
    backward=False,
  )
  backend.fp8_hybrid = False
  backend.fp4_hybrid = False

  for _ in range(9):
    _run(q, k, v, backend, None)

  ref = _run(q, k, v, backend, bias).view(torch.int32).flatten()
  for _ in range(6):
    out = _run(q, k, v, backend, bias).view(torch.int32).flatten()
    assert torch.equal(out, ref), (
      "fp4 m4n2 bias output is not bit-deterministic across back-to-back "
      "calls (PC-0-5 race regression): "
      f"{(out != ref).sum().item()} int32 lanes differ"
    )
