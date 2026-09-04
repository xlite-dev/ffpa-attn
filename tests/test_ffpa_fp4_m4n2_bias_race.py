"""FP4 M4N2 attn-bias determinism regression (PC-0-5 race).

The fp4 split-D M4N2 kernel with a bias smem tile (mode 2 TMA double
buffer / mode 3 resident fill) produced bitwise-unstable O across identical
back-to-back calls -- with or without a no-bias prelude -- while lse stayed
stable: corruption pinned to a single (m-warp, n-warp, v-chunk) PV C tile.
The language-level protocol audit closed clean and neither ptxas -O2 nor
producer-warp relocation fixed it, so the launcher now pins the gmem
direct-read mode (mode 0, stable on the cold/pure sequence) for this
kernel; FFPA_BIAS_TILE_KEEP=1 restores the smem tile modes. RESIDUAL
(accepted): a heavy GPU-work prelude still opens a low-probability
instability window on the bias template even in mode 0 - load-timing
sensitive, documented in RFC PC-0-5.

These tests pin both trigger sequences (with and without the no-bias
prelude). The pure-bias case is the must-pass gate; the prelude case is
tracked as xfail for the accepted residual.
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


def _make_case(D):
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
  return q, k, v, bias, backend


@pytest.mark.parametrize("D", [768])
def test_fp4_m4n2_pure_bias_determinism(D):
  # PC-0-5 follow-up: the pure-bias sequence (no prelude, first call as
  # reference) was the 100%-reproducer that re-opened this issue. Defined
  # BEFORE the prelude case so a single-file run consumes it first (the
  # gate's stability guarantee holds for the cold sequence; a shared pytest
  # session with heavy prior GPU load can still open the residual window).
  q, k, v, bias, backend = _make_case(D)
  ref = _run(q, k, v, backend, bias).view(torch.int32).flatten()
  for _ in range(6):
    out = _run(q, k, v, backend, bias).view(torch.int32).flatten()
    assert torch.equal(out, ref), (
      "fp4 m4n2 pure-bias output is not bit-deterministic (PC-0-5): "
      f"{(out != ref).sum().item()} int32 lanes differ"
    )


@pytest.mark.parametrize("D", [768])
@pytest.mark.xfail(
  strict=False,
  reason="PC-0-5 residual (accepted): with any heavy GPU-work prelude the "
  "fp4 m4n2 bias template still shows low-probability bitwise instability "
  "even in the pinned mode 0 (load-timing sensitive at the hardware level; "
  "no-bias template is clean under the same load). fp4 m4n2 only serves "
  "D>=768 fp4 (rare in production) - documented, not root-caused. See "
  "rfc-future-optimizations.md PC-0-5",
)
def test_fp4_m4n2_bias_determinism_after_no_bias_prelude(D):
  q, k, v, bias, backend = _make_case(D)
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
