"""Lazy-rescale per-row governance determinism (PC-11).

PC-11 replaced every warp-uniform __any_sync lazy-rescale vote with a
thread-level per-row guard (``row_scale < 1.0f``); the fp4 m4n2 kernel is
exempt (its vote is load-bearing for the PC-0-5 pure-sequence stability,
covered by test_ffpa_fp4_m4n2_bias_race). Removing the vote must not open
bitwise instability: after a no-bias prelude, a bias self-loop must stay
bit-identical.

The all-masked-row case pins the documented per-family behavior around the
per-row guard: fp4's InfCheck softmax drives scores_scale to exactly 0.0
and the guarded finalize emits zeros (no NaN anywhere); the fp16 family
has no row_sum guard, so a fully masked row is NaN (SDPA-consistent) --
what matters for PC-11 is that the NaN stays confined to the masked row
(the old warp-uniform vote let one row's rescale decision leak across the
warp).
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

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not FFPA_CUDA_EXT_BUILT,
  reason="ffpa CUDA ext on a CUDA device required",
)

# family -> (impl, enable_fp8, enable_fp4, D); D picks the kernel family
# segment (persist_d / split_d / split_d_m4n2) per the D dispatch table.
# fp4 m4n2 is exempt from PC-11 (its vote is load-bearing for the PC-0-5
# pure-sequence stability) and stays covered by test_ffpa_fp4_m4n2_bias_race.
_CASES = [
  ("fp16_persist", CudaBackendImpl.CUTE_TMA, False, False, 128),
  ("fp16_split", CudaBackendImpl.CUTE_TMA, False, False, 320),
  ("fp8_split", CudaBackendImpl.CUTE_TMA_FP8, True, False, 320),
  ("fp4_persist", CudaBackendImpl.CUTE_TMA_FP4, False, True, 128),
  ("fp4_split", CudaBackendImpl.CUTE_TMA_FP4, False, True, 320),
]


def _backend(enable_fp8, enable_fp4):
  backend = CUDABackend(
    forward=True,
    enable_fp8=enable_fp8,
    enable_fp4=enable_fp4,
    enable_tma=True,
    enable_cute=True,
    backward=False,
  )
  backend.fp8_hybrid = False
  backend.fp4_hybrid = False
  return backend


def _run(impl, q, k, v, backend, bias):
  set_cuda_backend_impl(impl)
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


def _make_inputs(D):
  torch.manual_seed(0)
  B, H, N = 1, 4, 2048
  q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) * 0.5
  k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) * 0.5
  v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) * 0.5
  # bias must match Q dtype (fp32 masks need the trimmed f=1 variants).
  bias = torch.randn(1, 1, 1, N, device="cuda", dtype=torch.bfloat16) * 0.25
  return q, k, v, bias


def _run_or_skip_headdim(impl, q, k, v, backend, bias):
  try:
    return _run(impl, q, k, v, backend, bias)
  except RuntimeError as exc:
    if "headdim not support" in str(exc):
      pytest.skip(f"D={q.size(-1)} not in the compiled headdim set")
    raise


@pytest.mark.parametrize("name,impl,fp8,fp4,D", _CASES)
def test_bias_selfloop_determinism(name, impl, fp8, fp4, D):
  q, k, v, bias = _make_inputs(D)
  backend = _backend(fp8, fp4)
  # fp4 m4n2 is deliberately absent from _CASES (vote exemption, see
  # test_ffpa_fp4_m4n2_bias_race); every case here takes the prelude.
  for _ in range(3):
    _run_or_skip_headdim(impl, q, k, v, backend, None)

  ref = _run_or_skip_headdim(impl, q, k, v, backend, bias)
  ref_bits = ref.view(torch.int16).flatten()
  for _ in range(3):
    out = _run_or_skip_headdim(impl, q, k, v, backend, bias)
    out_bits = out.view(torch.int16).flatten()
    assert torch.equal(out_bits, ref_bits), (
      f"{name} bias self-loop output is not bit-deterministic (PC-11): "
      f"{(out_bits != ref_bits).sum().item()} int16 lanes differ"
    )


@pytest.mark.parametrize(
  "name,impl,fp8,fp4,D",
  [c for c in _CASES if c[0] in ("fp16_persist", "fp4_persist")],
)
def test_all_masked_bias_row_behavior(name, impl, fp8, fp4, D):
  q, k, v, _ = _make_inputs(D)
  N = q.size(2)
  bias = torch.zeros(1, 1, N, N, device="cuda", dtype=torch.bfloat16)
  bias[:, :, 0, :] = float("-inf")
  backend = _backend(fp8, fp4)
  out = _run_or_skip_headdim(impl, q, k, v, backend, bias)
  if fp4:
    assert not torch.isnan(out).any(), f"{name}: NaN leaked into O"
    assert out[0, :, 0].abs().max().item(
    ) == 0.0, (f"{name}: fully masked row must finalize to zeros")
  else:
    # fp16/fp8 have no row_sum guard: fully masked rows are NaN (SDPA
    # parity); the masked row must not corrupt its warp-mate rows.
    assert torch.isnan(out[0, :, 0]).all()
  assert torch.isfinite(
    out[0, :, 1:]
  ).all(), (f"{name}: masked row corrupted neighbouring rows")
