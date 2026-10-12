"""FP8 sm89 (TMA-free cp.async) attention tests: parity vs fp32 SDPA.

Forces the sm89 fp8 family (persist-D D<=224 + split-D M8N1 224<D<=512)
through the ``force_fp8_sm89`` backend flag, which works both on the
sm120 dev box (the force path compiles the sm89 kernels as sm_120f SASS,
the FC-16 force-sm89 contract) and on real Ada hardware (sm_89 cubin).

Acceptance follows the FC-14 cross-implementation contract: each path is
deterministic across processes, outputs allclose against the fp32 SDPA
reference at the documented fp8 noise floor, and vs-fp32 precision
tracks the sm120 family. The retired sm89<->sm120 bitwise contract is
NOT asserted here.

Split-D specifics under test: D-chunked QK/PV (chunks 32/64), Q reload
per kv_tile, per-v_chunk float O accumulators, the perm-pack P layout
paired with the permuted V^T, q_start_row re-bases (hybrid stage-2), and
the 224<D<=512 (D%64==0) dispatch window.
"""

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cuda import (
  CUDA_FP8_SM89_AVAILABLE,
  CUDA_INPUT_FP16_AVAILABLE,
  CUDA_QUANT_SPLIT_D_AVAILABLE,
  set_cuda_backend_impl,
  CudaBackendImpl,
)
from ffpa_attn.cuda._ffpa_fwd import _ffpa_attn_forward_cuda
from ffpa_attn.functional import CUDABackend

pytestmark = [
  pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sm89 fp8 path requires a CUDA GPU",
  ),
  pytest.mark.skipif(
    not CUDA_FP8_SM89_AVAILABLE,
    reason="fp8 sm89 family not compiled; rebuild with the fp8 ext "
    "(ENABLE_FFPA_TMA_EXT or ENABLE_FFPA_FP8_SM89_EXT)",
  ),
  pytest.mark.skipif(
    not CUDA_QUANT_SPLIT_D_AVAILABLE,
    reason="the whole file exercises the sm89 split-D window (D>224), "
    "which is opt-in; rebuild with ENABLE_FFPA_QUANT_SPLIT_D=1 "
    "(persist-D sm89 coverage: ffpa_attn.bench --backend cuda "
    "--cuda-impl fp8_sm_89 --dtype bf16 --D 128)",
  ),
]

# Split-D window of the sm89 family: 224 < D <= 512, D % 64 == 0.
SPLIT_DIMS = [256, 320, 512]

# Default builds are bf16-input only; add fp16 when the build enables it.
DTYPES = [torch.bfloat16]
if CUDA_INPUT_FP16_AVAILABLE:
  DTYPES.append(torch.float16)

_QUANT_PER_BLOCK = 0
_QUANT_PER_THREAD = 2
_PV_ACC_F16 = 0
_PV_ACC_F32 = 1
_QK_MM_FP8 = 0
_QK_MM_INT8 = 1


def _sm89_backend(**kw) -> CUDABackend:
  """Backend forcing the sm89 fp8 family. Quant knobs are left at the
  CUDABackend production defaults (smooth-K on, per-channel V); the knob
  isolation tests below flip one at a time."""
  base = dict(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    force_fp8_sm89=True,
  )
  base.update(kw)
  return CUDABackend(**base)


def _sm89_out_lse(q, k, v, causal, **kw):
  """Direct op call under the CUTE_FP8_SM_89 hint (returns O + lse)."""
  set_cuda_backend_impl(CudaBackendImpl.CUTE_FP8_SM_89)
  return _ffpa_attn_forward_cuda(
    q,
    k,
    v,
    causal=int(causal),
    softmax_scale=q.size(-1)**-0.5,
    fp8_pv_acc_type=_PV_ACC_F16,
    **kw,
  )


def _rand(B, H, N, D, dtype=torch.bfloat16, seed=0):
  torch.manual_seed(seed)
  return torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5


# Split-D M8N1 parity against the fp32 SDPA reference. Bars are the bf16-
# fixture noise floor measured on this box (fp16 fixtures are tighter):
# the sm89 and sm120 split-D kernels show identical error statistics on
# the same inputs, so these bars document the fp8 family floor, not an
# sm89-specific gap. Dense 5e-2 (D=512 default recipe peaks at ~4.6e-2),
# causal 1e-1 (fp8 early-row quant-error floor).
@pytest.mark.parametrize("D", SPLIT_DIMS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp8_sm89_split_d_forward_parity(D, dtype, causal):
  q, k, v = (_rand(1, 32, 4096, D, dtype) for _ in range(3))

  out = ffpa_attn_func(
    q, k, v, is_causal=causal, forward_backend=_sm89_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  ).to(dtype)
  tol = 1e-1 if causal else 5e-2
  torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


# Determinism: same inputs -> bitwise identical outputs across calls (the
# FC-14 per-path determinism leg of the cross-implementation contract).
@pytest.mark.parametrize("D", SPLIT_DIMS)
def test_fp8_sm89_split_d_deterministic(D):
  q, k, v = (_rand(1, 8, 2048, D) for _ in range(3))

  out1 = ffpa_attn_func(
    q, k, v, is_causal=True, forward_backend=_sm89_backend()
  )
  out2 = ffpa_attn_func(
    q, k, v, is_causal=True, forward_backend=_sm89_backend()
  )
  assert torch.equal(out1, out2)


@pytest.mark.parametrize("D", SPLIT_DIMS)
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_sm89_split_d_qk_int8_parity(D, qk_int8):
  q, k, v = (_rand(1, 32, 2048, D) for _ in range(3))

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=True,
    forward_backend=_sm89_backend(fp8_qk_mm_type="int8" if qk_int8 else "fp8"),
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).to(q.dtype)
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# The four quant-granularity combos (per_block/per_thread QK x
# per_block/per_channel V) the launcher instantiates. This is variant
# coverage, not a precision ranking: per_block V at D=512 is the noisiest
# recipe (one block-wide amax; peaks at ~7.7e-2 with bf16 fixtures), so the
# bar is the coarse fp8-family ceiling. The production default
# (per-channel V) is covered by the parity tests above.
@pytest.mark.parametrize("D", SPLIT_DIMS)
@pytest.mark.parametrize(
  "qk_method", ["per_block", "per_thread"], ids=["qk-block", "qk-thread"]
)
@pytest.mark.parametrize(
  "v_method", ["per_block", "per_channel"], ids=["v-block", "v-channel"]
)
def test_fp8_sm89_split_d_quant_methods_parity(D, qk_method, v_method):
  q, k, v = (_rand(1, 32, 2048, D) for _ in range(3))

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=False,
    forward_backend=_sm89_backend(
      fp8_q_quant_method=qk_method,
      fp8_k_quant_method=qk_method,
      fp8_v_quant_method=v_method,
    ),
  )
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("D", SPLIT_DIMS)
def test_fp8_sm89_split_d_gqa_parity(D):
  q, k, v = (
    _rand(1, 32, 4096, D),
    _rand(1, 8, 4096, D),
    _rand(1, 8, 4096, D),
  )

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_sm89_backend()
  )
  k_rep = k.repeat_interleave(32 // 8, dim=1)
  v_rep = v.repeat_interleave(32 // 8, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)


# Nkv % 128 != 0: tail-tile masking + partial quant blocks; lse checked
# against the fp32 reference (5e-2 bar, same as the sm120 split-D test).
@pytest.mark.parametrize("D", SPLIT_DIMS)
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_sm89_split_d_nkv_unaligned_lse(D, qk_int8):
  N = 2120
  q, k, v = (_rand(1, 32, N, D) for _ in range(3))

  out, lse = _sm89_out_lse(
    q,
    k,
    v,
    causal=True,
    fp8_qk_mm_type=_QK_MM_INT8 if qk_int8 else _QK_MM_FP8,
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  # Early causal rows carry the documented fp8 lse floor (~6e-2 with the
  # e4m3 QK MMA; int8 QK tightens it back to ~5e-2).
  lse_tol = 5e-2 if qk_int8 else 1e-1
  torch.testing.assert_close(lse.float(), lse_ref, atol=lse_tol, rtol=1e-3)


@pytest.mark.parametrize("D", SPLIT_DIMS)
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
@pytest.mark.parametrize(
  "smooth_k", [False, True], ids=["smooth-off", "smooth-on"]
)
def test_fp8_sm89_split_d_smooth_k_and_lse(D, qk_int8, smooth_k):
  q, k, v = (_rand(1, 32, 4096, D) for _ in range(3))

  out, lse = _sm89_out_lse(
    q,
    k,
    v,
    causal=False,
    fp8_smooth_k=smooth_k,
    fp8_qk_mm_type=_QK_MM_INT8 if qk_int8 else _QK_MM_FP8,
  )
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  ref = torch.softmax(p, dim=-1) @ v.float()
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


# Additive attn_mask (mode 0 gmem-direct on the sm89 path).
@pytest.mark.parametrize("D", SPLIT_DIMS)
def test_fp8_sm89_split_d_attn_bias_parity(D):
  B, H, N = 1, 32, 2048
  q, k, v = (_rand(B, H, N, D) for _ in range(3))
  bias = (torch.randn(B, H, N, N, device="cuda") * 0.1).to(q.dtype)

  out = ffpa_attn_func(
    q, k, v, attn_mask=bias, is_causal=False, forward_backend=_sm89_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), attn_mask=bias.float()
  )
  torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)


# Hybrid: causal + fp8 auto-enables the fp16 early rows (sm_80 cp.async
# stage-1) + split-D late rows via q_start_row. The early region sits at
# the fp16-family floor (measured 7.8e-3 here, the FC-15 documented
# 0.0077); the whole output stays inside the fp8 causal bar.
@pytest.mark.parametrize("D", SPLIT_DIMS)
def test_fp8_sm89_split_d_hybrid_causal_parity(D):
  n_early = 512
  N = 2048
  q, k, v = (_rand(1, 32, N, D) for _ in range(3))

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=True,
    forward_backend=_sm89_backend(fp8_hybrid=True, fp8_hybrid_n_early=n_early),
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).to(q.dtype)
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
  early_err = (out[:, :, :n_early].float() - ref[:, :, :n_early].float()).abs()
  assert early_err.max().item(
  ) < 2e-2, (f"hybrid early rows off the fp16 floor: {early_err.max().item()}")


# Nq not a multiple of kBr=128: tail Q tile + partial quant blocks.
@pytest.mark.parametrize("D", SPLIT_DIMS)
def test_fp8_sm89_split_d_nq_unaligned_parity(D):
  N = 2056  # 2056 % 128 == 8
  q, k, v = (_rand(1, 32, N, D) for _ in range(3))

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_sm89_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).to(q.dtype)
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# Scope rejections (loud, per the sm89 v1 contract).
def test_fp8_sm89_split_d_rejects_dropout():
  q, k, v = (_rand(1, 8, 2048, 320) for _ in range(3))
  with pytest.raises(RuntimeError, match="dropout"):
    ffpa_attn_func(
      q,
      k,
      v,
      dropout_p=0.3,
      is_causal=False,
      forward_backend=_sm89_backend(),
    )


def test_fp8_sm89_split_d_rejects_nhd_output():
  # The NHD fast path is inference-only (no-grad); under no_grad the call
  # reaches the sm89 launcher, which rejects NHD O loudly (v1 BHND-only).
  B, N, H, D = 1, 2048, 8, 320
  q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
  k, v = q.clone(), q.clone()
  with torch.no_grad():
    with pytest.raises(RuntimeError, match="BHND"):
      ffpa_attn_func(
        q,
        k,
        v,
        is_causal=False,
        forward_backend=_sm89_backend(tensor_layout="NHD"),
      )


def test_fp8_sm89_split_d_rejects_f32_pv_acc():
  q, k, v = (_rand(1, 8, 2048, 320) for _ in range(3))
  set_cuda_backend_impl(CudaBackendImpl.CUTE_FP8_SM_89)
  with pytest.raises(RuntimeError, match="f16 PV accumulator"):
    _ffpa_attn_forward_cuda(
      q,
      k,
      v,
      causal=0,
      softmax_scale=320**-0.5,
      fp8_pv_acc_type=_PV_ACC_F32,
    )


def test_fp8_sm89_split_d_rejects_d_above_512():
  # D=576 is past the sm89 split-D window (m4n2 is Phase 3, sm120-only).
  # On a build without D=576 the wrapper raises "headdim not support"
  # before dispatch; either way the call must fail loudly.
  q, k, v = (_rand(1, 8, 2048, 576) for _ in range(3))
  with pytest.raises(RuntimeError, match="headdim not support|supports D<=512"):
    ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_sm89_backend())
