"""Additive attn-bias (mask) parity tests across broadcast shapes and dtypes.

Covers the PC-0 smem bias-tile paths (dense [B|1,H|1,Nq,Nkv] TMA tile and
row-broadcast [1,1,1,Nkv]) plus the gmem-direct fallback (column broadcast,
unaligned inner strides), for the fp16 CuTe family (``cute_tma``) across
persist-D / split-D / M4N2 head dims. Reference is fp32 SDPA with the same
mask.

Run: CUDA_VISIBLE_DEVICES=7 FFPA_CUDA_ALLOW_SMALL_D=1 \
  pytest tests/test_ffpa_attn_bias.py -x -q
"""

import math

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from ffpa_attn.functional import CUDABackend

FFPA_CUDA_EXT_BUILT = True
try:
  import ffpa_attn.cuda as _ffpa_cuda  # noqa: F401
except Exception:  # pragma: no cover
  FFPA_CUDA_EXT_BUILT = False

pytestmark = [
  pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
  pytest.mark.skipif(not FFPA_CUDA_EXT_BUILT, reason="ffpa CUDA ext required"),
]

# Nq/Nkv >= 512 keeps the CUDA fast path (short-seq declines below 512).
# D=64 exercises the small-D persist-D launcher guard (dense bias tile
# larger than the Q-persist area must demote to the gmem-direct path);
# needs FFPA_CUDA_ALLOW_SMALL_D=1 (see the Run line above).
SHAPES = [
  (1, 4, 1024, 1024, 64),  # persist-D small D (bias-tile demote guard)
  (1, 4, 1024, 1024, 128),  # persist-D (D<=128)
  (1, 4, 1024, 1024, 320),  # split-D M8N1
  (1, 4, 1024, 1024, 768),  # split-D M4N2
]
TAIL_SHAPES = [(1, 4, 1000, 1536, 128), (1, 4, 1000, 1536, 320)]

# fp16 masks must match the fp16 query dtype (API restriction); fp32 is
# always accepted. bf16 masks require bf16 QKV and are not covered here.
MASK_DTYPES = [torch.float32, torch.float16]


def _make_mask(kind, B, H, Nq, Nkv, dtype, seed=7):
  g = torch.Generator(device="cuda").manual_seed(seed)

  def rnd(*shape):
    return torch.randn(
      *shape, generator=g, device="cuda", dtype=torch.float32
    ) * 0.25

  if kind == "key":
    return rnd(1, 1, 1, Nkv).to(dtype)
  if kind == "query":
    return rnd(1, 1, Nq, 1).to(dtype)
  if kind == "batch-key":
    return rnd(B, 1, 1, Nkv).to(dtype)
  if kind == "head-key":
    return rnd(1, H, 1, Nkv).to(dtype)
  if kind == "dense":
    return rnd(B, H, Nq, Nkv).to(dtype)
  if kind == "batch-dense":
    return rnd(B, 1, Nq, Nkv).to(dtype)
  raise ValueError(kind)


def _ref(Q, K, V, mask):
  # SDPA requires the mask's last dim contiguous; materialize broadcasts.
  B, H, Nq = Q.shape[0], Q.shape[1], Q.shape[2]
  Nkv = K.shape[2]
  m = mask.float().expand(B, H, Nq, Nkv).contiguous()
  return F.scaled_dot_product_attention(
    Q.float(),
    K.float(),
    V.float(),
    attn_mask=m,
    scale=1.0 / math.sqrt(Q.size(-1)),
  )


@pytest.mark.parametrize("mask_dtype", MASK_DTYPES)
@pytest.mark.parametrize(
  "mask_kind",
  ["key", "query", "batch-key", "head-key", "batch-dense", "dense"]
)
@pytest.mark.parametrize("B,H,Nq,Nkv,D", SHAPES + TAIL_SHAPES)
def test_cute_tma_bias_parity(B, H, Nq, Nkv, D, mask_kind, mask_dtype):
  if D >= 256 and D != 320 and D != 768:
    pytest.skip("headdim not in build set")
  torch.manual_seed(0)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, mask_dtype)

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=False,
    enable_tma=True,
    enable_cute=True,
    backward=False
  )
  out = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  ref = _ref(Q, K, V, mask)

  err = (out.float() - ref).abs()
  # fp16 attention vs fp32 SDPA reference: mask adds its own rounding.
  assert err.max().item() < 3e-2, (
    f"cute_tma bias parity failed: kind={mask_kind} dtype={mask_dtype} "
    f"D={D} Nq={Nq} Nkv={Nkv} max={err.max().item():.4f} "
    f"mean={err.mean().item():.5f}"
  )


@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
  "mask_kind",
  ["key", "query", "batch-key", "head-key", "batch-dense", "dense"]
)
@pytest.mark.parametrize("D", [128, 320, 768])
def test_fp8_bias_parity(D, mask_kind, mask_dtype):
  B, H, Nq, Nkv = 1, 4, 1024, 1024
  torch.manual_seed(0)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.float16)
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16)
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16)
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, mask_dtype)

  backend = CUDABackend(forward=True, enable_fp8=True, backward=False)
  out = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  ref = _ref(Q, K, V, mask)

  err = (out.float() - ref).abs()
  assert err.max().item() < 5e-2, (
    f"fp8 bias parity failed: kind={mask_kind} dtype={mask_dtype} D={D} "
    f"max={err.max().item():.4f}"
  )


@pytest.mark.parametrize("mask_kind", ["key", "dense", "query"])
@pytest.mark.parametrize("D", [128, 320])
def test_cute_tma_bias_bf16_parity(D, mask_kind):
  """bf16 masks require bf16 QKV (API dtype rule); covers the bf16 reader."""
  B, H, Nq, Nkv = 1, 4, 1024, 1024
  torch.manual_seed(0)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.bfloat16) * 0.5
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.bfloat16) * 0.5
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.bfloat16) * 0.5
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, torch.bfloat16)

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=False,
    enable_tma=True,
    enable_cute=True,
    backward=False
  )
  out = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  ref = _ref(Q, K, V, mask)

  err = (out.float() - ref).abs()
  assert err.max().item() < 5e-2, (
    f"cute_tma bf16 bias parity failed: kind={mask_kind} D={D} "
    f"max={err.max().item():.4f}"
  )


@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
  "mask_kind",
  ["key", "query", "batch-key", "head-key", "batch-dense", "dense"]
)
@pytest.mark.parametrize("D", [128, 320, 768])
def test_fp4_bias_parity(D, mask_kind, mask_dtype):
  B, H, Nq, Nkv = 1, 4, 1024, 1024
  torch.manual_seed(0)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, mask_dtype)

  backend = CUDABackend(forward=True, enable_fp4=True, backward=False)
  out = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  ref = _ref(Q, K, V, mask)

  err = (out.float() - ref).abs()
  assert err.max().item() < 0.15, (
    f"fp4 bias parity failed: kind={mask_kind} dtype={mask_dtype} D={D} "
    f"max={err.max().item():.4f}"
  )


@pytest.mark.parametrize("mask_kind", ["key", "batch-key", "head-key"])
@pytest.mark.parametrize("D", [320])
def test_fp4_bias_tile_vs_gmem_paths(D, mask_kind):
  """fp4 tile path (mode 2/3) vs gmem-direct (mode 0) on identical values.

  Same values, storage offset +1 element -> the classifier keeps mode 0.
  Row-broadcast kinds, split_d family only (D=320, verified bitwise-stable);
  the fp4 persist_d family carries the FC-4 epilogue race and the fp4 m4n2
  family (D>=768) an interleaved-launch race (see repo memory) -- both make
  strict cross-path assertions flaky there, so they stay under the wide
  parity tolerance above."""
  B, H, Nq, Nkv = 2, 4, 512, 4096
  torch.manual_seed(11)
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, torch.float32)
  mask_gmem = _make_unaligned_view(mask)
  Q = torch.randn(B, H, Nq, D, device="cuda", dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float16) * 0.5

  backend = CUDABackend(forward=True, enable_fp4=True, backward=False)
  out_tile = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  out_gmem = ffpa_attn_func(
    Q, K, V, attn_mask=mask_gmem, forward_backend=backend
  )
  assert torch.equal(out_tile, out_gmem), (
    f"fp4 tile-vs-gmem paths diverged: kind={mask_kind} D={D} "
    f"max={(out_tile.float() - out_gmem.float()).abs().max().item():.4f}"
  )


@pytest.mark.parametrize("D", [320])
def test_bias_int32_overflow_scale(D):
  """H*Nq*Nkv > 2^31 flat elements: the gmem-direct fallback's offset math
    must stay in long long (int32 truncation caused illegal memory access).
    Compares the padded-stride fallback against the contiguous tile path on
    identical mask values."""
  B, H, Nq, Nkv = 1, 32, 8192, 8448
  torch.manual_seed(5)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  buf = torch.randn(B, H, Nq, Nkv + 1, device=dev, dtype=torch.float16) * 0.25
  mask_old = buf[..., :Nkv]
  mask_new = mask_old.clone()
  del buf

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=False,
    enable_tma=True,
    enable_cute=True,
    backward=False
  )
  out_old = ffpa_attn_func(Q, K, V, attn_mask=mask_old, forward_backend=backend)
  out_new = ffpa_attn_func(Q, K, V, attn_mask=mask_new, forward_backend=backend)

  err = (out_old.float() - out_new.float()).abs()
  assert err.max().item() < 1e-3, (
    f"int32-scale fallback vs tile parity failed: D={D} "
    f"max={err.max().item():.4f}"
  )


def _make_unaligned_view(mask):
  """Same values, storage offset +1 element: the base ptr loses its 16B
  alignment, so the classifier keeps the gmem-direct fallback (mode 0).
  A last-dim slice cannot do this -- it moves neither ptr nor stride."""
  n = mask.numel()
  flat = torch.empty(n + 1, device="cuda", dtype=mask.dtype)
  out = flat[1:1 + n].view_as(mask)
  out.copy_(mask)
  assert out.data_ptr() % 16 == mask.element_size()
  return out


@pytest.mark.parametrize(
  "mask_kind", ["key", "batch-key", "head-key", "dense", "batch-dense"]
)
@pytest.mark.parametrize(
  "B,H,Nq,Nkv,D", [(2, 4, 512, 4096, 128), (2, 4, 512, 4096, 320),
                   (1, 4, 1024, 1024, 768)]
)
def test_bias_tile_vs_gmem_paths(B, H, Nq, Nkv, D, mask_kind):
  """Tile path (mode 2/3) vs gmem-direct (mode 0) on identical values.

  The two runs differ ONLY in the bias storage address, so outputs must be
  bitwise equal; this pins the (b,h) row fold of the row-broadcast plane
  (head-key/batch-key) that the 3e-2 SDPA tolerance cannot catch."""
  if D >= 256 and D != 320 and D != 768:
    pytest.skip("headdim not in build set")
  torch.manual_seed(11)
  mask = _make_mask(mask_kind, B, H, Nq, Nkv, torch.float32)
  mask_gmem = _make_unaligned_view(mask)
  Q = torch.randn(B, H, Nq, D, device="cuda", dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float16) * 0.5

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=False,
    enable_tma=True,
    enable_cute=True,
    backward=False
  )
  out_tile = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  out_gmem = ffpa_attn_func(
    Q, K, V, attn_mask=mask_gmem, forward_backend=backend
  )
  assert torch.equal(out_tile, out_gmem), (
    f"tile-vs-gmem paths diverged: kind={mask_kind} D={D} "
    f"max={(out_tile.float() - out_gmem.float()).abs().max().item():.4f}"
  )


@pytest.mark.parametrize("D", [128, 320])
def test_causal_plus_bias_parity(D):
  """Bias must compose with -inf masked positions.

    ``ffpa_attn_func`` rejects ``is_causal`` together with ``attn_mask``, so
    the causal composition is expressed as a dense additive mask: key-broadcast
    bias with fp16-min on the strict upper triangle (the kernel's -INFINITY
    override semantics must match SDPA's masked-out positions).
    """
  B, H, Nq, Nkv = 1, 4, 1024, 1024
  torch.manual_seed(3)
  dev = "cuda"
  Q = torch.randn(B, H, Nq, D, device=dev, dtype=torch.float16) * 0.5
  K = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  V = torch.randn(B, H, Nkv, D, device=dev, dtype=torch.float16) * 0.5
  g = torch.Generator(device=dev).manual_seed(11)
  key_bias = torch.randn(
    1, 1, 1, Nkv, generator=g, device=dev, dtype=torch.float32
  ) * 0.25
  m = key_bias.expand(1, 1, Nq, Nkv).clone()
  iu = torch.triu(torch.ones(Nq, Nkv, dtype=torch.bool, device=dev), diagonal=1)
  m[0, 0][iu] = torch.finfo(torch.float16).min
  mask = m.to(torch.float16)

  backend = CUDABackend(
    forward=True,
    enable_fp8=False,
    enable_fp4=False,
    enable_tma=True,
    enable_cute=True,
    backward=False
  )
  out = ffpa_attn_func(Q, K, V, attn_mask=mask, forward_backend=backend)
  ref = _ref(Q, K, V, mask)

  err = (out.float() - ref).abs()
  assert err.max().item(
  ) < 3e-2, (f"causal+bias parity failed: D={D} max={err.max().item():.4f}")
