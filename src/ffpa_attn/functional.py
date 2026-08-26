"""Autograd Function and metadata for FFPA attention.

Houses the ``FFPAAttnMeta`` dataclass and ``FFPAAttnFunc`` autograd Function
that routes forward/backward across the CUDA, Triton, and aten flash-attention
backends. Imported by ``ffpa_attn_interface.py`` and other callers that need
to access the low-level dispatch layer.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .triton import (
  _ffpa_attn_forward_triton,
  _ffpa_attn_backward_triton,
)  # Large-D by default; small-D when FFPA_TRITON_ALLOW_SMALL_D=1.
from .aten import (
  _flash_attn_forward_aten,
  _flash_attn_backward_aten,
  _efficient_attn_backward_aten,
)  # D <= 256
try:
  from .cute import (
    _ffpa_attn_forward_cute,
    _ffpa_attn_backward_cute,
    _ffpa_attn_varlen_cute,
  )  # Large-D by default; small-D when FFPA_CUTE_ALLOW_SMALL_D=1.
except Exception:
  _ffpa_attn_forward_cute = None
  _ffpa_attn_backward_cute = None
  _ffpa_attn_varlen_cute = None

try:
  from .cuda import _ffpa_attn_forward_cuda, F16_ACC_AVAILABLE  # D > 256
except Exception:
  _ffpa_attn_forward_cuda = None
  F16_ACC_AVAILABLE = False

try:
  # Hoisted to module level: the per-call `from .cuda import ...` inside
  # _apply_cuda_backend_hint costs ~0.5us (sys.modules lookup + attr access).
  from .cuda import set_cuda_backend_impl, CudaBackendImpl
except Exception:
  set_cuda_backend_impl = None
  CudaBackendImpl = None

if TYPE_CHECKING:
  from typing import Tuple, Union, Optional  # noqa: F401

# MMA Acc encoding kept in sync with csrc/pybind/ffpa_attn_api.cc::ffpa_attn.
_ACC_F16 = 0
_ACC_F32 = 1
# FP8 quant granularity encoding (kept in sync with cute/launch.cuh).
_QUANT_METHOD_PER_BLOCK = 0
_QUANT_METHOD_PER_CHANNEL = 1
_QUANT_METHOD_PER_THREAD = 2
_QUANT_METHOD_CODE = {
  "per_block": _QUANT_METHOD_PER_BLOCK,
  "per_channel": _QUANT_METHOD_PER_CHANNEL,
  "per_thread": _QUANT_METHOD_PER_THREAD,
}
# FP8 PV accumulator dtype encoding (kept in sync with cute/launch.cuh).
_PV_ACC_F16 = 0
_PV_ACC_F32 = 1
_PV_ACC_CODE = {"f16": _PV_ACC_F16, "f32": _PV_ACC_F32}
# FP8 QK MMA dtype encoding (kept in sync with cute/launch.cuh).
_QK_MM_FP8 = 0
_QK_MM_INT8 = 1
_QK_MM_TYPE_CODE = {"fp8": _QK_MM_FP8, "int8": _QK_MM_INT8}
_ATEN_SMALL_HEAD_DIM_MAX = 256
_FFPA_SMALL_HEAD_DIM_MIN = 64


def _env_flag_enabled(name: str) -> bool:
  return bool(int(os.environ.get(name, "0")))


def _allow_triton_small_d() -> bool:
  return _env_flag_enabled("FFPA_TRITON_ALLOW_SMALL_D")


def _allow_cute_small_d() -> bool:
  return _env_flag_enabled("FFPA_CUTE_ALLOW_SMALL_D")


def _allow_cuda_small_d() -> bool:
  return _env_flag_enabled("FFPA_CUDA_ALLOW_SMALL_D")


def _backend_allows_small_d(backend: Backend, head_dim: int) -> bool:
  if not (_FFPA_SMALL_HEAD_DIM_MIN <= head_dim <= _ATEN_SMALL_HEAD_DIM_MAX):
    return False
  if isinstance(backend, TritonBackend):
    return _allow_triton_small_d()
  if isinstance(backend, CuTeDSLBackend):
    return _allow_cute_small_d()
  if isinstance(backend, CUDABackend):
    return _allow_cuda_small_d()
  return False


def _should_use_aten_small_d_forward(
  forward_backend: Backend,
  head_dim: int,
) -> bool:
  return head_dim <= _ATEN_SMALL_HEAD_DIM_MAX and not _backend_allows_small_d(
    forward_backend, head_dim
  )


def _is_hopper_or_later() -> bool:
  if not torch.cuda.is_available():
    return False
  major, minor = torch.cuda.get_device_capability()
  return (major, minor) >= (9, 0)


def _is_sm120_or_later() -> bool:
  if not torch.cuda.is_available():
    return False
  major, minor = torch.cuda.get_device_capability()
  return (major, minor) >= (12, 0)


def _cuda_cute_tma_available() -> bool:
  """Whether the CuTe-TMA sm120 forward kernel was compiled and the current
  device can run it. Drives the CUDABackend default impl hint auto-resolve.
  """
  from .cuda import CUDA_CUTE_TMA_AVAILABLE
  return bool(CUDA_CUTE_TMA_AVAILABLE) and _is_sm120_or_later()


def _cuda_tma_available() -> bool:
  """Whether the TMA forward kernel was compiled and the current device can
  run it. Sm120+ auto-resolve fallback below CUTE_TMA.
  """
  from .cuda import CUDA_TMA_AVAILABLE
  return bool(CUDA_TMA_AVAILABLE) and _is_sm120_or_later()


def _apply_cuda_backend_hint(backend: CUDABackend) -> None:
  """Set C++ backend impl hint from CUDABackend flags before kernel launch.

  Mapping: (enable_tma, enable_cute) → hint. No flag set → NATIVE (Legacy).
  """
  if getattr(backend, "enable_fp4", False):
    set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_FP4)
  elif getattr(backend, "enable_fp8", False):
    set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_FP8)
  elif backend.enable_tma and backend.enable_cute:
    set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA)
  elif backend.enable_tma:
    set_cuda_backend_impl(CudaBackendImpl.TMA)
  elif backend.enable_cute:
    set_cuda_backend_impl(CudaBackendImpl.CUTE)
  else:
    set_cuda_backend_impl(CudaBackendImpl.NATIVE)


def _ffpa_attn_forward(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  is_causal: bool,
  scale: float | None,
  enable_gqa: bool,
  fm: CUDABackend,
) -> torch.Tensor | None:
  """Inference-only forward: run the CUDA op directly, skipping meta
  validation and the autograd Function wrapper.

  Currently supports the CUDA backend only; Triton / CuTeDSL inference
  fast paths may be added in the future.

  Returns ``None`` when the config needs the full :func:`ffpa_attn_func`
  chain (grad mode on, small-D SDPA routing, ``D > 1024``, ``Nq < 512``,
  ``Nkv < 512``, bf16 with ``acc='f16'``, or ``fm.tensor_layout='NHD'``
  on a non-persist-D path). The caller is responsible for pre-validated
  packed inputs (BHND fp16/bf16, or NHD when ``fm.tensor_layout='NHD'``).

  :param enable_gqa: Unused by the CUDA kernel (GQA head grouping is
      native); accepted for signature compatibility with future
      Triton / CuTeDSL fast paths.
  """
  nhd = fm.tensor_layout == "NHD"
  if nhd:
    # NHD output packing is implemented by the fp8 persist-D CUDA kernel
    # only (runtime store-layout branch); every other impl/hint combination
    # must take the full chain. The hybrid stage-1 slice is BHND-only, so
    # decline exactly the configs C++ would route into it (fp8_hybrid may
    # still be None; the resolution below maps causal to True).
    if (
      _ffpa_attn_forward_cuda is None or torch.is_grad_enabled()
      or fm.fp8_hadamard or fm.fp4_hybrid or fm.fp4_hadamard
      or not fm.enable_fp8 or (
        fm.enable_fp8 and
        (fm.fp8_hybrid or (fm.fp8_hybrid is None and is_causal))
        and query.size(1) >= (fm.fp8_hybrid_n_early or 256)
      )
    ):
      return None
    nq, nkv = query.size(1), key.size(1)
  else:
    nq, nkv = query.size(2), key.size(2)
  if (
    _ffpa_attn_forward_cuda is None or torch.is_grad_enabled()
    or query.dtype is torch.bfloat16 and fm.acc == "f16"
    or _should_use_aten_small_d_forward(fm, query.size(-1))
    or query.size(-1) > 1024 or 8 <= nq < 512 or nkv < 512
  ):
    return None
  # Same in-place resolution normalize_inputs would perform (idempotent).
  fm.is_causal = is_causal
  if fm.fp8_hybrid is None:
    fm.fp8_hybrid = bool(fm.enable_fp8 and is_causal)
  if fm.fp4_hybrid is None:
    fm.fp4_hybrid = bool(fm.enable_fp4 and is_causal)
  _apply_cuda_backend_hint(fm)
  O, _ = _ffpa_attn_forward_cuda(
    query,
    key,
    value,
    None,
    None,
    fm.stages,
    fm.acc_code,
    int(is_causal),
    1.0 / math.sqrt(query.size(-1)) if scale is None else float(scale),
    0.0,
    0,
    0,
    fm.fp8_smooth_k,
    fm.fp8_smooth_v,
    fm.fp8_q_quant_method_code,
    fm.fp8_k_quant_method_code,
    fm.fp8_v_quant_method_code,
    fm.fp8_pv_acc_code,
    fm.fp8_qk_mm_type_code,
    fm.fp8_hybrid,
    fm.fp8_hybrid_n_early,
    fm.fp4_hybrid,
    fm.fp4_hybrid_n_early,
    fm.fp8_hadamard,
    fm.fp4_hadamard,
    1 if fm.fp4_pv_mm_type == "fp8" else 0,
    fm.fp4_smooth_v,
    0 if nhd else 1,
  )
  return O


def _normalize_grad_kv_storage_dtype(
  dtype: torch.dtype | str | None
) -> torch.dtype | None:
  if dtype is None:
    return None
  if dtype == "fp16":
    return torch.float16
  if dtype == "fp32":
    return torch.float32
  if dtype in (torch.float16, torch.float32):
    return dtype
  raise ValueError(
    "grad_kv_storage_dtype must be None, 'fp16', 'fp32', torch.float16, or torch.float32, "
    f"got {dtype!r}"
  )


@dataclass
class Backend:
  """Base backend configuration.

  :ivar name: Backend identifier (e.g. "triton", "cutedsl").
  :ivar forward: Whether this instance configures the forward pass.  ``None``
      (default) means "not explicitly set"; resolved by :meth:`__post_init__`.
  :ivar backward: Whether this instance configures the backward pass.
      Same ``None`` semantics as *forward*.
  """
  name: str
  forward: bool | None = None
  backward: bool | None = None

  def __post_init__(self) -> None:
    if self.forward is None and self.backward is None:
      self.forward = True
      self.backward = True
    elif self.forward is None:
      self.forward = not self.backward
    elif self.backward is None:
      self.backward = not self.forward


@dataclass
class SDPABackend(Backend):
  """PyTorch native ``scaled_dot_product_attention`` backend.

  Forward always short-circuits via :meth:`FFPAAttnMeta.fallback`.
  When used as ``backward_backend`` it delegates to
  :func:`_efficient_attn_backward_aten`.

  :ivar high_precision_grad: When ``True`` request higher numerical
      precision for the backward pass (passed through to aten).
  """
  name: str = "sdpa"
  high_precision_grad: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()


@dataclass
class CUDABackend(Backend):
  """Hand-written CUDA forward-only backend.

  :ivar acc: MMA accumulator precision (``"f16"`` or ``"f32"``).
  :ivar stages: Pipeline stages for the CUDA kernel (default 4; C++ smem
      physics cap may reduce for large V chunks or TMA path).
  :ivar enable_tma: Select the TMA-based kernel implementation. ``None``
      (default) auto-resolves with ``enable_cute``: when both are ``None`` the
      backend picks ``CUTE_TMA`` if the CuTe-TMA sm120 kernel was compiled and
      the device is sm120+, else ``TMA`` if the TMA kernel was compiled, else
      ``NATIVE``. An explicit ``True``/``False`` opts out of auto. Combined
      with ``enable_cute`` it picks the C++ backend hint: neither -> NATIVE
      (legacy cp.async), tma only -> TMA, cute only -> CUTE (CuTe cp.async),
      both -> CUTE_TMA (CuTe TMA). Ignored on architectures lacking the path.
  :ivar enable_cute: Select the CuTe-based kernel implementation. ``None``
      participates in auto-resolve (see ``enable_tma``); an explicit bool opts
      out. CUDA-backend only.
  :ivar enable_ws: Accepted for API compatibility with the Triton backend;
      the CUDA sm120 path is always warp-specialised when ``enable_tma`` is on.
  :ivar enable_fp8: FP8 persist-D sm120 path (fp16/bf16 inputs, Q/K/V
      quantized inside the kernel).
  :ivar enable_fp4: NVFP4 persist-D sm120 path (any D%8==0 within [8,256],
      pads up to {64,128,192,256}). Mutually exclusive with ``enable_fp8``.
  :ivar fp8_smooth_k: FP8 only: subtract the per-(b,h) K sequence mean
      before quantization.
  :ivar fp8_smooth_v: FP8 only: subtract the per-(b,h) V dim mean.
      Requires ``fp8_v_quant_method='per_channel'``.
  :ivar fp8_q_quant_method: FP8 Q quant granularity, ``"per_block"`` or
      ``"per_thread"``.
  :ivar fp8_k_quant_method: FP8 K quant granularity, ``"per_block"`` or
      ``"per_thread"`` (must match ``fp8_q_quant_method``).
  :ivar fp8_v_quant_method: FP8 V quant granularity, ``"per_block"`` or
      ``"per_channel"``. Defaults to ``"per_channel"``: SageAttention
      (arXiv:2406.12943) shows V exhibits channel-wise outliers that a
      per-block scale cannot contain (one block-wide amax lets a single
      outlier crush the whole block), so it quantizes V per-channel
      ("per-channel quantization can address the channel-wised outlier of
      V"); SageAttention2 (arXiv:2410.21265) keeps V per-channel and adds
      optional smooth-V. Per-block V is kept for exact regressions but is
      not recommended for outlier-heavy activations.
  :ivar fp8_pv_acc_type: FP8 PV accumulator dtype, ``"f32"`` or ``"f16"``.
  :ivar fp8_qk_mm_type: FP8 QK MMA dtype, ``"fp8"`` (e4m3) or ``"int8"``.
  :ivar fp8_hybrid: 2-stage hybrid for the fp8 path — fp16 computes the
      [0:n_early] rows and fp8 the [n_early:N) rows via a q_start_row offset.
      ``None`` (default) is auto: enabled when causal + ``enable_fp8``.
  :ivar fp8_hybrid_n_early: Leading fp16 row count for the fp8 hybrid mode
      (multiple of 128, default 256).
  :ivar fp4_hybrid: Same 2-stage hybrid for the fp4 path; ``None`` is auto
      (causal + ``enable_fp4``).
  :ivar fp4_hybrid_n_early: Leading fp16 row count for the fp4 hybrid mode
      (multiple of 128, default 256).
  :ivar fp8_hadamard: FP8 only: rotate Q/K by an orthogonal Walsh-Hadamard
      matrix before quantization (incoherent processing as in
      FlashAttention-3, arXiv:2407.08608 Sec 3.3; exact in fp32 math).
  :ivar fp4_hadamard: FP4 only: same Walsh-Hadamard Q/K pre-rotation for
      the NVFP4 path (pow2 D rotates inside the quantize kernel).
  :ivar fp4_pv_mm_type: FP4 PV MMA dtype, ``"fp4"`` (NVFP4 e2m1+ue4m3/16)
      or ``"fp8"`` (MXFP8 e4m3+ue8m0/32, QK stays NVFP4; smem budget
      limits fp8 to D<=192).
  :ivar fp4_smooth_v: FP4 only (persist_d, D<=256): subtract the
      per-(b,hkv) V column mean before V quantize; the kernel epilogue adds
      it back (softmax rows sum to 1).
  :ivar tensor_layout: Storage layout of Q/K/V inputs and the returned O:
      ``"HND"`` (default) for BHND packed ``[B, H, N, D]`` tensors, or
      ``"NHD"`` for diffusers-style ``[B, N, H, D]`` packed storage (output
      follows the input layout). NHD is implemented by the persist-D sm120
      CUDA kernels (fp8 / fp16 / fp4) on the inference fast path only.
  :ivar is_causal: Runtime flag propagated from
      ``ffpa_attn_func(is_causal=...)`` by ``normalize_inputs``; drives the
      hybrid auto-resolve. Not a user constructor argument.
  """
  name: str = "cuda"
  acc: str = "f32"
  stages: int = None
  enable_tma: bool | None = None
  enable_cute: bool | None = None
  enable_ws: bool = False  # For future use.
  enable_fp8: bool = False  # FP8 persist-D sm120 path (fp16/bf16 in).
  enable_fp4: bool = False  # NVFP4 persist-D sm120 path (any D%8==0 within [8,256], pads up to {64,128,192,256}).
  fp8_smooth_k: bool = True  # FP8 only: subtract per-(b,h) K seq mean pre-quant.
  fp8_smooth_v: bool = False  # FP8 only: subtract per-(b,h) V dim mean.
  fp8_q_quant_method: str = "per_block"  # FP8 only: per_block / per_thread.
  fp8_k_quant_method: str = "per_block"  # FP8 only: per_block / per_thread.
  # Default per_channel: V has channel-wise outliers; a per_block scale is
  # shared by an entire 128-row tile and a single outlier drives the amax,
  # crushing every other row (SageAttention, arXiv:2406.12943 Sec 4.3).
  fp8_v_quant_method: str = "per_channel"  # FP8 only; per_block / per_channel.
  fp8_pv_acc_type: str = "f32"  # FP8 only; f32/f16 PV accumulator.
  fp8_qk_mm_type: str = "fp8"  # FP8 only: QK MMA dtype; "fp8" or "int8".
  # Hybrid — fp16 computes [0:n_early] rows, the quantized path computes
  # [n_early:N] via q_start_row offset (zero-redundancy). Works for causal
  # (fixes early-row accuracy loss) and non-causal (user-selected rows get
  # full fp16 precision). None=auto: enabled when the matching quant path
  # (enable_fp8 / enable_fp4) + is_causal. fp8_hybrid and fp4_hybrid are
  # independent switches; each is honored only by its own quant path.
  fp8_hybrid: bool | None = None
  fp8_hybrid_n_early: int = 256
  fp4_hybrid: bool | None = None
  fp4_hybrid_n_early: int = 256
  # Incoherent processing (FlashAttention-3, arXiv:2407.08608 Sec 3.3):
  # rotate Q/K by an orthogonal Walsh-Hadamard matrix before quantization
  # to spread per-dim outliers (exact in fp32 math; FA-3's 2.6x lower FP8
  # RMSE is jointly with block quantization). Each switch is honored only
  # by its own quant path.
  fp8_hadamard: bool = False
  fp4_hadamard: bool = False
  # FP4 only: PV MMA dtype; "fp4" (NVFP4 e2m1+ue4m3/16) or "fp8" (MXFP8
  # e4m3+ue8m0/32, QK stays NVFP4). smem budget limits fp8 to D<=192.
  fp4_pv_mm_type: str = "fp4"
  # FP4 only (persist_d, D<=256): subtract the per-(b,hkv) V column mean
  # before V quantize; the epilogue adds it back (softmax rows sum to 1),
  # concentrating the residual so the V blockscale tracks the dynamic range.
  # NOTE: Q/K smoothing is NOT an option in the fp4 path - it is always on
  # and required for accuracy (e2m1's +-6 dynamic range; qm/km means +
  # sub_qm/sub_km quantize + delta_s/lse exact corrections are hardwired in
  # the fp4 kernels). fp4_smooth_v only adds the missing V side.
  fp4_smooth_v: bool = False
  # Storage layout of Q/K/V inputs and the returned O: "HND" (default) for
  # BHND packed [B, H, N, D] tensors, or "NHD" for diffusers-style
  # [B, N, H, D] packed storage (output follows the input layout). NHD is
  # implemented by the persist-D sm120 CUDA kernels only (fp8 / fp16 / fp4)
  # on the inference fast path.
  tensor_layout: str = "HND"
  # Runtime: propagated from ffpa_attn_func(is_causal=...) by normalize_inputs.
  is_causal: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()
    assert not self.backward, "cuda backend does not support backward"
    assert self.acc in (
      "f16", "f32"
    ), f"acc must be 'f16' or 'f32', got {self.acc!r}"
    assert not (self.enable_fp8 and self.enable_fp4
                ), ("enable_fp8 and enable_fp4 are mutually exclusive")
    if self.acc == "f16" and not F16_ACC_AVAILABLE:
      raise ValueError(
        "CUDABackend(acc='f16') requires the fp16 MMA acc kernels, which were "
        "not compiled. Rebuild with ENABLE_FFPA_F16_ACC=1 to enable them."
      )
    assert self.fp8_q_quant_method in ("per_block", "per_thread"), (
      f"fp8_q_quant_method must be 'per_block' or 'per_thread', "
      f"got {self.fp8_q_quant_method!r}"
    )
    assert self.fp8_k_quant_method in ("per_block", "per_thread"), (
      f"fp8_k_quant_method must be 'per_block' or 'per_thread', "
      f"got {self.fp8_k_quant_method!r}"
    )
    assert self.fp8_v_quant_method in ("per_block", "per_channel"), (
      f"fp8_v_quant_method must be 'per_block' or 'per_channel', "
      f"got {self.fp8_v_quant_method!r}"
    )
    assert self.fp8_pv_acc_type in _PV_ACC_CODE, (
      f"fp8_pv_acc_type must be 'f32' or 'f16', got {self.fp8_pv_acc_type!r}"
    )
    assert self.fp8_qk_mm_type in _QK_MM_TYPE_CODE, (
      f"fp8_qk_mm_type must be 'fp8' or 'int8', got {self.fp8_qk_mm_type!r}"
    )
    assert not self.fp8_smooth_v or self.fp8_v_quant_method == "per_channel", (
      "fp8_smooth_v requires fp8_v_quant_method='per_channel'"
    )
    assert not self.fp4_hadamard or self.enable_fp4, (
      "fp4_hadamard requires enable_fp4"
    )
    assert not self.fp8_hadamard or self.enable_fp8, (
      "fp8_hadamard requires enable_fp8"
    )
    assert self.fp4_pv_mm_type in ("fp4", "fp8"), (
      f"fp4_pv_mm_type must be 'fp4' or 'fp8', "
      f"got {self.fp4_pv_mm_type!r}"
    )
    assert self.fp4_pv_mm_type == "fp4" or self.enable_fp4, (
      "fp4_pv_mm_type requires enable_fp4"
    )
    assert not self.fp4_smooth_v or self.enable_fp4, (
      "fp4_smooth_v requires enable_fp4"
    )
    assert self.tensor_layout in ("HND", "NHD"), (
      f"tensor_layout must be 'HND' or 'NHD', got {self.tensor_layout!r}"
    )
    self._resolve_impl_defaults()
    self.stages = self._default_cuda_stages(
    ) if self.stages is None else self.stages

  def _resolve_impl_defaults(self) -> None:
    """Fill ``enable_tma``/``enable_cute`` defaults before kernel launch.

    Both ``None`` -> auto. On sm120+ the priority is ``CUTE_TMA`` (TMA+CUTE
    exts) -> ``TMA`` (TMA ext only) -> ``NATIVE``. A single ``None`` resolves
    to ``False`` so an explicit ``--fwd-tma``/``--cute`` still selects
    ``TMA``/``CUTE``.
    """
    if self.enable_tma is None and self.enable_cute is None:
      if _cuda_cute_tma_available():
        self.enable_tma = True
        self.enable_cute = True
      elif _cuda_tma_available():
        self.enable_tma = True
        self.enable_cute = False
      else:
        self.enable_tma = False
        self.enable_cute = False
    else:
      if self.enable_tma is None:
        self.enable_tma = False
      if self.enable_cute is None:
        self.enable_cute = False

  @property
  def acc_code(self) -> int:
    return _ACC_F32 if self.acc == "f32" else _ACC_F16

  @property
  def fp8_q_quant_method_code(self) -> int:
    return _QUANT_METHOD_CODE[self.fp8_q_quant_method]

  @property
  def fp8_k_quant_method_code(self) -> int:
    return _QUANT_METHOD_CODE[self.fp8_k_quant_method]

  @property
  def fp8_v_quant_method_code(self) -> int:
    return _QUANT_METHOD_CODE[self.fp8_v_quant_method]

  @property
  def fp8_pv_acc_code(self) -> int:
    return _PV_ACC_CODE[self.fp8_pv_acc_type]

  @property
  def fp8_qk_mm_type_code(self) -> int:
    return _QK_MM_TYPE_CODE[self.fp8_qk_mm_type]

  def _default_cuda_stages(self) -> int:
    from .cuda import CudaBackendImpl
    """Default pipeline depth for CUDA backend (non-TMA path)."""
    if _is_hopper_or_later():
      if self.impl_hint in (CudaBackendImpl.NATIVE, CudaBackendImpl.TMA):
        return 4  # sm>=90, native or TMA path
      elif self.impl_hint in (CudaBackendImpl.CUTE, CudaBackendImpl.CUTE_TMA):
        return 2  # sm>=90, CuTe path
    if self.impl_hint == CudaBackendImpl.CUTE:
      return 2  # sm<90, cute path
    return 3  # sm<90, native path

  @property
  def impl_hint(self) -> int:
    from .cuda import CudaBackendImpl
    if self.enable_tma and self.enable_cute:
      return CudaBackendImpl.CUTE_TMA
    if self.enable_tma:
      return CudaBackendImpl.TMA
    if self.enable_cute:
      return CudaBackendImpl.CUTE
    return CudaBackendImpl.NATIVE


@dataclass
class TritonBackend(Backend):
  """Triton forward + backward backend (default).

  :ivar autotune: Enable Triton autotuning for kernel parameters.
  :ivar autotune_mode: Autotune search granularity (``"fast"`` or ``"max"``).
  :ivar enable_tma: Enable experimental SM90+ TMA hardware acceleration.
  :ivar enable_ws: Force warp-specialized configs (requires *enable_tma*).
  :ivar persist_dkdv: Keep ``dK``/``dV`` accumulator in fp32 across
      backward invocations (requires *enable_tma* and ``backward=True``).
  :ivar split_launch: Issue separate backward launches for ``dKdV`` and
      ``dQ`` for finer-grained scheduling.
  :ivar preprocess_d_chunk: Split the ``d_chunk`` preprocess across tiles.
  :ivar grad_kv_storage_dtype: Optional ``torch.float32`` / ``torch.float16``
      storage dtype for ``dK``/``dV``, workaround for causal bf16 precision.
  :ivar grad_q_storage_dtype: Optional ``torch.float32`` / ``torch.float16``
      storage dtype for ``dQ``, workaround for cross-tile atomic-add precision
      when ``USE_DKDVDQ_FUSION`` is enabled (also effective in non-fused path).
  """
  name: str = "triton"
  autotune: bool = False
  autotune_mode: str = "fast"
  enable_tma: bool = False
  enable_ws: bool = False
  persist_dkdv: bool = False
  split_launch: bool = False
  preprocess_d_chunk: bool = False
  grad_kv_storage_dtype: torch.dtype | str | None = None
  grad_q_storage_dtype: torch.dtype | str | None = None

  def __post_init__(self) -> None:
    super().__post_init__()
    assert self.autotune_mode in ("fast", "max"), \
      f"Unsupported autotune_mode={self.autotune_mode!r}; choose 'fast' or 'max'."
    self.grad_kv_storage_dtype = _normalize_grad_kv_storage_dtype(
      self.grad_kv_storage_dtype
    )
    self.grad_q_storage_dtype = _normalize_grad_kv_storage_dtype(
      self.grad_q_storage_dtype
    )
    if self.persist_dkdv:
      assert self.backward, "persist_dkdv is only valid for Triton backward"
      assert self.enable_tma, "persist_dkdv requires enable_tma=True"
    if self.split_launch or self.preprocess_d_chunk or self.grad_kv_storage_dtype is not None or self.grad_q_storage_dtype is not None:
      assert self.backward, "backward-only Triton options require backward=True"


@dataclass
class CuTeDSLBackend(Backend):
  """CuTeDSL SM90-specialized backend (Hopper only, dense 320<D<=512, fp16/bf16 training).

  :ivar grad_kv_storage_dtype: Optional ``torch.float32`` / ``torch.float16``
      storage dtype for the internal SM80 dK/dV HBM buffer; final gradients
      are always cast back to ``k.dtype`` / ``v.dtype``. Workaround for
      causal bf16 cross-tile accumulation precision (mirrors the Triton
      option of the same name). SM90 path currently ignores this knob and
      will raise if it is set.
  """
  name: str = "cutedsl"
  grad_kv_storage_dtype: torch.dtype | str | None = None

  def __post_init__(self) -> None:
    super().__post_init__()
    self.grad_kv_storage_dtype = _normalize_grad_kv_storage_dtype(
      self.grad_kv_storage_dtype
    )
    if self.grad_kv_storage_dtype is not None:
      assert self.backward, \
        "grad_kv_storage_dtype is a backward-only option; requires backward=True"


@dataclass
class AttentionMeta:
  is_causal: bool = False
  scale: float = 0.0
  dropout_p: float = 0.0
  is_grad_enabled: bool = False


def _resolve_backend_pair(
  forward_backend: Backend | None,
  backward_backend: Backend | None,
) -> tuple[Backend, Backend]:
  forward_backend = TritonBackend(
    forward=True
  ) if forward_backend is None else forward_backend
  backward_backend = TritonBackend(
    backward=True
  ) if backward_backend is None else backward_backend

  if not isinstance(forward_backend, Backend):
    raise TypeError("forward_backend must be a Backend object")
  if not isinstance(backward_backend, Backend):
    raise TypeError("backward_backend must be a Backend object")

  assert forward_backend.forward, "forward_backend must be configured with forward=True"
  assert backward_backend.backward, "backward_backend must be configured with backward=True"

  if forward_backend.name == "cutedsl" and backward_backend.name != "cutedsl":
    raise ValueError(
      "forward_backend='cutedsl' requires backward_backend='cutedsl'"
    )
  if backward_backend.name == "cutedsl" and forward_backend.name != "cutedsl":
    raise ValueError(
      "backward_backend='cutedsl' requires forward_backend='cutedsl'"
    )

  return forward_backend, backward_backend


def _coerce_backend(backend: Backend | str, *, source: str) -> Backend:
  if isinstance(backend, str):
    _BACKEND_MAP = {
      "cuda": CUDABackend,
      "triton": TritonBackend,
      "cutedsl": CuTeDSLBackend,
      "sdpa": SDPABackend,
    }
    cls_name = _BACKEND_MAP.get(backend)
    if cls_name is None:
      raise ValueError(
        f"ffpa_attn_func: {source} must be 'cuda', 'triton', 'cutedsl', or 'sdpa', got {backend!r}"
      )
    if source == "backend":
      return cls_name()
    is_forward = source.startswith("forward")
    return cls_name(forward=is_forward, backward=not is_forward)
  if not isinstance(backend, Backend):
    raise TypeError(
      f"ffpa_attn_func: {source} must be a str or Backend instance, got {type(backend).__name__}"
    )
  return backend


def _coerce_optional_backend(
  backend: Backend | str | None,
  *,
  source: str,
) -> Backend | None:
  return None if backend is None else _coerce_backend(backend, source=source)


def _reserve_large_d_dropout_rng(
  q: torch.Tensor,
  k: torch.Tensor,
  dropout_p: float,
) -> torch.Tensor:
  """Reserve SDPA-compatible Philox RNG state for large-D dropout.

  PyTorch efficient attention reserves one random number for every logical
  attention score ``[B, Hq, Nq, Nkv]`` and rounds the CUDA generator offset to
  a multiple of four Philox outputs. The returned CPU int64 tensor stores
  ``[seed, offset]`` for backward recomputation.
  """
  if dropout_p <= 0.0:
    return torch.empty(0, dtype=torch.int64)
  if q.device.type != "cuda":
    raise RuntimeError("ffpa_attn_func: large-D dropout requires CUDA tensors")

  seed = int(torch.cuda.initial_seed())
  offset = int(torch.cuda._get_rng_state_offset())
  attn_elems = q.size(0) * q.size(1) * q.size(2) * k.size(2)
  offset_increment = ((attn_elems + 3) // 4) * 4
  torch.cuda._set_rng_state_offset(offset + offset_increment)
  return torch.tensor([seed, offset], dtype=torch.int64)


def _validate_attn_mask_shape(
  attn_mask: torch.Tensor,
  batch: int,
  nheads_q: int,
  seqlen_q: int,
  seqlen_k: int,
) -> None:
  """Validate SDPA-style attention mask broadcast dimensions.

  :param attn_mask: User-provided attention mask.
  :param batch: Query batch size.
  :param nheads_q: Number of query heads.
  :param seqlen_q: Query sequence length.
  :param seqlen_k: Key/value sequence length.
  :raises ValueError: If ``attn_mask`` is not broadcastable to
    ``[B, Nh_q, Nq, Nkv]`` under SDPA fused-kernel conventions.
  """
  if attn_mask.dim() not in (2, 3, 4):
    raise ValueError(
      "ffpa_attn_func: attn_mask must be 2-D, 3-D, or 4-D and broadcastable "
      "to [B, Nh_q, Nq, Nkv]"
    )
  if attn_mask.size(-2) not in (1, seqlen_q):
    raise ValueError(
      f"ffpa_attn_func: attn_mask query dimension must be 1 or {seqlen_q}, "
      f"got {attn_mask.size(-2)}"
    )
  if attn_mask.size(-1) not in (1, seqlen_k):
    raise ValueError(
      f"ffpa_attn_func: attn_mask key dimension must be 1 or {seqlen_k}, "
      f"got {attn_mask.size(-1)}"
    )
  if attn_mask.dim() == 3 and attn_mask.size(0) not in (1, batch):
    raise ValueError(
      f"ffpa_attn_func: 3-D attn_mask batch dimension must be 1 or {batch}, "
      f"got {attn_mask.size(0)}"
    )
  if attn_mask.dim() == 4:
    if attn_mask.size(0) not in (1, batch):
      raise ValueError(
        f"ffpa_attn_func: 4-D attn_mask batch dimension must be 1 or {batch}, "
        f"got {attn_mask.size(0)}"
      )
    if attn_mask.size(1) not in (1, nheads_q):
      raise ValueError(
        f"ffpa_attn_func: 4-D attn_mask head dimension must be 1 or {nheads_q}, "
        f"got {attn_mask.size(1)}"
      )


@dataclass
class FFPAAttnMeta:
  """Non-tensor FFPA options passed through the autograd Function."""

  attn_meta: AttentionMeta = field(default_factory=AttentionMeta)
  forward_meta: Backend = field(
    default_factory=lambda: TritonBackend(forward=True)
  )
  backward_meta: Backend = field(
    default_factory=lambda: TritonBackend(backward=True)
  )

  def __post_init__(self) -> None:
    self.forward_meta, self.backward_meta = _resolve_backend_pair(
      self.forward_meta, self.backward_meta
    )

  @classmethod
  def from_kwargs(cls, **kwargs) -> FFPAAttnMeta:
    """Create a validated ``FFPAAttnMeta`` from ``ffpa_attn_func`` kwargs.

    Pops ``backend``, ``forward_backend``, and ``backward_backend`` from
    ``kwargs``.  The ``backend`` shorthand (str or ``Backend`` instance)
    auto-fills both ``forward_backend`` and ``backward_backend`` when
    neither is explicitly set.  Priority: explicit ``forward_backend`` /
    ``backward_backend`` > ``backend`` > default Triton.

    Raises ``TypeError`` for any unexpected keyword arguments.
    """
    backend = kwargs.pop("backend", None)
    forward_backend = _coerce_optional_backend(
      kwargs.pop("forward_backend", None), source="forward_backend"
    )
    backward_backend = _coerce_optional_backend(
      kwargs.pop("backward_backend", None), source="backward_backend"
    )

    if kwargs:
      unexpected = ", ".join(sorted(kwargs))
      raise TypeError(
        f"ffpa_attn_func() got unexpected keyword argument(s): {unexpected}"
      )

    if forward_backend is None and backward_backend is None and backend is not None:
      backend = _coerce_backend(backend, source="backend")
      forward_backend = backend
      backward_backend = backend

    if forward_backend is not None and backward_backend is None and forward_backend.name == "cutedsl":
      backward_backend = CuTeDSLBackend()
    if backward_backend is not None and forward_backend is None and backward_backend.name == "cutedsl":
      forward_backend = CuTeDSLBackend()

    forward_backend, backward_backend = _resolve_backend_pair(
      forward_backend, backward_backend
    )
    return cls(
      forward_meta=forward_backend,
      backward_meta=backward_backend,
    )

  @classmethod
  def from_backends(
    cls,
    forward_backend: Backend | None = None,
    backward_backend: Backend | None = None,
  ) -> FFPAAttnMeta:
    forward_backend, backward_backend = _resolve_backend_pair(
      forward_backend, backward_backend
    )
    return cls(
      forward_meta=forward_backend,
      backward_meta=backward_backend,
    )

  @classmethod
  def from_options(
    cls,
    forward_backend: Backend | None = None,
    backward_backend: Backend | None = None,
  ) -> FFPAAttnMeta:
    return cls.from_backends(forward_backend, backward_backend)

  def fallback(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
  ) -> bool:
    """Return whether the public API should delegate to SDPA directly.

    This is a method on ``FFPAAttnMeta`` so callers do not need to
    re-derive the backend name or hardware check outside the meta object.
    """
    assert query.dim() == 4, "Expected query shape [B, Nh_q, Nq, D]"
    assert key.dim() == 4, "Expected key shape [B, Nh_kv, Nkv, D]"
    B, Nh_q, Nq, D = query.shape  # noqa: F841
    _, Nh_kv, Nkv, D_k = key.shape
    assert D == D_k, "Query and key must have the same head dimension"

    # sdpa forward always short-circuits to the native aten path regardless
    # of backward_meta. _FFPAAttnFunc has no general dispatch for sdpa-based
    # forward, and aten flash-attention only handles D<=256.  Letting sdpa
    # forward pass through fallback keeps the public API layer responsible for
    # the full sdpa forward+backward path instead of routing through the
    # incomplete Function dispatch below.
    if self.forward_meta.name == "sdpa":
      return True

    if self.forward_meta.name == "cutedsl":
      from .cute import (
        cute_forward_available,
        cute_max_supported_head_dim,
      )
      cutedsl_hw_unsupported = ((
        D < _FFPA_SMALL_HEAD_DIM_MIN or (
          D <= _ATEN_SMALL_HEAD_DIM_MAX
          and not _backend_allows_small_d(self.forward_meta, D)
        )
      ) or D > cute_max_supported_head_dim(query.device)
                                or not cute_forward_available(query.device))
      return cutedsl_hw_unsupported

    return any([
      _should_use_aten_small_d_forward(self.forward_meta, D),
      D > 1024,
      attn_mask is not None and self.forward_meta.name == "cutedsl",
      dropout_p > 0.0 and self.forward_meta.name == "cutedsl",
      (8 <= Nq < 512),
      Nkv < 512,
    ])

  def normalize_inputs(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
  ) -> FFPAAttnMeta:
    """Fill user-facing fields and validate all inputs in place.

    Call this right after :meth:`from_backends` to get a fully validated meta::

      meta = FFPAAttnMeta.from_backends(forward_backend, backward_backend).normalize(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
        )

    Raises ``TypeError``, ``ValueError``, or ``NotImplementedError`` for
    invalid or unsupported combinations.
    """
    if not 0.0 <= dropout_p <= 1.0:
      raise ValueError(
        f"ffpa_attn_func: dropout_p must be in [0, 1], got {dropout_p}"
      )
    if dropout_p >= 1.0:
      raise ValueError(
        "ffpa_attn_func: dropout_p=1.0 is not supported by SDPA fused kernels"
      )
    if dropout_p > 0.0 and query.size(-1) > 256 and isinstance(
      self.forward_meta, CuTeDSLBackend
    ):
      raise NotImplementedError(
        "ffpa_attn_func: large-D dropout is not supported by forward_backend='cutedsl'"
      )
    if attn_mask is not None and isinstance(self.forward_meta, CuTeDSLBackend):
      raise NotImplementedError(
        "ffpa_attn_func: attn_mask is not supported by forward_backend='cutedsl'. "
        "Use forward_backend='triton' when attn_mask is required."
      )
    if attn_mask is not None and is_causal:
      raise RuntimeError(
        "ffpa_attn_func: explicit attn_mask should not be set when is_causal=True"
      )
    if attn_mask is not None and attn_mask.dtype == torch.bool and attn_mask.requires_grad:
      raise TypeError(
        "ffpa_attn_func: boolean attn_mask cannot require gradients"
      )

    # Fill in user-facing fields.
    self.attn_meta.is_causal = is_causal
    self.attn_meta.dropout_p = float(dropout_p)
    self.attn_meta.is_grad_enabled = torch.is_grad_enabled()

    # Propagate is_causal to the CUDA backend and auto-resolve the hybrid
    # switches. *_hybrid=None (default) means "auto": enable hybrid when
    # causal + the matching quant path (fp8_hybrid<->enable_fp8,
    # fp4_hybrid<->enable_fp4) to protect early-row precision; explicit
    # True/False is honored as-is (fp16 stage-1 + quant stage-2).
    if isinstance(self.forward_meta, CUDABackend):
      self.forward_meta.is_causal = is_causal
      if self.forward_meta.fp8_hybrid is None:
        self.forward_meta.fp8_hybrid = bool(
          self.forward_meta.enable_fp8 and is_causal
        )
      if self.forward_meta.fp4_hybrid is None:
        self.forward_meta.fp4_hybrid = bool(
          self.forward_meta.enable_fp4 and is_causal
        )

    # Validate that acc-code is compatible with activation dtype.
    if isinstance(
      self.forward_meta, CUDABackend
    ) and query.dtype == torch.bfloat16 and self.forward_meta.acc_code == _ACC_F16:
      raise ValueError(
        "bf16 activations require acc='f32'; no bf16-acc mma PTX exists."
      )
    if query.dtype not in (torch.float16, torch.bfloat16):
      raise TypeError(
        f"ffpa_attn_func only supports fp16/bf16, got {query.dtype}"
      )

    # Validate tensor shapes.
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
      raise ValueError("query/key/value must be 4-D [B, H, N, D] tensors")
    if query.size(0) != key.size(0) or query.size(0) != value.size(0):
      raise ValueError("query/key/value must share the same batch size")
    if key.size(1) != value.size(1):
      raise ValueError(
        f"key and value must share the same num_heads, "
        f"got Nh_k={key.size(1)}, Nh_v={value.size(1)}"
      )
    if query.size(1) % key.size(1) != 0:
      raise ValueError(
        f"query num_heads must be an integer multiple of key/value num_heads (GQA/MQA), "
        f"got Nh_q={query.size(1)}, Nh_kv={key.size(1)}"
      )
    if key.size(2) != value.size(2):
      raise ValueError(
        f"key and value must share the same seqlen, got Nk={key.size(2)}, Nv={value.size(2)}"
      )
    if query.size(3) != key.size(3) or query.size(3) != value.size(3):
      raise ValueError("query/key/value must share the same head dim")

    if not enable_gqa and query.size(1) != key.size(1):
      raise ValueError(
        f"enable_gqa=False but query num_heads ({query.size(1)}) != "
        f"key/value num_heads ({key.size(1)}). "
        f"Set enable_gqa=True or use matching head counts."
      )

    if is_causal and key.size(2) < query.size(2):
      raise ValueError(
        f"is_causal=True requires Nkv >= Nq (queries are aligned to the KV tail), "
        f"got Nq={query.size(2)}, Nkv={key.size(2)}"
      )

    if scale is None:
      self.attn_meta.scale = 1.0 / math.sqrt(query.size(-1))
    else:
      self.attn_meta.scale = float(scale)

    return self

  def normalize_attn_mask(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
  ) -> torch.Tensor | None:
    """Convert a user SDPA ``attn_mask`` into an additive FFPA attention bias.

    The returned tensor is a 4-D additive bias that remains compact when the
    user mask broadcasts over batch or head dimensions. Triton wrappers pass
    zero strides for broadcast dimensions instead of materializing an expanded
    ``[B, Nh_q, Nq, Nkv]`` view. Boolean masks follow SDPA semantics: ``True``
    means the element participates in attention and ``False`` maps to ``-inf``
    additive bias.

    :param query: Query tensor with shape ``[B, Nh_q, Nq, D]``.
    :param key: Key tensor with shape ``[B, Nh_kv, Nkv, D]``.
    :param attn_mask: Optional user-provided SDPA attention mask.
    :returns: Additive attention bias or ``None``.
    :raises TypeError: If the mask dtype or device is unsupported.
    :raises ValueError: If the mask shape is not broadcastable to attention scores.
    """
    if attn_mask is None:
      return None

    if attn_mask.device != query.device:
      raise TypeError(
        f"ffpa_attn_func: attn_mask must be on the same device as query, "
        f"got {attn_mask.device} and {query.device}"
      )
    if attn_mask.dtype not in (torch.bool, torch.float32, query.dtype):
      raise TypeError(
        "ffpa_attn_func: attn_mask dtype must be bool, torch.float32, or match query dtype, "
        f"got attn_mask.dtype={attn_mask.dtype} and query.dtype={query.dtype}"
      )

    batch, nheads_q, seqlen_q, _ = query.shape
    seqlen_k = key.size(2)
    _validate_attn_mask_shape(attn_mask, batch, nheads_q, seqlen_q, seqlen_k)

    if attn_mask.dtype == torch.bool:
      neg_inf = torch.tensor(
        float("-inf"), dtype=query.dtype, device=query.device
      )
      attn_bias = torch.where(
        attn_mask, torch.zeros((), dtype=query.dtype, device=query.device),
        neg_inf
      )
    else:
      attn_bias = attn_mask

    if attn_bias.dim() == 2:
      attn_bias = attn_bias.view(1, 1, attn_bias.size(0), attn_bias.size(1))
    elif attn_bias.dim() == 3:
      attn_bias = attn_bias.view(
        attn_bias.size(0), 1, attn_bias.size(1), attn_bias.size(2)
      )

    if attn_bias.stride(-1) != 1:
      attn_bias = attn_bias.contiguous()
    return attn_bias

  def normalize(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
  ) -> tuple[FFPAAttnMeta, torch.Tensor, torch.Tensor, torch.Tensor,
             torch.Tensor | None]:
    """Validate public inputs and return metadata plus autograd inputs.

    :param query: Query tensor passed to the public API.
    :param key: Key tensor passed to the public API.
    :param value: Value tensor passed to the public API.
    :param attn_mask: Optional SDPA-style attention mask.
    :param dropout_p: Dropout probability.
    :param is_causal: Whether causal masking is requested.
    :param scale: Optional softmax scale.
    :param enable_gqa: Whether GQA/MQA semantics are enabled.
    :returns: ``(meta, query, key, value, attn_bias)``. ``meta`` is non-tensor
      dispatch state; the remaining values are passed directly to
      :class:`FFPAAttnFunc` so autograd sees all differentiable inputs.
    """
    self.normalize_inputs(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
    attn_bias = self.normalize_attn_mask(query, key, attn_mask)
    return self, query, key, value, attn_bias


class _FFPAAttnFunc(torch.autograd.Function):
  """FFPA attention with autograd support.

    Forward routes by headdim. ``D <= 256`` uses PyTorch's flash-attention
    forward/backward pair. ``D > 256`` continues to use the FFPA CUDA or
    Triton kernels. When any input requires gradients and grad mode is
    enabled, the intermediate tensors needed by the selected backward path
    are saved on the context.

    Backward is intentionally dispatch-only: backend-specific tensor
    preparation and result restoration live in the backend wrappers under
    ``ffpa_attn.aten`` / ``ffpa_attn.triton`` / ``ffpa_attn.cuda`` rather than
    inside :meth:`backward` itself.

    Large-D dropout stores SDPA-compatible Philox seed/offset metadata
    and recomputes the attention dropout mask in backward.
  """

  @staticmethod
  def forward(
    ctx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor | None,
    meta: FFPAAttnMeta,
  ) -> torch.Tensor:
    is_grad = meta.attn_meta.is_grad_enabled and any(
      x.requires_grad for x in (q, k, v, attn_bias) if x is not None
    )
    head_dim = q.size(-1)
    use_aten_small_d_forward = _should_use_aten_small_d_forward(
      meta.forward_meta, head_dim
    )
    # O is allocated per path: the aten/Triton paths consume a caller-allocated
    # buffer, while the CUDA/CuTeDSL wrappers ignore it (del O) and allocate
    # their own storage.

    if use_aten_small_d_forward:
      O = torch.empty_like(q)  # noqa: E741
      O, lse, rng_state, unused = _flash_attn_forward_aten(
        q,
        k,
        v,
        O,
        meta.attn_meta.is_causal,
        meta.attn_meta.scale,
        meta.attn_meta.dropout_p,
      )
    elif isinstance(meta.forward_meta, CUDABackend):
      forward_meta = meta.forward_meta
      assert _ffpa_attn_forward_cuda is not None, "CUDA backend is not available."
      _apply_cuda_backend_hint(forward_meta)
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.attn_meta.dropout_p)
      O, lse = _ffpa_attn_forward_cuda(
        q,
        k,
        v,
        None,
        attn_bias,
        forward_meta.stages,
        forward_meta.acc_code,
        int(meta.attn_meta.is_causal),
        meta.attn_meta.scale,
        meta.attn_meta.dropout_p,
        int(rng_state[0].item()) if rng_state.numel() else 0,
        int(rng_state[1].item()) if rng_state.numel() else 0,
        forward_meta.fp8_smooth_k,
        forward_meta.fp8_smooth_v,
        forward_meta.fp8_q_quant_method_code,
        forward_meta.fp8_k_quant_method_code,
        forward_meta.fp8_v_quant_method_code,
        forward_meta.fp8_pv_acc_code,
        forward_meta.fp8_qk_mm_type_code,
        forward_meta.fp8_hybrid,
        forward_meta.fp8_hybrid_n_early,
        forward_meta.fp4_hybrid,
        forward_meta.fp4_hybrid_n_early,
        forward_meta.fp8_hadamard,
        forward_meta.fp4_hadamard,
        1 if forward_meta.fp4_pv_mm_type == "fp8" else 0,
        forward_meta.fp4_smooth_v,
      )
    elif isinstance(meta.forward_meta, TritonBackend):
      forward_meta = meta.forward_meta
      assert forward_meta.forward, "forward_meta must be configured with forward=True"
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.attn_meta.dropout_p)
      O = torch.empty_like(q)  # noqa: E741
      O, lse = _ffpa_attn_forward_triton(
        q,
        k,
        v,
        O,
        meta.attn_meta.is_causal,
        meta.attn_meta.scale,
        forward_meta.autotune,
        forward_meta.autotune_mode,
        attn_bias,
        meta.attn_meta.dropout_p,
        int(rng_state[0].item()) if rng_state.numel() else 0,
        int(rng_state[1].item()) if rng_state.numel() else 0,
        forward_meta.enable_tma,
        forward_meta.enable_ws,
      )
    elif isinstance(meta.forward_meta, CuTeDSLBackend):
      # CuTeDSL backend. Layout conversion (B,H,N,D ↔ B,N,H,D) is
      # handled inside _ffpa_attn_forward_cute.
      O, lse = _ffpa_attn_forward_cute(
        q,
        k,
        v,
        softmax_scale=meta.attn_meta.scale,
        causal=meta.attn_meta.is_causal,
        return_lse=True,
      )
      # CuTeDSL does not implement dropout.
      rng_state = torch.empty(0, dtype=torch.uint8, device=q.device)
    else:
      raise ValueError(
        f"Unsupported forward_backend={meta.forward_meta.name!r};"
      )

    # No unused output from the FFPA large-D forward kernels, but the
    # autograd contract requires a consistent number of saved tensors across
    # all backends. The small-D aten path already fills unused above.
    if not use_aten_small_d_forward:
      unused = torch.empty(0, dtype=torch.uint8, device=q.device)

    if is_grad:
      ctx.save_for_backward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        O.contiguous(),
        lse,
        rng_state,
        unused,
      )
      ctx.attn_bias = attn_bias
      ctx.meta = meta

    return O

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    q, k, v, O, lse, rng_state, unused = ctx.saved_tensors
    attn_bias = getattr(ctx, "attn_bias", None)
    meta: FFPAAttnMeta = ctx.meta
    D = q.size(-1)
    use_aten_small_d_forward = _should_use_aten_small_d_forward(
      meta.forward_meta, D
    )

    if not use_aten_small_d_forward:
      if isinstance(meta.backward_meta, TritonBackend):
        backward_meta = meta.backward_meta
        assert backward_meta.backward, "backward_meta must be configured with backward=True"
        dq, dk, dv, grad_attn_bias = _ffpa_attn_backward_triton(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.attn_meta.is_causal,
          softmax_scale=meta.attn_meta.scale,
          autotune=backward_meta.autotune,
          autotune_mode=backward_meta.autotune_mode,
          preprocess_d_chunk=backward_meta.preprocess_d_chunk,
          attn_bias=attn_bias,
          return_attn_bias_grad=ctx.needs_input_grad[3],
          grad_kv_storage_dtype=backward_meta.grad_kv_storage_dtype,
          grad_q_storage_dtype=backward_meta.grad_q_storage_dtype,
          dropout_p=meta.attn_meta.dropout_p,
          philox_seed=int(rng_state[0].item()) if rng_state.numel() else 0,
          philox_offset=int(rng_state[1].item()) if rng_state.numel() else 0,
          enable_tma=backward_meta.enable_tma,
          enable_ws=backward_meta.enable_ws,
          enable_persist_dkdv=backward_meta.persist_dkdv,
          enable_split_launch=backward_meta.split_launch,
        )
      elif isinstance(meta.backward_meta, CuTeDSLBackend):
        # CuTeDSL backward. Layout conversion and kernel dispatch are
        # handled inside _ffpa_attn_backward_cute.
        dq, dk, dv = _ffpa_attn_backward_cute(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          out=O,
          lse=lse,
          softmax_scale=meta.attn_meta.scale,
          causal=meta.attn_meta.is_causal,
          grad_kv_storage_dtype=meta.backward_meta.grad_kv_storage_dtype,
        )
        grad_attn_bias = None  # CuTeDSL does not support attn_mask
      else:
        assert isinstance(meta.backward_meta, SDPABackend), \
          f"Unsupported backward_backend={meta.backward_meta.name!r}"
        dq, dk, dv, grad_attn_bias = _efficient_attn_backward_aten(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.attn_meta.is_causal,
          softmax_scale=meta.attn_meta.scale,
          high_precision_grad=meta.backward_meta.high_precision_grad,
          attn_bias=attn_bias,
          return_attn_bias_grad=ctx.needs_input_grad[3],
          dropout_p=meta.attn_meta.dropout_p,
          philox_seed=int(rng_state[0].item()) if rng_state.numel() else 0,
          philox_offset=int(rng_state[1].item()) if rng_state.numel() else 0,
        )
    else:
      # Aten flash-attention backward for D <= 256, which also supports dropout gradients
      # (currently always 0.0 since dropout is not supported).
      dq, dk, dv = _flash_attn_backward_aten(
        grad_out,
        q,
        k,
        v,
        O,
        lse,
        meta.attn_meta.is_causal,
        rng_state,
        unused,
        meta.attn_meta.scale,
        meta.attn_meta.dropout_p,
      )
      grad_attn_bias = None

    # Gradients for: q, k, v, attn_bias, meta.
    return dq, dk, dv, grad_attn_bias, None


# We cannot use ``torch.library.register_autograd`` on the forward ops
# (``_fwd_cuda`` / ``_fwd_triton``) because each forward backend supports
# *multiple* backward backends selected at runtime via ``backward_backend``:
#
#   forward_backend   │  backward_backend
#   ──────────────────┼───────────────────
#   sdpa              │  (n/a — always short-circuits via meta.fallback())
#   cuda              │  triton, sdpa
#   triton            │  triton, sdpa
#   cutedsl           │  cutedsl, triton, sdpa
#
# ``register_autograd`` binds a forward op to exactly one backward formula.
# Hard-coding one backward (e.g. always Triton) would silently ignore the
# user-requested ``backward_backend`` under ``torch.compile``, breaking the
# sdpa backward path when ``fullgraph=True``.
#
# Instead ``FFPAAttnFunc.apply`` delegates through a module-level function
# guarded by ``torch._dynamo.disable``, which creates a graph break at the
# autograd Function boundary.  The real ``_FFPAAttnFunc.backward`` (with
# full backend dispatch) then runs eagerly.
@torch._dynamo.disable
def _ffpa_apply(*args, **kwargs):
  return _FFPAAttnFunc.apply(*args, **kwargs)


class FFPAAttnFunc:
  """Public-facing autograd Function wrapper.

  ``_FFPAAttnFunc`` holds the real ``forward`` / ``backward``, but its
  auto-generated ``apply`` cannot be directly called under
  ``torch.compile`` — Dynamo would inline it and replace the real backward
  with an auto-generated template that produces zero gradients.  This
  wrapper delegates to :func:`_ffpa_apply`, which is guarded by
  ``torch._dynamo.disable`` so Dynamo leaves the autograd boundary intact.

  Callers that need the real autograd Function (e.g. to inspect
  ``forward`` / ``backward``) can access ``_FFPAAttnFunc`` directly.
  """

  @classmethod
  def apply(cls, *args, **kwargs):
    return _ffpa_apply(*args, **kwargs)


@torch._dynamo.disable
def _ffpa_varlen_apply(
  q,
  k,
  v,
  cu_seqlens_q,
  cu_seqlens_k,
  max_seqlen_q,
  max_seqlen_k,
  dropout_p,
  softmax_scale,
  causal,
  enable_gqa,
  return_lse,
  **kwargs,
):
  return _ffpa_attn_varlen_cute(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
    enable_gqa=enable_gqa,
    return_lse=return_lse,
    kwargs=kwargs,
  )


class FFPAAttnVarlenFunc:
  """Public-facing varlen autograd Function wrapper.

  Follows the same pattern as :class:`FFPAAttnFunc`: delegates through
  :func:`_ffpa_varlen_apply` which is guarded by ``torch._dynamo.disable``
  so ``torch.compile`` leaves the autograd boundary intact.
  """

  @classmethod
  def apply(cls, *args, **kwargs):
    return _ffpa_varlen_apply(*args, **kwargs)


__all__ = [
  "FFPAAttnMeta",
  "FFPAAttnFunc",
  "FFPAAttnVarlenFunc",
]
