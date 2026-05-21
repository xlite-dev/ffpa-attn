"""Autograd Function and metadata for FFPA attention.

Houses the ``FFPAAttnMeta`` dataclass and ``FFPAAttnFunc`` autograd Function
that routes forward/backward across the CUDA, Triton, and aten flash-attention
backends. Imported by ``ffpa_attn_interface.py`` and other callers that need
to access the low-level dispatch layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .triton import (
  _ffpa_attn_forward_triton,
  _ffpa_attn_backward_triton,
)  # D > 256
from .aten import (
  _aten_flash_attn_forward,
  _aten_flash_attn_backward,
  _aten_efficient_attn_backward,
)  # D <= 256
from .cutedsl import (
  _ffpa_attn_forward_cutedsl,
  _ffpa_attn_backward_cutedsl,
  _ffpa_attn_varlen_cutedsl,
)  # D == 512 SM90

if TYPE_CHECKING:
  from typing import Tuple, Union, Optional  # noqa: F401

# MMA Acc encoding kept in sync with csrc/pybind/ffpa_attn_api.cc::ffpa_attn.
_ACC_F16 = 0
_ACC_F32 = 1


def _is_hopper_or_later() -> bool:
  if not torch.cuda.is_available():
    return False
  major, minor = torch.cuda.get_device_capability()
  return (major, minor) >= (9, 0)


def _normalize_grad_kv_storage_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
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
  name: str = "sdpa"
  high_precision_grad: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()


@dataclass
class CUDABackend(Backend):
  name: str = "cuda"
  acc: str = "f32"
  stages: int = 4 if _is_hopper_or_later() else 3

  def __post_init__(self) -> None:
    super().__post_init__()
    assert not self.backward, "cuda backend does not support backward"
    assert self.acc in ("f16", "f32"), f"acc must be 'f16' or 'f32', got {self.acc!r}"

  @property
  def acc_code(self) -> int:
    return _ACC_F32 if self.acc == "f32" else _ACC_F16


@dataclass
class TritonBackend(Backend):
  name: str = "triton"
  autotune: bool = False
  autotune_mode: str = "fast"
  enable_tma: bool = False
  enable_ws: bool = False
  persist_dkdv: bool = False
  split_launch: bool = False
  preprocess_d_chunk: bool = False
  grad_kv_storage_dtype: torch.dtype | str | None = None

  def __post_init__(self) -> None:
    super().__post_init__()
    assert self.autotune_mode in ("fast", "max"), \
      f"Unsupported autotune_mode={self.autotune_mode!r}; choose 'fast' or 'max'."
    self.grad_kv_storage_dtype = _normalize_grad_kv_storage_dtype(self.grad_kv_storage_dtype)
    if self.persist_dkdv:
      assert self.backward, "persist_dkdv is only valid for Triton backward"
      assert self.enable_tma, "persist_dkdv requires enable_tma=True"
    if self.split_launch or self.preprocess_d_chunk or self.grad_kv_storage_dtype is not None:
      assert self.backward, "backward-only Triton options require backward=True"


@dataclass
class CuTeDSLBackend(Backend):
  name: str = "cutedsl"


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
  forward_backend = TritonBackend(forward=True) if forward_backend is None else forward_backend
  backward_backend = TritonBackend(backward=True) if backward_backend is None else backward_backend

  if not isinstance(forward_backend, Backend):
    raise TypeError("forward_backend must be a Backend object")
  if not isinstance(backward_backend, Backend):
    raise TypeError("backward_backend must be a Backend object")

  assert forward_backend.forward, "forward_backend must be configured with forward=True"
  assert backward_backend.backward, "backward_backend must be configured with backward=True"

  return forward_backend, backward_backend


_CUDA_BACKEND_LOADED = False
_CUDA_BACKEND_IMPORT_ERROR: Exception | None = None
_CUDA_FWD_AVAILABLE = False
_CUDA_FORWARD_IMPL = None


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
    raise ValueError("ffpa_attn_func: attn_mask must be 2-D, 3-D, or 4-D and broadcastable "
                     "to [B, Nh_q, Nq, Nkv]")
  if attn_mask.size(-2) not in (1, seqlen_q):
    raise ValueError(
      f"ffpa_attn_func: attn_mask query dimension must be 1 or {seqlen_q}, "
      f"got {attn_mask.size(-2)}"
    )
  if attn_mask.size(-1) not in (1, seqlen_k):
    raise ValueError(f"ffpa_attn_func: attn_mask key dimension must be 1 or {seqlen_k}, "
                     f"got {attn_mask.size(-1)}")
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


def _load_cuda_backend() -> None:
  global _CUDA_BACKEND_LOADED
  global _CUDA_BACKEND_IMPORT_ERROR
  global _CUDA_FWD_AVAILABLE
  global _CUDA_FORWARD_IMPL

  if _CUDA_BACKEND_LOADED:
    return

  _CUDA_BACKEND_LOADED = True
  try:
    from . import cuda as cuda_backend
  except Exception as exc:
    _CUDA_BACKEND_IMPORT_ERROR = exc
    return

  _CUDA_FORWARD_IMPL = cuda_backend._ffpa_attn_forward_cuda
  _CUDA_FWD_AVAILABLE = bool(getattr(cuda_backend, "CUDA_FWD_AVAILABLE", False))


def cuda_forward_available() -> bool:
  _load_cuda_backend()
  return _CUDA_FWD_AVAILABLE


def cuda_backward_available() -> bool:
  return False


def _require_cuda_forward_impl():
  _load_cuda_backend()
  if _CUDA_FWD_AVAILABLE and _CUDA_FORWARD_IMPL is not None:
    return _CUDA_FORWARD_IMPL

  message = (
    "ffpa_attn_func: forward_backend='cuda' requested but the CUDA forward backend is unavailable. "
    "Rebuild with ENABLE_FFPA_CUDA_IMPL=1 to enable it."
  )
  if _CUDA_BACKEND_IMPORT_ERROR is not None:
    message = f"{message} Original import error: {_CUDA_BACKEND_IMPORT_ERROR}"
  raise RuntimeError(message)


def _require_cuda_backward_impl():
  raise RuntimeError(
    "ffpa_attn_func: backward_backend='cuda' has been removed from the active backend. "
    "Use backward_backend='triton' or backward_backend='sdpa'."
  )


@dataclass
class FFPAAttnMeta:
  """Non-tensor FFPA options passed through the autograd Function."""

  attn_meta: AttentionMeta = field(default_factory=AttentionMeta)
  forward_meta: Backend = field(default_factory=lambda: TritonBackend(forward=True))
  backward_meta: Backend = field(default_factory=lambda: TritonBackend(backward=True))

  def __post_init__(self) -> None:
    self.forward_meta, self.backward_meta = _resolve_backend_pair(self.forward_meta, self.backward_meta)

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
    forward_backend = kwargs.pop("forward_backend", None)
    backward_backend = kwargs.pop("backward_backend", None)

    if kwargs:
      unexpected = ", ".join(sorted(kwargs))
      raise TypeError(f"ffpa_attn_func() got unexpected keyword argument(s): {unexpected}")

    if forward_backend is None and backward_backend is None and backend is not None:
      if isinstance(backend, str):
        _BACKEND_MAP = {"cuda": CUDABackend, "triton": TritonBackend, "cutedsl": CuTeDSLBackend, "sdpa": SDPABackend}
        cls_name = _BACKEND_MAP.get(backend)
        if cls_name is None:
          raise ValueError(f"ffpa_attn_func: backend must be 'cuda', 'triton', or 'cutedsl', got {backend!r}")
        backend = cls_name()
      if not isinstance(backend, Backend):
        raise TypeError(f"ffpa_attn_func: backend must be a str or Backend instance, got {type(backend).__name__}")
      forward_backend = backend
      backward_backend = backend

    forward_backend, backward_backend = _resolve_backend_pair(forward_backend, backward_backend)
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
    forward_backend, backward_backend = _resolve_backend_pair(forward_backend, backward_backend)
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
      from .cutedsl import cutedsl_forward_available
      cutedsl_hw_unsupported = D != 512 or not cutedsl_forward_available(query.device)
      return cutedsl_hw_unsupported

    return any([
      D <= 256,
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
      raise ValueError(f"ffpa_attn_func: dropout_p must be in [0, 1], got {dropout_p}")
    if dropout_p >= 1.0:
      raise ValueError("ffpa_attn_func: dropout_p=1.0 is not supported by SDPA fused kernels")
    if dropout_p > 0.0 and query.size(-1) > 256 and isinstance(self.forward_meta, CuTeDSLBackend):
      raise NotImplementedError("ffpa_attn_func: large-D dropout is not supported by forward_backend='cutedsl'")
    if attn_mask is not None and isinstance(self.forward_meta, CuTeDSLBackend):
      raise NotImplementedError(
        "ffpa_attn_func: attn_mask is not supported by forward_backend='cutedsl'. "
        "Use forward_backend='triton' when attn_mask is required."
      )
    if attn_mask is not None and is_causal:
      raise RuntimeError("ffpa_attn_func: explicit attn_mask should not be set when is_causal=True")
    if attn_mask is not None and attn_mask.dtype == torch.bool and attn_mask.requires_grad:
      raise TypeError("ffpa_attn_func: boolean attn_mask cannot require gradients")

    # Fill in user-facing fields.
    self.attn_meta.is_causal = is_causal
    self.attn_meta.dropout_p = float(dropout_p)
    self.attn_meta.is_grad_enabled = torch.is_grad_enabled()

    # Validate that acc-code is compatible with activation dtype.
    if isinstance(
      self.forward_meta, CUDABackend
    ) and query.dtype == torch.bfloat16 and self.forward_meta.acc_code == _ACC_F16:
      raise ValueError("bf16 activations require acc='f32'; no bf16-acc mma PTX exists.")
    if query.dtype not in (torch.float16, torch.bfloat16):
      raise TypeError(f"ffpa_attn_func only supports fp16/bf16, got {query.dtype}")

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
      raise ValueError(f"key and value must share the same seqlen, got Nk={key.size(2)}, Nv={value.size(2)}")
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
      neg_inf = torch.tensor(float("-inf"), dtype=query.dtype, device=query.device)
      attn_bias = torch.where(attn_mask, torch.zeros((), dtype=query.dtype, device=query.device), neg_inf)
    else:
      attn_bias = attn_mask

    if attn_bias.dim() == 2:
      attn_bias = attn_bias.view(1, 1, attn_bias.size(0), attn_bias.size(1))
    elif attn_bias.dim() == 3:
      attn_bias = attn_bias.view(attn_bias.size(0), 1, attn_bias.size(1), attn_bias.size(2))

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
  ) -> tuple[FFPAAttnMeta, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    self.normalize_inputs(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
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
    is_grad = meta.attn_meta.is_grad_enabled and any(x.requires_grad for x in (q, k, v, attn_bias) if x is not None)
    head_dim = q.size(-1)
    O = torch.empty_like(q)  # noqa: E741

    if head_dim <= 256:
      O, lse, rng_state, unused = _aten_flash_attn_forward(
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
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.attn_meta.dropout_p)
      cuda_forward_impl = _require_cuda_forward_impl()
      O, lse = cuda_forward_impl(
        q,
        k,
        v,
        O,
        attn_bias,
        forward_meta.stages,
        forward_meta.acc_code,
        int(meta.attn_meta.is_causal),
        meta.attn_meta.scale,
        meta.attn_meta.dropout_p,
        int(rng_state[0].item()) if rng_state.numel() else 0,
        int(rng_state[1].item()) if rng_state.numel() else 0,
        0,
      )
    elif isinstance(meta.forward_meta, TritonBackend):
      forward_meta = meta.forward_meta
      assert forward_meta.forward, "forward_meta must be configured with forward=True"
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.attn_meta.dropout_p)
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
      # handled inside _ffpa_attn_forward_cutedsl.
      O, lse = _ffpa_attn_forward_cutedsl(
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
      raise ValueError(f"Unsupported forward_backend={meta.forward_meta.name!r};")

    # NO unused output from the FFPA CUDA forward / backward kernels, but we
    # need to return something to keep the autograd contract
    # consistent across backends. Return empty tensors on large-D paths since the
    # small-D path's backward expects tensors to be returned and saved.
    if head_dim > 256 and not isinstance(meta.forward_meta, TritonBackend):
      unused = torch.empty(0, dtype=torch.uint8, device=q.device)
    elif head_dim > 256:
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

    if D > 256:
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
        # handled inside _ffpa_attn_backward_cutedsl.
        dq, dk, dv = _ffpa_attn_backward_cutedsl(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          out=O,
          lse=lse,
          softmax_scale=meta.attn_meta.scale,
          causal=meta.attn_meta.is_causal,
        )
        grad_attn_bias = None  # CuTeDSL does not support attn_mask
      else:
        assert isinstance(meta.backward_meta, SDPABackend), \
          f"Unsupported backward_backend={meta.backward_meta.name!r}"
        dq, dk, dv, grad_attn_bias = _aten_efficient_attn_backward(
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
      dq, dk, dv = _aten_flash_attn_backward(
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
  return _ffpa_attn_varlen_cutedsl(
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
  "cuda_forward_available",
  "cuda_backward_available",
]
