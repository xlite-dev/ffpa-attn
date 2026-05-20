"""Autograd Function and metadata for FFPA attention.

Houses the ``FFPAAttnMeta`` dataclass and ``FFPAAttnFunc`` autograd Function
that routes forward/backward across the CUDA, Triton, and aten flash-attention
backends. Imported by ``ffpa_attn_interface.py`` and other callers that need
to access the low-level dispatch layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


_FFPA_ATTN_IMPL_DEFAULTS: dict[str, object] = {
  "stages": 4 if _is_hopper_or_later() else 3,
  "acc": "f32",
  "enable_tma": False,
  "enable_ws": False,
  "enable_forward_tma": False,
  "enable_backward_tma": False,
  "enable_forward_ws": False,
  "enable_backward_ws": False,
  "triton_backward_enable_persist_dkdv": False,
  "triton_backward_enable_split_launch": False,
  "high_precision_grad": False,
  "forward_backend": "triton",
  "triton_autotune": False,
  "triton_autotune_mode": "fast",
  "backward_backend": "triton",
  "triton_backward_preprocess_d_chunk": False,
  "triton_backward_grad_kv_storage_dtype": None,
}

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


def _resolve_directional_flag(
  kwargs: dict[str, object],
  impl_options: dict[str, object],
  legacy_key: str,
  forward_key: str,
  backward_key: str,
) -> tuple[int, int]:
  """Resolve a legacy global bool into direction-specific bools."""
  legacy_present = legacy_key in kwargs
  forward_present = forward_key in kwargs
  backward_present = backward_key in kwargs
  legacy_value = bool(impl_options[legacy_key])
  forward_value = bool(impl_options[forward_key]) if forward_present else legacy_value
  backward_value = bool(impl_options[backward_key]) if backward_present else legacy_value
  if legacy_present:
    conflicts = []
    if forward_present and forward_value != legacy_value:
      conflicts.append(forward_key)
    if backward_present and backward_value != legacy_value:
      conflicts.append(backward_key)
    if conflicts:
      names = ", ".join(conflicts)
      raise ValueError(f"{legacy_key} conflicts with direction-specific option(s): {names}")
  return int(forward_value), int(backward_value)


@dataclass
class FFPAAttnMeta:
  """Non-tensor FFPA options passed through the autograd Function.

  :param is_causal: Whether to apply lower-right causal masking.
  :param scale: Scale applied to ``QK^T``.
  :param stages: CUDA forward pipeline stages.
  :param acc: Native CUDA accumulator code.
  :param enable_forward_tma: Experimental SM90+ Triton forward path
    (descriptor/TMA). Falls back silently when unsupported. Defaults to
    ``False``.
  :param enable_backward_tma: Experimental SM90+ Triton backward path
    (descriptor/TMA). Falls back silently when unsupported. Defaults to
    ``False``.
  :param enable_forward_ws: Force warp-specialized SM90 TMA forward configs.
    Only effective with ``enable_forward_tma=True``. Defaults to ``False``.
  :param enable_backward_ws: Force warp-specialized SM90 TMA backward configs.
    Only effective with ``enable_backward_tma=True``. Defaults to ``False``.
  :param triton_backward_enable_persist_dkdv: Use the experimental SM90 TMA
    backward dK/dV path that keeps fp32 dK/dV accumulators in registers across
    Q blocks. Requires ``enable_backward_tma=True``. Defaults to ``False``.
  :param triton_backward_enable_split_launch: Use separate SM90 TMA backward
    launches for dK/dV and dQ. Requires ``enable_backward_tma=True``. Defaults
    to ``False``.
  :param enable_tma: Compatibility alias for setting both
    ``enable_forward_tma`` and ``enable_backward_tma``.
  :param enable_ws: Compatibility alias for setting both
    ``enable_forward_ws`` and ``enable_backward_ws``.
  :param dropout_p: Dropout probability (default 0.0).
  :param is_grad_enabled: Grad-mode state captured at the public API.
  :param high_precision_grad: Whether SDPA backward should upcast.
  :param forward_backend: Forward backend name, ``"cuda"`` or ``"triton"``.
  :param triton_autotune: Whether to enable Triton runtime autotune.
  :param triton_autotune_mode: Triton autotune search-space mode,
    ``"fast"`` or ``"max"``.
  :param backward_backend: Backward backend name. ``"sdpa"`` or ``"triton"``.
  :param triton_backward_preprocess_d_chunk: Whether Triton backward should
    compute delta with the split-D preprocess kernel.
  :param triton_backward_grad_kv_storage_dtype: Optional storage dtype for
    Triton backward ``DK`` / ``DV`` buffers. ``None`` keeps k/v dtype;
    currently ``torch.float16`` and ``torch.float32`` are accepted as overrides.
  """

  is_causal: bool
  scale: float
  stages: int
  acc: int
  enable_forward_tma: int
  enable_backward_tma: int
  enable_forward_ws: int
  enable_backward_ws: int
  triton_backward_enable_persist_dkdv: bool
  triton_backward_enable_split_launch: bool
  dropout_p: float
  is_grad_enabled: bool
  high_precision_grad: bool
  forward_backend: str
  triton_autotune: bool
  triton_autotune_mode: str
  backward_backend: str
  triton_backward_preprocess_d_chunk: bool
  triton_backward_grad_kv_storage_dtype: torch.dtype | None

  @classmethod
  def from_kwargs(cls, **kwargs: object) -> FFPAAttnMeta:
    """Build a meta from the impl-specific ``**kwargs``.

    Merges the given kwargs with :data:`_FFPA_ATTN_IMPL_DEFAULTS`, validates
    unknown keys, backends and ``acc``, and returns a fully populated meta
    instance.  User-facing fields (``is_causal``, ``scale``, ``dropout_p``,
    ``is_grad_enabled``) are set to safe defaults; call :meth:`normalize`
    to fill and validate them from the public-API inputs.
    """
    unknown = sorted(set(kwargs) - set(_FFPA_ATTN_IMPL_DEFAULTS))
    if unknown:
      keys = ", ".join(unknown)
      raise TypeError(f"ffpa_attn_func got unexpected keyword argument(s): {keys}")

    impl_options = {**_FFPA_ATTN_IMPL_DEFAULTS, **kwargs}

    stages = int(impl_options["stages"])
    acc_str = impl_options["acc"]
    enable_forward_tma, enable_backward_tma = _resolve_directional_flag(
      kwargs,
      impl_options,
      "enable_tma",
      "enable_forward_tma",
      "enable_backward_tma",
    )
    enable_forward_ws, enable_backward_ws = _resolve_directional_flag(
      kwargs,
      impl_options,
      "enable_ws",
      "enable_forward_ws",
      "enable_backward_ws",
    )
    high_precision_grad = bool(impl_options["high_precision_grad"])
    forward_backend = str(impl_options["forward_backend"])
    triton_autotune = bool(impl_options["triton_autotune"])
    triton_autotune_mode = str(impl_options["triton_autotune_mode"])
    backward_backend = str(impl_options["backward_backend"])
    triton_backward_preprocess_d_chunk = bool(impl_options["triton_backward_preprocess_d_chunk"])
    triton_backward_grad_kv_storage_dtype = impl_options["triton_backward_grad_kv_storage_dtype"]
    triton_backward_enable_persist_dkdv = bool(impl_options["triton_backward_enable_persist_dkdv"])
    triton_backward_enable_split_launch = bool(impl_options["triton_backward_enable_split_launch"])

    assert forward_backend in ("cuda", "triton", "cutedsl"), \
      f"Unsupported forward_backend={forward_backend!r}; choose 'cuda', 'triton', or 'cutedsl'."
    assert backward_backend in ("sdpa", "triton", "cutedsl"), \
      f"Unsupported backward_backend={backward_backend!r}; choose 'sdpa', 'triton', or 'cutedsl'."

    # cutedsl forward/backward are bound as a pair: switching one implicitly
    # selects the other, and any cross-backend combination is rejected.
    backward_backend_explicit = "backward_backend" in kwargs
    if forward_backend == "cutedsl" and backward_backend != "cutedsl":
      if backward_backend_explicit:
        raise ValueError(
          f"forward_backend='cutedsl' requires backward_backend='cutedsl'; "
          f"got backward_backend={backward_backend!r}"
        )
      backward_backend = "cutedsl"
    elif backward_backend == "cutedsl" and forward_backend != "cutedsl":
      raise ValueError(
        f"backward_backend='cutedsl' requires forward_backend='cutedsl'; "
        f"got forward_backend={forward_backend!r}"
      )
    assert triton_autotune_mode in ("fast", "max"), \
      f"Unsupported triton_autotune_mode={triton_autotune_mode!r}; choose 'fast' or 'max'."
    if triton_backward_grad_kv_storage_dtype not in (None, torch.float16, torch.float32):
      raise ValueError(
        "triton_backward_grad_kv_storage_dtype must be None, torch.float16, or torch.float32, "
        f"got {triton_backward_grad_kv_storage_dtype!r}"
      )
    if triton_backward_enable_persist_dkdv and not enable_backward_tma:
      raise ValueError("triton_backward_enable_persist_dkdv requires enable_backward_tma=True")
    if triton_backward_enable_split_launch and not enable_backward_tma:
      raise ValueError("triton_backward_enable_split_launch requires enable_backward_tma=True")

    if acc_str == "f32":
      acc = _ACC_F32
    elif acc_str == "f16":
      acc = _ACC_F16
    else:
      raise ValueError(f"acc must be 'f16' or 'f32', got {acc_str!r}")

    return cls(
      # NOTE: Some of these fields maybe updated later by normalize() based on
      # the public API inputs, but we need to set them to some value here to create
      # the instance.
      is_causal=False,
      scale=0.0,
      dropout_p=0.0,
      is_grad_enabled=False,
      acc=acc,
      stages=stages,
      enable_forward_tma=enable_forward_tma,
      enable_backward_tma=enable_backward_tma,
      enable_forward_ws=enable_forward_ws,
      enable_backward_ws=enable_backward_ws,
      triton_backward_enable_persist_dkdv=triton_backward_enable_persist_dkdv,
      triton_backward_enable_split_launch=triton_backward_enable_split_launch,
      high_precision_grad=high_precision_grad,
      forward_backend=forward_backend,
      triton_autotune=triton_autotune,
      triton_autotune_mode=triton_autotune_mode,
      backward_backend=backward_backend,
      triton_backward_preprocess_d_chunk=triton_backward_preprocess_d_chunk,
      triton_backward_grad_kv_storage_dtype=triton_backward_grad_kv_storage_dtype,
    )

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
  ) -> FFPAAttnMeta:
    """Fill user-facing fields and validate all inputs in place.

    Call this right after :meth:`from_kwargs` to get a fully validated meta::

        meta = FFPAAttnMeta.from_kwargs(**kwargs).normalize(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
        )

    Raises ``TypeError``, ``ValueError``, or ``NotImplementedError`` for
    invalid or unsupported combinations.
    """
    if not 0.0 <= dropout_p <= 1.0:
      raise ValueError(f"ffpa_attn_func: dropout_p must be in [0, 1], got {dropout_p}")
    if dropout_p >= 1.0:
      raise ValueError("ffpa_attn_func: dropout_p=1.0 is not supported by SDPA fused kernels")
    if dropout_p > 0.0 and query.size(-1) > 256 and self.forward_backend == "cutedsl":
      raise NotImplementedError("ffpa_attn_func: large-D dropout is not supported by forward_backend='cutedsl'")
    if attn_mask is not None and is_causal:
      raise RuntimeError("ffpa_attn_func: explicit attn_mask should not be set when is_causal=True")
    if attn_mask is not None and attn_mask.dtype == torch.bool and attn_mask.requires_grad:
      raise TypeError("ffpa_attn_func: boolean attn_mask cannot require gradients")

    # Fill in user-facing fields.
    self.is_causal = is_causal
    self.dropout_p = float(dropout_p)
    self.is_grad_enabled = torch.is_grad_enabled()

    # Validate that acc-code is compatible with activation dtype.
    if query.dtype == torch.bfloat16 and self.acc == _ACC_F16:
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
      self.scale = 1.0 / math.sqrt(query.size(-1))
    else:
      self.scale = float(scale)

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
    self.normalize(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
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
    is_grad = meta.is_grad_enabled and any(x.requires_grad for x in (q, k, v, attn_bias) if x is not None)
    head_dim = q.size(-1)
    O = torch.empty_like(q)  # noqa: E741

    if head_dim <= 256:
      O, lse, rng_state, unused = _aten_flash_attn_forward(
        q,
        k,
        v,
        O,
        meta.is_causal,
        meta.scale,
        meta.dropout_p,
      )
    elif meta.forward_backend == "cuda":
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.dropout_p)
      cuda_forward_impl = _require_cuda_forward_impl()
      O, lse = cuda_forward_impl(
        q,
        k,
        v,
        O,
        attn_bias,
        meta.stages,
        meta.acc,
        int(meta.is_causal),
        meta.scale,
        meta.dropout_p,
        int(rng_state[0].item()) if rng_state.numel() else 0,
        int(rng_state[1].item()) if rng_state.numel() else 0,
        0,
      )
    elif meta.forward_backend == "triton":
      rng_state = _reserve_large_d_dropout_rng(q, k, meta.dropout_p)
      O, lse = _ffpa_attn_forward_triton(
        q,
        k,
        v,
        O,
        meta.is_causal,
        meta.scale,
        meta.triton_autotune,
        meta.triton_autotune_mode,
        attn_bias,
        meta.dropout_p,
        int(rng_state[0].item()) if rng_state.numel() else 0,
        int(rng_state[1].item()) if rng_state.numel() else 0,
        bool(meta.enable_forward_tma),
        bool(meta.enable_forward_ws),
      )
    else:
      raise ValueError(f"Unsupported forward_backend={meta.forward_backend!r};")

    # NO unused output from the FFPA CUDA forward / backward kernels, but we
    # need to return something to keep the autograd contract
    # consistent across backends. Return empty tensors on large-D paths since the
    # small-D path's backward expects tensors to be returned and saved.
    if head_dim > 256 and meta.forward_backend != "triton":
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
      if meta.backward_backend == "triton":
        dq, dk, dv, grad_attn_bias = _ffpa_attn_backward_triton(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.is_causal,
          softmax_scale=meta.scale,
          autotune=meta.triton_autotune,
          autotune_mode=meta.triton_autotune_mode,
          preprocess_d_chunk=meta.triton_backward_preprocess_d_chunk,
          attn_bias=attn_bias,
          return_attn_bias_grad=ctx.needs_input_grad[3],
          grad_kv_storage_dtype=meta.triton_backward_grad_kv_storage_dtype,
          dropout_p=meta.dropout_p,
          philox_seed=int(rng_state[0].item()) if rng_state.numel() else 0,
          philox_offset=int(rng_state[1].item()) if rng_state.numel() else 0,
          enable_tma=bool(meta.enable_backward_tma),
          enable_ws=bool(meta.enable_backward_ws),
          enable_persist_dkdv=meta.triton_backward_enable_persist_dkdv,
          enable_split_launch=meta.triton_backward_enable_split_launch,
        )
      else:
        dq, dk, dv, grad_attn_bias = _aten_efficient_attn_backward(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.is_causal,
          softmax_scale=meta.scale,
          high_precision_grad=meta.high_precision_grad,
          attn_bias=attn_bias,
          return_attn_bias_grad=ctx.needs_input_grad[3],
          dropout_p=meta.dropout_p,
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
        meta.is_causal,
        rng_state,
        unused,
        meta.scale,
        meta.dropout_p,
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
#   cuda              │  triton, sdpa
#   triton            │  triton, sdpa
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


__all__ = [
  "FFPAAttnMeta",
  "FFPAAttnFunc",
  "cuda_forward_available",
  "cuda_backward_available",
]
