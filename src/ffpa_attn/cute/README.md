# FFPA CuTeDSL Internal Package (`ffpa_attn.cute`)

This package hosts the internal CuTeDSL implementation used by the public
FFPA entry points. End users should normally call `ffpa_attn_func` or
`ffpa_attn_varlen_func` from the top-level `ffpa_attn` namespace and select the
backend with `backend="cutedsl"` or `forward_backend="cutedsl"` /
`backward_backend="cutedsl"`.

The package name is `ffpa_attn.cute`, but the public backend string remains
`"cutedsl"` and `CuTeDSLBackend` remains the public configuration class.

[cutedsl]: https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL

---

## Public entry points

| Public API | Where it lives | How it reaches this package |
|---|---|---|
| `ffpa_attn.ffpa_attn_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Dense SDPA-layout wrapper. Opt in with `backend="cutedsl"`. |
| `ffpa_attn.ffpa_attn_varlen_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Packed-THD varlen wrapper. CuTeDSL is the only backend. |

Both APIs route into this package through architecture-aware wrappers defined in
`__init__.py`. Dense and varlen paths share the same SM90 specialised kernels
when `head_dim <= 512` on Hopper and otherwise fall back to the generic SM80
Split-D implementation.

---

## Usage

### 1. Dense path through `ffpa_attn_func`

```python
import torch
from ffpa_attn import ffpa_attn_func

B, H, N, D = 1, 32, 8192, 512
q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = ffpa_attn_func(q, k, v, is_causal=True, backend="cutedsl")
out.sum().backward()
```

Dense tensors use SDPA layout `[B, H, N, D]`. The wrapper converts to the
CuTeDSL-native `[B, N, H, D]` layout internally, dispatches through
`ffpa_attn::_fwd_cute` / `ffpa_attn::_bwd_cute`, then converts outputs back.

`meta.fallback()` only handles hardware and head-dim mismatches. Unsupported
features such as `attn_mask`, `dropout_p > 0`, unsupported dtypes, or invalid
head-dim layouts still raise immediately.

### 2. Varlen path through `ffpa_attn_varlen_func`

```python
import torch
from ffpa_attn import ffpa_attn_varlen_func

lengths = [4096, 4096]
total_q = sum(lengths)
num_heads, head_dim = 32, 512

q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)
cu_seqlens = torch.tensor([0, 4096, 8192], dtype=torch.int32, device="cuda")

out, lse = ffpa_attn_varlen_func(
  q,
  k,
  v,
  cu_seqlens_q=cu_seqlens,
  cu_seqlens_k=cu_seqlens,
  max_seqlen_q=max(lengths),
  max_seqlen_k=max(lengths),
  causal=True,
  return_lse=True,
)
out.sum().backward()
```

The varlen path consumes packed `[T, H, D]` tensors natively and dispatches
through the autograd-registered `ffpa_attn::_varlen_fwd_cute` /
`ffpa_attn::_varlen_bwd_cute` custom ops.

### 3. Internal availability helpers

```python
import torch
from ffpa_attn.cute import cute_forward_available, cute_backward_available

device = torch.device("cuda", 0)
if cute_forward_available(device) and cute_backward_available(device):
  ...
```

These helpers only check the device-level prerequisite `compute capability >= 8.0`.
Per-call head-dim, dtype, and unsupported-feature validation still happens in
`_require_cute_supported()`.

---

## Architecture

```
ffpa_attn_func(backend='cutedsl')
  └── FFPAAttnFunc.apply
        forward  → _ffpa_attn_forward_cute  (__init__.py)
                    └── torch.ops.ffpa_attn._fwd_cute
                          └── _forward_impl_for_device(...)
                                ├── _ffpa_attn_forward_sm90
                                └── _ffpa_attn_forward_sm80
        backward → _ffpa_attn_backward_cute  (__init__.py)
                    └── torch.ops.ffpa_attn._bwd_cute
                          └── _backward_impl_for_device(...)
                                ├── _ffpa_attn_backward_sm90
                                └── _ffpa_attn_backward_sm80

ffpa_attn_varlen_func(...)
  └── _ffpa_attn_varlen_cute  (__init__.py)
        └── _ffpa_attn_varlen_impl
              └── torch.ops.ffpa_attn._varlen_fwd_cute / _varlen_bwd_cute
```

The routing rule is simple:

- `major == 9` and symmetric `head_dim <= 512` uses the SM90 specialised path.
- Every other supported architecture (`SM80/SM89/SM100/SM103/SM120/...`) and
  every larger supported dense head-dim uses the SM80 generic Split-D path.

---

## File roles

| File | Role |
|---|---|
| `__init__.py` | Dense/varlen entry shims, SDPA↔FA layout adapters, torch custom ops, fake registrations, autograd registration, `_require_cute_supported()`, `cute_{forward,backward}_available()`, `cute_max_supported_head_dim()`. |
| `_utils.py` | Shared constants, validation helpers, optional-int encoding, and fake-mode helpers. |
| `_ffpa_fwd_sm90.py` | SM90 forward entry and compile cache. |
| `_ffpa_bwd_sm90.py` | SM90 backward entry, preprocess wiring, and compile caches. |
| `_ffpa_fwd_sm80.py` | SM80/SM89 and generic fallback forward entry. |
| `_ffpa_bwd_sm80.py` | SM80/SM89 and generic fallback backward entry. |
| `_fwd_d512_sm90.py` | Hopper D512 forward kernel. |
| `_fwd_d384_sm90.py` | Hopper D384-aware forward kernel. |
| `_fwd_generic_sm90.py` | Hopper generic forward wrapper. |
| `_fwd_generic_sm80.py` | Ampere-style generic forward kernel and scheduling logic. |
| `_bwd_preprocess.py` | Shared backward preprocess kernel. |
| `_dkdv_d512_sm90.py` | Hopper D512 dK/dV kernel. |
| `_dkdv_d384_sm90.py` | Hopper D384 dK/dV kernel. |
| `_dkdv_generic_sm90.py` | Hopper generic dK/dV wrapper. |
| `_dkdv_generic_sm80.py` | Generic SM80 fallback dK/dV kernel. |
| `_dq_d512_sm90.py` | Hopper D512 dQ kernel. |
| `_dq_d384_sm90.py` | Hopper D384 dQ kernel. |
| `_dq_generic_sm90.py` | Hopper generic dQ wrapper. |
| `_dq_generic_sm80.py` | Generic SM80 fallback dQ kernel. |
| `utils/` | Shared scheduling, masking, softmax, pipeline, cache, and codegen helpers. |

Kernel class names are implementation details. File paths and exported wrapper
functions are the stable internal contract for this package.

---

## Hard constraints

| Constraint | Detail |
|---|---|
| **GPU** | Device-level gate is `sm >= 80`. SM90 gets specialised kernels; other supported archs use the SM80 fallback. |
| **Dense head dim** | `320 <= D <= cute_max_supported_head_dim()`. Today that ceiling is `1024` via the SM80 fallback. |
| **SM90 specialised range** | `320 <= D <= 512` with symmetric q/k/v head-dim. |
| **SM80 fallback divisibility** | Dense fallback requires `D % 32 == 0` in forward and symmetric q/k/v head-dim. |
| **Varlen** | Uses the same architecture routing; feature support is still more constrained than dense and should be validated through the wrapper. |
| **Dtype** | fp16 or bf16 only. |
| **attn_mask** | Not supported. |
| **Dropout** | `dropout_p == 0.0` only. |
| **Varlen extras** | `seqused_k`, `block_table`, `num_splits`, `window_size`, `alibi_slopes`, `softcap`, `score_mod`, `aux_tensors`, `sink`, `attention_mask`, and `block_mask` are rejected. |
| **GQA / MQA** | Supported when head-count divisibility rules are satisfied. |

Any ineligible call surfaces `NotImplementedError`, `TypeError`, or
`ValueError`; there is no silent feature fallback inside this package.

---

## Blackwell / SM120 investigation note

The current kernels are not made Blackwell-compatible by simply relaxing the
Python-side `sm == 90` checks to `sm >= 90`. A May 2026 experiment on
`NVIDIA RTX PRO 6000 Blackwell Server Edition`, (compute capability 12.0) showed that the D512 path reaches CuTeDSL JIT after the local gates are relaxed, but then fails inside CUTLASS DSL's Hopper
warpgroup MMA implementation:

```text
cutlass.cute.nvgpu.common.OpError: expects arch to be Arch.sm_90a, but got Arch.sm_120a
```

Forcing `CUTE_DSL_ARCH=sm_90a` can compile the Hopper target, but it fails at
runtime on Blackwell with `cudaErrorNoKernelImageForDevice`. Temporarily
relaxing the installed CUTLASS DSL `warpgroup.MmaOp` arch check from
`Arch.sm_90a` to `>= Arch.sm_90a` only moves the failure to NVVM module
serialization for `sm_120` / `sm_120a`. In other words, the blocker is not just
an over-strict guard in this package.

Future Blackwell work should add a real SM100+/SM120 CuTeDSL path instead of
trying to reuse the Hopper `warpgroup` path. CUTLASS provides a separate
Blackwell stack based on `cutlass.cute.nvgpu.tcgen05` and
`cutlass.utils.blackwell_helpers.make_trivial_tiled_mma`. That helper constructs
`tcgen05.MmaF16BF16Op`, whose signature includes `CtaGroup` and uses Blackwell
operand sources such as `tcgen05.OperandSource.SMEM` / `TMEM`. This differs from
the current Hopper helper, which constructs `warpgroup.MmaF16BF16Op` and only
accepts `Arch.sm_90a`.

Useful references for a future port:

- `cutlass/utils/blackwell_helpers.py` - Blackwell `make_trivial_tiled_mma` and
  TMEM/TMA helper utilities.
- `cutlass/cute/nvgpu/tcgen05/mma.py` - Blackwell MMA op definitions and arch
  admissibility checks.
- `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/`
  in the CUTLASS checkout - attention examples using `tcgen05`, `CtaGroup`, and
  TMEM-backed PV-style MMA.

---

## References

- **Public API** — `ffpa_attn_func` / `ffpa_attn_varlen_func` docstrings in `src/ffpa_attn/ffpa_attn_interface.py`.
- **CuTeDSL** — https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL
