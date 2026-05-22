# FFPA CuTeDSL Backend (`ffpa_attn.cutedsl`)

SM90 (Hopper) dense `320<D<=512` forward and backward kernels
for FFPA, implemented in [CuTeDSL][cutedsl]. This package is an **internal
backend**: end users should not import from `ffpa_attn.cutedsl` directly.
The supported entry points live at the top of the `ffpa_attn` namespace:

| Public API | Where it lives | When to use the cutedsl backend |
|---|---|---|
| `ffpa_attn.ffpa_attn_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Fixed-shape batched attention. Opt in with `backend="cutedsl"`. |
| `ffpa_attn.ffpa_attn_varlen_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Variable-length packed `THD` attention. CuTeDSL is the **only** backend; no flag needed. |

Both APIs route into this package, but along different paths — see
[Architecture](#architecture) below. The kernels are bit-identical across
paths (same `_ffpa_attn_forward_sm90` / `_ffpa_attn_backward_sm90`).

[cutedsl]: https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL

---

## Usage

### 1. `ffpa_attn_func` — fixed-shape batched attention (opt-in)

Tensors use SDPA layout `[B, Nh, N, D]`. Forward and backward accept fp16 or bf16.

```python
import torch
from ffpa_attn import ffpa_attn_func

B, Nh, N, D = 1, 32, 8192, 512
q = torch.randn(B, Nh, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Opt into CuTeDSL with the backend shorthand (auto-fills forward + backward).
out = ffpa_attn_func(q, k, v, is_causal=True, backend="cutedsl")
out.sum().backward()

# Or use the explicit forward_backend / backward_backend pair.
from ffpa_attn import CuTeDSLBackend
fwd = CuTeDSLBackend(forward=True)
bwd = CuTeDSLBackend(backward=True)
out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=fwd, backward_backend=bwd)
```

The cutedsl path flows through the standard `FFPAAttnMeta` → `FFPAAttnFunc`
dispatch.  `meta.fallback()` handles cutedsl hardware mismatches
(head_dim outside `320<D<=512` or non-SM90) by falling back to native SDPA with a
`warning_once`.  All other constraints (dtype, dropout>0,
explicit attn_mask, FA-extension kwargs) raise `NotImplementedError`
immediately — there is no silent fallback.

### 2. `ffpa_attn_varlen_func` — packed THD (CuTeDSL-only)

Tensors are packed `[T, H, D]` (FlashAttention-style varlen layout).
`cu_seqlens_q` / `cu_seqlens_k` are `int32` CUDA tensors of length
`B + 1`, starting at 0 and ending at `T_q` / `T_k`.

```python
import torch
from ffpa_attn import ffpa_attn_varlen_func

# Two sequences of length 4096 each, packed.
seqlens = [4096, 4096]
T = sum(seqlens)
H, D = 32, 512

q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

cu_seqlens = torch.tensor([0, *torch.cumsum(torch.tensor(seqlens), 0).tolist()],
                          dtype=torch.int32, device="cuda")

out, lse = ffpa_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max(seqlens),
    max_seqlen_k=max(seqlens),
    causal=True,
    return_lse=True,
)
out.sum().backward()
```

`ffpa_attn_varlen_func` dispatches directly to the autograd-registered
`torch.ops.ffpa_attn._varlen_fwd_cutedsl` custom op, registered in `__init__.py`.

The following FlashAttention-compat kwargs are explicitly **rejected**
(they raise `NotImplementedError`):

- `seqused_k`
- `block_table`
- `num_splits`
- `window_size`
- `alibi_slopes`
- `softcap`

In addition: `dropout_p > 0.0` is rejected. For other shapes / backends,
unpack the batch and call `ffpa_attn_func` per sequence.

### 3. Backend availability gate

```python
import torch
from ffpa_attn.cutedsl import cutedsl_forward_available, cutedsl_backward_available

dev = torch.device("cuda", 0)
if cutedsl_forward_available(dev):
    # SM 9.x Hopper — dense head_dim / dtype / no-mask / no-dropout are still
    # validated per-call inside _require_cutedsl_supported.
    ...
```

`cutedsl_forward_available` and `cutedsl_backward_available` only check
the hardware prerequisite (SM 9.x).

---

## Architecture

All call paths converge on the same kernel entry functions:

```
ffpa_attn_func(backend='cutedsl')
  └── FFPAAttnFunc.apply
        forward  → _ffpa_attn_forward_cutedsl  (__init__.py)
                    └── _ffpa_attn_forward_sm90  (_ffpa_fwd_sm90.py)
        backward → _ffpa_attn_backward_cutedsl  (__init__.py)
                    └── _ffpa_attn_backward_sm90  (_ffpa_bwd_sm90.py)

ffpa_attn_varlen_func(...)
  └── _ffpa_attn_varlen_cutedsl  (__init__.py)
        └── torch.ops.ffpa_attn._varlen_fwd_cutedsl
              → _ffpa_attn_forward_sm90 / _ffpa_attn_backward_sm90
```

### File roles

| File | Role |
|---|---|
| `__init__.py` | Public API wrappers, SDPA↔FA layout adapters, torch custom ops, varlen autograd, `_require_cutedsl_supported`, `cutedsl_{forward,backward}_available`. |
| `_utils.py` | Shared constants (`SUPPORTED_HEAD_DIM`, tile sizes, dtype map) and validation helpers. |
| `_ffpa_fwd_sm90.py` | Forward entry: `_ffpa_attn_forward_sm90()` + JIT compile cache. |
| `_ffpa_bwd_sm90.py` | Backward entry: `_ffpa_attn_backward_sm90()`, `_bwd_preprocess()`, compile caches. |
| `_fwd_d512_sm90.py` | Forward kernel class `FFPAAttnFwdSm90SplitD`: D512 full-D 3-warpgroup pipeline. |
| `_fwd_generic_sm90.py` | Generic dense forward wrapper `FFPAAttnFwdSm90SplitDGeneric`: dispatches logical large-D tensors through the D512 physical tile. |
| `_fwd_d384_sm90.py` | D384-aware forward wrapper `FFPAAttnFwdSm90SplitDD384Aware`: uses a D384 physical tile for `320<D<=384`. |
| `_bwd_preprocess.py` | Bwd preprocess kernel class `FFPAAttnBwdPreprocess`: computes `D_i = (O⊙dO).sum(-1)`. |
| `_dkdv_d512_sm90.py` | D512 `dK`+`dV` kernel class `FFPAAttnBwdDKDVSm90SplitD`. |
| `_dkdv_generic_sm90.py` | Generic dense `dK`+`dV` wrapper `FFPAAttnBwdDKDVSm90SplitDGeneric`: dispatches logical large-D tensors through the D512 physical tile. |
| `_dkdv_d384_sm90.py` | D384 `dK`+`dV` kernel class `FFPAAttnBwdDKDVSm90SplitDD384`: full/tail D split with D384-aware stores. |
| `_dq_d512_sm90.py` | D512 `dQ` kernel class `FFPAAttnBwdDQSm90SplitD`: dual-asymmetric MMA warpgroups. |
| `_dq_generic_sm90.py` | Generic dense `dQ` wrapper `FFPAAttnBwdDQSm90SplitDGeneric`: dispatches logical large-D tensors through the D512 physical tile. |
| `_dq_d384_sm90.py` | D384 `dQ` kernel class `FFPAAttnBwdDQSm90SplitDD384`: full/tail D split with D384-aware stores. |
| `utils/` | Shared kernel helpers (tile scheduling, softmax, mask, pipeline, etc.). |

Kernel class names are implementation details. File paths are the stable contract.

---

## Hard constraints

| Constraint | Detail |
|---|---|
| **GPU** | SM 9.x (Hopper). WGMMA, TMA, named barriers. |
| **Head dim** | Dense path supports `320<D<=512`; varlen remains `D == 512`. |
| **Dtype (fwd)** | fp16 or bf16. |
| **Dtype (bwd)** | fp16 or bf16. |
| **attn_mask** | Not supported. |
| **Dropout** | `dropout_p == 0.0`. |
| **Varlen extras** | seqused_k, block_table, num_splits, window_size, alibi_slopes, softcap rejected. |
| **GQA / MQA** | Supported; pass `enable_gqa=True`. |

Any ineligible call surfaces `NotImplementedError` — no silent fallback.

---

## Blackwell / SM120 investigation note

The current kernels are not made Blackwell-compatible by simply relaxing the
Python-side `sm == 90` checks to `sm >= 90`. A May 2026 experiment on
`NVIDIA RTX PRO 6000 Blackwell Server Edition`,
compute capability 12.0) showed that the D512 path reaches CuTeDSL JIT after
the local gates are relaxed, but then fails inside CUTLASS DSL's Hopper
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
