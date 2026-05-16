# FFPA CuTeDSL Backend (`ffpa_attn.cutedsl`)

SM90 (Hopper) + `head_dim == 512` specialized forward and backward kernels
for FFPA, implemented in [CuTeDSL][cutedsl]. This package is an **internal
backend**: end users should not import from `ffpa_attn.cutedsl` directly.
The supported entry points live at the top of the `ffpa_attn` namespace:

| Public API | Where it lives | When to use the cutedsl backend |
|---|---|---|
| `ffpa_attn.ffpa_attn_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Fixed-shape batched attention. Opt in with `forward_backend='cutedsl'`. |
| `ffpa_attn.ffpa_attn_varlen_func` | `src/ffpa_attn/ffpa_attn_interface.py` | Variable-length packed `THD` attention. CuTeDSL is the **only** backend; no flag needed. |

Both APIs route into this package, but along different paths — see
[Architecture](#architecture) below. The kernels are bit-identical across
paths (same `_flash_attn_fwd_sm90` / `_flash_attn_bwd_sm90`).

[cutedsl]: https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL

---

## Usage

### 1. `ffpa_attn_func` — fixed-shape batched attention (opt-in)

Tensors use SDPA layout `[B, Nh, N, D]`. Forward accepts fp16 or bf16;
backward (`requires_grad=True`) requires bf16.

```python
import torch
from ffpa_attn import ffpa_attn_func

B, Nh, N, D = 1, 32, 8192, 512
q = torch.randn(B, Nh, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Opt into the CuTeDSL backend explicitly. Default is 'triton'.
out = ffpa_attn_func(
    q, k, v,
    is_causal=True,
    forward_backend="cutedsl",
)
out.sum().backward()
```

#### Fast-path

When the call satisfies all four conditions of `_should_take_cutedsl_fast_path`
(see `ffpa_attn_interface.py`), `ffpa_attn_func` short-circuits the standard
multi-backend dispatcher and routes straight to
`cutedsl.interface.split_flash_attn_func`:

1. `forward_backend == 'cutedsl'`
2. `query.size(-1) == 512`
3. `attn_mask is None` and `dropout_p == 0.0`
4. `cutedsl_forward_available(device) == True` (SM 9.x Hopper)

The fast-path skips `FFPAAttnMeta.normalize_inputs`, the `FFPAAttnFunc`
autograd boundary, and the `_wrappers.py` layout shim — saving roughly
50–200 µs of Python dispatch overhead and one autograd Function boundary
(which is also a `torch.compile` graph break). It calls the **same**
underlying kernel as the slow path, so numerics are bit-identical.

If any condition fails the call falls through to the standard FFPA
dispatcher. If the user still asked for `forward_backend='cutedsl'` but
something else is unsupported (wrong head_dim, dropout, mask, ...) the
slow path surfaces the canonical `NotImplementedError` / `TypeError`
from `_require_cutedsl_supported` instead of silently switching backends.

A deeper analysis of the fast-path bypass — including a decision matrix
and what each layer skips — lives in
`dev_md/cutedsl_version/analsys/How_to_use_cutedsl_fast_path_interface.md`
(Chinese).

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

`ffpa_attn_varlen_func` has no fast/slow split — it always dispatches
directly to the autograd-registered `torch.ops.splitd_flash_attn.varlen_fwd`
custom op, which is registered when this package is imported.

The following FlashAttention-compat kwargs are explicitly **rejected**
(they raise `NotImplementedError`):

- `seqused_k`
- `block_table`
- `num_splits`
- `window_size`
- `alibi_slopes`
- `softcap`

In addition: `dropout_p > 0.0` and any `forward_backend` other than
`'cutedsl'` (or unset) are rejected. For other shapes / backends, unpack
the batch and call `ffpa_attn_func` per sequence.

### 3. Backend availability gate

Before allocating tensors, callers can probe whether the CuTeDSL backend
can run on a given device:

```python
import torch
from ffpa_attn.cutedsl import cutedsl_forward_available, cutedsl_backward_available

dev = torch.device("cuda", 0)
if cutedsl_forward_available(dev):
    # SM 9.x Hopper — D=512 / dtype / no-mask / no-dropout are still
    # validated per-call inside _require_cutedsl_supported.
    ...
```

`cutedsl_forward_available` and `cutedsl_backward_available` only check
the hardware prerequisite (SM 9.x). All other constraints — head_dim,
dtype, dropout, mask — are validated per call by
`_require_cutedsl_supported`. Backward additionally requires bf16, which
is checked at call time (not by the availability probe).

---

## Architecture

Three call paths converge on `cutedsl.interface`:

```
user code
  │
  ├── ffpa_attn_func(forward_backend='cutedsl', ...)
  │     │
  │     ├── fast-path: _ffpa_attn_func_cutedsl_fast_path
  │     │     └── cutedsl.interface.split_flash_attn_func
  │     │           └── FlashAttnFunc.apply
  │     │                 → _flash_attn_fwd_sm90 / _flash_attn_bwd_sm90
  │     │
  │     └── slow-path: FFPAAttnFunc.apply
  │           └── cutedsl._wrappers._ffpa_attn_{forward,backward}_cutedsl
  │                 └── cutedsl.interface._flash_attn_{fwd,bwd}_sm90
  │
  └── ffpa_attn_varlen_func(...)
        └── _ffpa_attn_varlen_cutedsl
              └── torch.ops.splitd_flash_attn.varlen_fwd
                    (registered in cutedsl.interface)
                      → _flash_attn_fwd_sm90 / _flash_attn_bwd_sm90
```

All three paths bottom out at the same `_flash_attn_fwd_sm90` /
`_flash_attn_bwd_sm90` functions in `interface.py`, which compile and
launch the CuTeDSL kernels. The slow path additionally pays for an
SDPA↔FA layout transpose (in `_wrappers.py`) and the multi-backend
`FFPAAttnFunc` autograd boundary; the fast path and the varlen path
skip both.

### File roles

| File | Role |
|---|---|
| `__init__.py` | Side-effect imports `interface` to register the `splitd_flash_attn::varlen_{fwd,bwd}` torch ops; re-exports the helpers from `_wrappers.py`. |
| `_wrappers.py` | Slow-path SDPA `[B, H, N, D]` ↔ CuTeDSL-native `[B, N, H, D]` layout adapter. Owns `_require_cutedsl_supported` (the canonical gate: SM major, head_dim, dtype, dropout, attn_bias) and the `cutedsl_{forward,backward}_available` device probes. |
| `interface.py` | Top of the CuTeDSL-internal stack. Owns `SUPPORTED_HEAD_DIM = 512`, the `_flash_attn_fwd_sm90` / `_flash_attn_bwd_sm90` entry functions, `split_flash_attn_func` (the fast-path target), the `FlashAttnFunc` autograd `Function`, the `splitd_flash_attn::varlen_{fwd,bwd}` torch.library ops + their autograd wiring, and the SM90 arch / dtype / GQA-pack guards. |
| `_ffpa_fwd_d512_sm90.py` | Forward kernel: full-D 3-warpgroup pipeline (TMA producer + WG1 QK / softmax / PV-front + WG2 PV-back epilogue). |
| `_ffpa_dkdv_d512_sm90.py` | Backward `dK` + `dV` kernel: 1 TMA producer + 1 MMA consumer warpgroup, `d_chunk=256`, K/V persistent in SMEM across `d`-passes. |
| `_ffpa_dq_d512_sm90.py` | Backward `dQ` kernel: dual-asymmetric MMA warpgroups, `d_chunk=256`, cooperative `dQ_front` + `dQ_back`. |
| `flash_bwd_preprocess.py` | Backward preprocess: computes `D_i = (O ⊙ dO).sum(-1)` with optional `dLSE` adjustment. Called from `_flash_attn_bwd_sm90` before `dK/dV` and `dQ`. |
| `utils/` | Shared kernel helpers. |

#### `utils/` contents

| File | Purpose |
|---|---|
| `tile_scheduler.py` | SM90 tile scheduling (single-tile fixed-shape and varlen). |
| `named_barrier.py` | Named barrier ID enums for cross-warpgroup synchronization. |
| `softmax.py` | Online softmax building blocks (row reductions, scale application). |
| `mask.py` | Attention masking (causal, local / sliding-window, FlexAttention-style `mask_mod`, seqlen bounds). |
| `seqlen_info.py` | `SeqlenInfoQK`: per-batch ragged / varlen metadata. |
| `block_info.py` | `BlockInfo`: tile range queries (e.g. causal/local min/max `n_block`). |
| `pack_gqa.py` | Group-query attention packing — folds `qhead_per_kvhead` into the seqlen mode. |
| `pipeline.py` | Async pipeline abstractions (`PipelineAsync`, `PipelineTmaAsync`). |
| `cute_dsl_utils.py` | Tensor / dtype conversion helpers between `torch.Tensor` and CuTeDSL. |
| `cache_utils.py` | JIT / AOT kernel cache (hash-based). |
| `cute_dsl_ptxas.py` | `ptxas` override wrapper and spill / LDS warnings. |
| `fa_logging.py` | `FA_LOG_LEVEL` env-var-controlled host + kernel tracing. |
| `runtime.py` | Fake-mode detection and TVM FFI stream integration. |

The diagram and table intentionally avoid naming kernel **classes** —
those are implementation details and have churned recently. File paths
are the stable contract.

---

## Hard constraints

| Constraint | Detail |
|---|---|
| **GPU** | SM 9.x (Hopper) only. The kernels use WGMMA, TMA, and named barriers. Verified by `_require_cutedsl_supported` and `cutedsl_forward_available`. |
| **Head dim** | `D == 512` only. Other head dims raise `NotImplementedError`; use FFPA's Triton backend instead. |
| **Dtype (forward)** | `torch.float16` or `torch.bfloat16`. |
| **Dtype (backward)** | `torch.bfloat16` **required** for `requires_grad=True`. fp16 training raises `NotImplementedError`. |
| **`attn_mask` / `attn_bias`** | Not supported in either direction. |
| **Dropout** | `dropout_p` must be `0.0`. |
| **Varlen FA-extras** | `seqused_k`, `block_table`, `num_splits`, `window_size`, `alibi_slopes`, `softcap` all rejected. |
| **GQA / MQA** | Supported. With `ffpa_attn_func` pass `enable_gqa=True` when `Nh_q != Nh_kv`. The kernel auto-packs when `qhead_per_kvhead > 1`. |

Any ineligible call surfaces an actionable error from
`_require_cutedsl_supported` — there is no silent fallback.

---

## References

- **Public API contracts** — per-argument docstrings on `ffpa_attn_func`
  and `ffpa_attn_varlen_func` in `src/ffpa_attn/ffpa_attn_interface.py`.
- **Fast-path deep-dive** —
  `dev_md/cutedsl_version/analsys/How_to_use_cutedsl_fast_path_interface.md`
  (Chinese). Covers the fast-path bypass mechanics, a decision matrix,
  and the exact set of layers each path skips.
- **CuTeDSL** — https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL
