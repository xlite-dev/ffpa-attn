# Triton Autotune and Persistent Tuned Configs

FFPA's Triton backend can autotune forward and backward launch parameters for
large-head-dimension attention. The autotune result can be persisted as a
device-specific JSON file and reused later when runtime autotune is disabled.

This is useful for production inference or training jobs where you want stable
startup latency and do not want each process to pay the Triton autotune cost.

## Overview

There are two ways to use Triton tuned configs:

1. Run with runtime autotune enabled for one process:

```python
	from ffpa_attn import ffpa_attn_func

	out = ffpa_attn_func(
		 q,
		 k,
		 v,
		 forward_backend="triton",
		 backward_backend="triton",
		 triton_autotune=True,
		 triton_autotune_mode="fast",
	)
```

Triton benchmarks candidate configs and caches the best config in the current process. This is convenient for experiments, but the chosen config is not stored in the FFPA repository.

2. Generate persistent tuned configs once, then use the default non-autotune path:

```bash
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

The generated JSON is saved under `src/ffpa_attn/triton/configs/{device_name}.json`, for example
`src/ffpa_attn/triton/configs/NVIDIA_L20.json`.

Later calls with `triton_autotune=False` will automatically load the matching
device config when it exists.

## Generate Persistent Configs

Run the autotune generator from the repository root or from an installed FFPA
environment:

```bash
python -m ffpa_attn.autotune --mode fast
```

By default, the command refuses to overwrite an existing device config. Use
`--overwrite` when you intentionally want to regenerate it:

```bash
python -m ffpa_attn.autotune --mode fast --overwrite
```

The generator defaults to `B=1` and `H=32`. You can change them when your
deployment shape uses a different batch size or query-head count:

```bash
python -m ffpa_attn.autotune --mode fast --B 1 --H 32 --overwrite
```

By default, the generated task grid covers the baseline no-mask, no-dropout,
equal-head cases. Add `--full-tasks` to also tune canonical `attn_mask`,
dropout, GQA, and MQA variants modeled after `examples/perf.py`:

```bash
python -m ffpa_attn.autotune \
	--mode fast \
	--directions both \
	--dtypes bf16,fp16 \
	--full-tasks \
	--overwrite
```

`--full-tasks` can increase autotune time substantially because each additional
variant is benchmarked separately. It is intentionally disabled by default so
existing generation jobs keep their current coverage and runtime.

You can generate only forward configs, only backward configs, or both:

```bash
python -m ffpa_attn.autotune --mode fast --directions forward --overwrite
python -m ffpa_attn.autotune --mode fast --directions backward --overwrite
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

`both` is the default.

The default dtype set is `bf16`. For benchmarks such as `examples/perf.py` that
run both `fp16` and `bf16`, generate both dtype configs explicitly:

```bash
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

### Output Location

The default output directory is the package config directory:

```text
src/ffpa_attn/triton/configs/
```

The file name is derived from `torch.cuda.get_device_name()` with non-file-name
characters replaced by underscores. For example:

```text
NVIDIA L20 -> NVIDIA_L20.json
NVIDIA GeForce RTX 5090 -> NVIDIA_GeForce_RTX_5090.json
```

For smoke tests or CI, write to a temporary directory so partial configs are
not mistaken for full device configs:

```bash
python -m ffpa_attn.autotune \
  --mode fast \
  --directions both \
  --overwrite \
  --output-dir /tmp/ffpa-config-smoke
```

At runtime, FFPA can also load configs from a custom directory with
`FFPA_TUNED_CONFIG_DIR`:

```bash
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke python your_script.py
```

## Autotune Modes

The generator supports the same mode names as the runtime Triton autotune path:

| Mode | Purpose |
| --- | --- |
| `fast` | Smaller search space. Recommended as the default persistent config mode. |
| `max` | Larger search space. Slower to generate, but may find better configs. |

The runtime lookup requires the mode to match. A JSON generated with
`--mode fast` is used when `triton_autotune_mode="fast"`; a JSON generated with
`--mode max` is used when `triton_autotune_mode="max"`.

## Shape Coverage

The generator tunes the following head dimensions:

```text
320, 512, 640, 768, 1024
```

The sequence-length grid is:

```text
1, 512, 1024, 2048, 4096, 8192, 16384
```

The `1` entry is used only for decode query length (`Nq=1`). Decode tuning does
not generate `Nkv=1` cases because a single-token KV cache is not a meaningful
decode-attention benchmark target.

The `16384` sequence length is generated only when the current GPU has at least
48 GiB of memory. Smaller-memory devices skip it.

Persistent config generation tunes every target sequence length in this grid
with exact Triton autotune keys. It does not reuse the online runtime seqlen
buckets while generating JSON, so an entry for `512`, `1024`, or `2048` means
that shape was benchmarked independently. Runtime lookup still performs reuse
when the workload shape is not an exact persisted entry.

The generated matrix covers:

| Direction | Kernels |
| --- | --- |
| Forward | generic forward, split-KV/decode stage1 |
| Backward | delta preprocess, main backward, decode backward stage1 |

Forward tasks include self-attention, cross-attention, decode attention
(`Nq=1`), causal, and non-causal variants.

Backward tasks include main backward shapes (`Nq >= 512`) and decode backward
shapes (`Nq=1`, `Nkv>1`), with causal and non-causal variants.

When `--full-tasks` is enabled, the generator adds square prefill variants for
each tuned sequence length:

| Case | Shape / variant |
| --- | --- |
| `attn-mask` | Compact additive key-position mask `[1, 1, 1, Nkv]`. Backward tunes the bias-gradient path. |
| `dropout` | `dropout_p=0.1`, using the Triton dropout path. |
| `gqa` | `Hq=H`, `Hkv` chosen with the same divisor rule as `examples/perf.py`. |
| `mqa` | `Hq=H`, `Hkv=1`. |

These are single-feature canonical variants, not a full Cartesian product. For
example, `--full-tasks` tunes `gqa` and `dropout` separately, but it does not
generate a combined GQA+dropout case.

## Runtime Lookup Rules

When runtime autotune is disabled, FFPA tries to load the current device JSON.
If no compatible entry is found, it falls back to the built-in default launch
parameters.

Forward lookup filters by:

| Field | Meaning |
| --- | --- |
| `direction` | Must be `forward`. |
| `kernel` | `fwd_generic` or `decode_fwd_stage1`. |
| `autotune_mode` | Must match `triton_autotune_mode`. |
| `dtype` | `fp16` or `bf16`. |
| `causal` | Must match `is_causal`. |
| `has_attn_bias` | Whether `attn_mask` / additive bias is present. |
| `has_dropout` | Whether dropout is active. |

Backward lookup additionally filters by:

| Field | Meaning |
| --- | --- |
| `kernel` | `bwd_preproc`, `bwd_generic`, or `decode_bwd_stage1`. |
| `preprocess_d_chunk` | Applies to the delta preprocess kernel. |
| `bias_grad` | Whether attention-bias gradients are requested. |
| `grad_v_storage_dtype` | Optional internal `dV` storage override. |
| `use_gemv` | Decode backward single-query specialization. |
| `has_dropout` | Whether dropout replay is active. |
| `has_attn_bias` | Whether additive bias is active. |

Generated entries may include `nheads_q` and `nheads_kv` for logging and JSON
metadata. Runtime lookup can prefer an exact recorded head layout when one is
available, but it does not require the head layout to match. Batch size and
head count commonly vary across workloads, so FFPA reuses the same launch
config across compatible mask/dropout/causal/kernel variants instead of missing
the persistent config because `Hq` or `Hkv` changed.

Configs generated before these variant fields existed are treated as no-mask,
no-dropout entries. They can still satisfy baseline requests, while
`has_attn_bias` and `has_dropout` continue to prevent semantically different
kernel variants from being mixed.

After variant filtering, FFPA chooses the nearest persisted head dimension. Ties
prefer the larger candidate. Examples:

| Runtime `D` | Persisted `D` |
| --- | --- |
| 384 | 320 |
| 448 | 512 |
| 900 | 1024 |

For sequence length, FFPA chooses the smallest persisted sequence length that is
greater than or equal to the runtime value. If the runtime value is larger than
all persisted values, FFPA uses the largest available persisted value. Examples:

| Runtime seqlen | Persisted seqlen |
| --- | --- |
| 3000 | 4096 |
| 32768 | 8192 or 16384, depending on what was generated |

### Debug Persistent Lookup

Set `FFPA_LOGGER_LEVEL=DEBUG` when you want to verify that runtime calls are
using persistent tuned configs instead of falling back to the built-in defaults:

```bash
FFPA_LOGGER_LEVEL=DEBUG \
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke \
python examples/perf.py --case decode-attn --backend ffpa-triton
```

On repeated runtime lookup hits, FFPA logs the kernel name and sanitized launch
config selected from the in-process persistent config cache. The message uses
`debug_once` semantics, so the same cache-hit/config line is emitted once per
process instead of repeating on every attention call.

## Development Smoke Tests

Full autotune generation can take a long time. Use `FFPA_AUTOTUNE_MAX_CONFIGS`
to cap the number of shapes during development:

```bash
CUDA_VISIBLE_DEVICES=0 \
FFPA_AUTOTUNE_MAX_CONFIGS=4 \
python -m ffpa_attn.autotune \
  --mode fast \
  --directions both \
  --overwrite \
  --output-dir /tmp/ffpa-config-smoke
```

Then run a small workload with that temporary directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke \
python your_script.py
```

For repository tests, a focused check is:

```bash
pytest tests/test_persistent_autotune_config.py tests/test_triton_autotune_mode.py -q
```

## Production Workflow

A typical workflow is:

1. Select the deployment GPU type.
2. Generate a full config on that GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ffpa_attn.autotune \
	--mode fast --directions both --dtypes bf16,fp16 --full-tasks --overwrite
```

3. Commit the generated JSON under `src/ffpa_attn/triton/configs/`.
4. Run normal workloads with runtime autotune disabled, which is the default:

	```python
	out = ffpa_attn_func(
		 q,
		 k,
		 v,
		 forward_backend="triton",
		 backward_backend="triton",
	)
	```

If the runtime shape is outside the generated grid, FFPA uses the nearest
compatible persisted config. If the device JSON is missing, malformed, or does
not contain a compatible entry, FFPA silently keeps the existing built-in launch
defaults.

## Current Scope and Limitations

The first persistent-config generator focuses on the common no-bias and
no-dropout path. The JSON schema already records `bias_grad`, `has_dropout`,
and `grad_v_storage_dtype`, so bias-gradient and dropout-specific configs can be
added later without changing the runtime lookup design.

`decode_dq_reduce` and key-bias gradient reduction are not currently autotuned
by FFPA, so they keep their fixed launch parameters.

Persistent configs are device-specific. Do not reuse a JSON generated on one GPU
class as a performance baseline for a different GPU class.
