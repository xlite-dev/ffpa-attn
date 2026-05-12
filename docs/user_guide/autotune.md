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
		 triton_forward_autotune=True,
		 triton_backward_autotune=True,
		 triton_autotune_mode="fast",
	)
	```

	Triton benchmarks candidate configs and caches the best config in the
	current process. This is convenient for experiments, but the chosen config is
	not stored in the FFPA repository.

2. Generate persistent tuned configs once, then use the default non-autotune
	path:

	```bash
	python -m ffpa_attn.autotune --mode fast --directions both --overwrite
	```

	The generated JSON is saved under
	`src/ffpa_attn/triton/configs/{device_name}.json`, for example
	`src/ffpa_attn/triton/configs/NVIDIA_L20.json`.

	Later calls with `triton_forward_autotune=False` and
	`triton_backward_autotune=False` will automatically load the matching
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

The first version supports `B=1` and `H=32` for config generation:

```bash
python -m ffpa_attn.autotune --mode fast --B 1 --H 32 --overwrite
```

You can generate only forward configs, only backward configs, or both:

```bash
python -m ffpa_attn.autotune --mode fast --directions forward --overwrite
python -m ffpa_attn.autotune --mode fast --directions backward --overwrite
python -m ffpa_attn.autotune --mode fast --directions both --overwrite
```

`both` is the default.

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

The `16384` sequence length is generated only when the current GPU has at least
48 GiB of memory. Smaller-memory devices skip it.

The generated matrix covers:

| Direction | Kernels |
| --- | --- |
| Forward | generic forward, split-KV/decode stage1 |
| Backward | delta preprocess, main backward, decode backward stage1 |

Forward tasks include self-attention, cross-attention, decode attention
(`Nq=1`), causal, and non-causal variants.

Backward tasks include main backward shapes (`Nq >= 512`) and decode backward
shapes (`Nq=1` and a small `Nq < 8` case), with causal and non-causal variants.

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

Backward lookup additionally filters by:

| Field | Meaning |
| --- | --- |
| `kernel` | `bwd_preprocess`, `bwd_generic`, or `decode_bwd_stage1`. |
| `preprocess_d_chunk` | Applies to the delta preprocess kernel. |
| `bias_grad` | Whether attention-bias gradients are requested. |
| `grad_v_storage_dtype` | Optional internal `dV` storage override. |
| `use_gemv` | Decode backward single-query specialization. |
| `has_dropout` | Whether dropout replay is active. |

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
	CUDA_VISIBLE_DEVICES=0 \
	python -m ffpa_attn.autotune --mode fast --directions both --overwrite
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
		 triton_forward_autotune=False,
		 triton_backward_autotune=False,
		 triton_autotune_mode="fast",
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
