# Triton Autotune and Persistent Tuned Configs

FFPA's Triton backend can autotune forward and backward launch parameters for large-head-dimension attention. The autotune result can be persisted as a device-specific JSON file and reused later when runtime autotune is disabled.

This is useful for production inference or training jobs where you want stable startup latency and do not want each process to pay the Triton autotune cost.

## Overview

There are two ways to use Triton tuned configs:

1. Run with runtime autotune enabled for one process:

```python
from ffpa_attn import ffpa_attn_func

out = ffpa_attn_func(
	q,
	k,
	v,
	forward_backend="triton", # default is 'triton'
	backward_backend="triton", # default is 'triton'
	triton_autotune=True, # default is False
	triton_autotune_mode="fast", # default is 'fast'.
)
```

Triton benchmarks candidate configs and caches the best config in the current process. This is convenient for experiments, but the chosen config is not stored in the FFPA repository.

2. Generate persistent tuned configs once, then use the default non-autotune path:

```bash
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

The generated JSON is saved under <span style="color:#c77dff;">src/ffpa_attn/triton/configs/{device_name}.json </span>, for example <span style="color:#c77dff;">src/ffpa_attn/triton/configs/NVIDIA_L20.json</span>.

Later calls with <span style="color:#c77dff;">triton_autotune=False</span> will automatically load the matching
device config when it exists.

## Generate Persistent Configs

Run the autotune generator from the repository root or from an installed FFPA environment:

```bash
python -m ffpa_attn.autotune --mode fast
```

By default, the command refuses to overwrite an existing device config. Use
`--overwrite` when you intentionally want to regenerate it:

```bash
python -m ffpa_attn.autotune --mode fast --overwrite
```

The generator defaults to <span style="color:#c77dff;">B=1</span> and <span style="color:#c77dff;">H=32</span>. You can change them when your
deployment shape uses a different batch size or query-head count:

```bash
python -m ffpa_attn.autotune --mode fast --B 1 --H 32 --overwrite
```

By default, the generated task grid covers the baseline no-mask, no-dropout,
equal-head cases. Add <span style="color:#c77dff;">--full-tasks</span> to also tune canonical <span style="color:#c77dff;">attn_mask</span>,
dropout, GQA, and MQA variants modeled after `examples/perf.py`:

```bash
python -m ffpa_attn.autotune \
	--mode fast \
	--directions both \
	--dtypes bf16,fp16 \
	--full-tasks \
	--overwrite
```

<span style="color:#c77dff;">--full-tasks</span> can increase autotune time substantially because each additional variant is benchmarked separately. It is intentionally disabled by default so existing generation jobs keep their current coverage and runtime. You can generate only forward configs, only backward configs, or both:

```bash
python -m ffpa_attn.autotune --mode fast --directions forward --overwrite
python -m ffpa_attn.autotune --mode fast --directions backward --overwrite
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

<span style="color:#c77dff;">both</span> is the default. The default dtype set is <span style="color:#c77dff;">bf16</span>. For benchmarks such as <span style="color:#c77dff;">examples/perf.py</span> that run both <span style="color:#c77dff;">fp16</span> and <span style="color:#c77dff;">bf16</span>, generate both dtype configs explicitly:

```bash
python -m ffpa_attn.autotune --mode fast --directions both --dtypes bf16,fp16 --overwrite
```

On SM90+ devices, add <span style="color:#c77dff;">--enable-fwd-tma</span> or <span style="color:#c77dff;">--enable-bwd-tma</span> to additionally generate persistent configs for the descriptor/TMA forward or backward path when each task shape supports it. The baseline <span style="color:#c77dff;">fwd_generic</span> and <span style="color:#c77dff;">bwd_generic</span> configs are still generated as compatibility fallbacks. Add <span style="color:#c77dff;">--enable-fwd-ws</span> or <span style="color:#c77dff;">--enable-bwd-ws</span> with the matching TMA flag when you also want warp-specialized TMA candidates. The legacy <span style="color:#c77dff;">--enable-tma</span> and <span style="color:#c77dff;">--enable-ws</span> flags remain as aliases that enable both directions:

```bash
python -m ffpa_attn.autotune \
	--mode fast \
	--directions forward \
	--dtypes bf16,fp16 \
	--enable-fwd-tma \
	--enable-fwd-ws \
	--overwrite
```

### Output Location

The default output directory is the package config directory:

```text
src/ffpa_attn/triton/configs/
```

The file name is derived from <span style="color:#c77dff;">torch.cuda.get_device_name()</span> with non-file-name characters replaced by underscores. For example:

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

At runtime, FFPA can also load configs from a custom directory with <span style="color:#c77dff;">FFPA_TUNED_CONFIG_DIR</span>:

```bash
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke python your_script.py
```

## Autotune Modes

The generator supports the same mode names as the runtime Triton autotune path:

| Mode | Purpose |
| --- | --- |
| <span style="color:#c77dff;">fast</span> | Smaller search space. Recommended as the default persistent config mode. |
| <span style="color:#c77dff;">max</span> | Larger search space. Slower to generate, but may find better configs. |

The runtime lookup requires the mode to match. A JSON generated with <span style="color:#c77dff;">--mode fast</span> is used when <span style="color:#c77dff;">triton_autotune_mode="fast"</span>; a JSON generated with <span style="color:#c77dff;">--mode max</span> is used when <span style="color:#c77dff;">triton_autotune_mode="max"</span>.

## Shape Coverage

The generator tunes the following head dimensions:

```text
320, 512, 640, 768, 1024 # headdim
```

The sequence-length grid is:

```text
1, 512, 1024, 2048, 4096, 8192, 16384 # seqlen
```

The <span style="color:#c77dff;">1</span> entry is used only for decode query length (<span style="color:#c77dff;">Nq=1</span>`). Decode tuning does not generate <span style="color:#c77dff;">Nkv=1</span> cases because a single-token KV cache is not a meaningful decode-attention benchmark target.

The <span style="color:#c77dff;">16384</span> sequence length is generated only when the current GPU has at least <span style="color:#c77dff;">48 GiB</span> of memory. Smaller-memory devices skip it.

Persistent config generation tunes every target sequence length in this grid with exact Triton autotune keys. It does not reuse the online runtime seqlen buckets while generating JSON, so an entry for <span style="color:#c77dff;">512</span>, <span style="color:#c77dff;">1024</span>, or <span style="color:#c77dff;">2048</span> means that shape was benchmarked independently. Runtime lookup still performs reuse when the workload shape is not an exact persisted entry. The generated matrix covers:

| Direction | Kernels |
| --- | --- |
| Forward | generic forward, split-KV/decode stage1 |
| Backward | delta preprocess, main backward, decode backward stage1 |

Forward tasks include self-attention, cross-attention, decode attention (<span style="color:#c77dff;">Nq=1</span>), causal, and non-causal variants.

Backward tasks include main backward shapes (<span style="color:#c77dff;">Nq >= 512</span>) and decode backward
shapes (<span style="color:#c77dff;">Nq=1</span>, <span style="color:#c77dff;">Nkv>1</span>), with causal and non-causal variants.

When <span style="color:#c77dff;">--full-tasks</span> is enabled, the generator adds square prefill variants for
each tuned sequence length:

| Case | Shape / variant |
| --- | --- |
| <span style="color:#c77dff;">attn-mask</span> | Compact additive key-position mask <span style="color:#c77dff;">[1, 1, 1, Nkv]</span>. Backward tunes the bias-gradient path. |
| <span style="color:#c77dff;">dropout</span> | <span style="color:#c77dff;">dropout_p=0.1</span>, using the Triton dropout path. |
| <span style="color:#c77dff;">gqa</span> | <span style="color:#c77dff;">Hq=H</span>, <span style="color:#c77dff;">Hkv</span> chosen with the same divisor rule as <span style="color:#c77dff;">examples/perf.py</span>. |
| <span style="color:#c77dff;">mqa</span> | <span style="color:#c77dff;">Hq=H</span>, <span style="color:#c77dff;">Hkv=1</span>. |

These are single-feature canonical variants, not a full Cartesian product. For
example, <span style="color:#c77dff;">--full-tasks</span> tunes <span style="color:#c77dff;">gqa</span> and <span style="color:#c77dff;">dropout</span> separately, but it does not
generate a combined GQA+dropout case.

## Runtime Lookup Rules

When runtime autotune is disabled, FFPA tries to load the current device JSON. If no compatible entry is found, it falls back to the built-in default launch parameters. Forward lookup filters by:

| Field | Meaning |
| --- | --- |
| <span style="color:#c77dff;">direction</span> | Must be <span style="color:#c77dff;">forward</span>. |
| <span style="color:#c77dff;">kernel</span> | <span style="color:#c77dff;">fwd_generic</span> or <span style="color:#c77dff;">decode_fwd_stage1</span>. |
| <span style="color:#c77dff;">autotune_mode</span> | Must match <span style="color:#c77dff;">triton_autotune_mode</span>. |
| <span style="color:#c77dff;">dtype</span> | <span style="color:#c77dff;">fp16</span> or <span style="color:#c77dff;">bf16</span>. |
| <span style="color:#c77dff;">causal</span> | Must match <span style="color:#c77dff;">is_causal</span>. |
| <span style="color:#c77dff;">has_attn_bias</span> | Whether <span style="color:#c77dff;">attn_mask</span> / additive bias is present. |
| <span style="color:#c77dff;">has_dropout</span> | Whether dropout is active. |

Backward lookup additionally filters by:

| Field | Meaning |
| --- | --- |
| <span style="color:#c77dff;">kernel</span> | <span style="color:#c77dff;">bwd_preproc</span>, <span style="color:#c77dff;">bwd_generic</span>, or <span style="color:#c77dff;">decode_bwd_stage1</span>. |
| <span style="color:#c77dff;">preprocess_d_chunk</span> | Applies to the delta preprocess kernel. |
| <span style="color:#c77dff;">bias_grad</span> | Whether attention-bias gradients are requested. |
| <span style="color:#c77dff;">grad_v_storage_dtype</span> | Optional internal <span style="color:#c77dff;">dV</span> storage override. |
| <span style="color:#c77dff;">use_gemv</span> | Decode backward single-query specialization. |
| <span style="color:#c77dff;">has_dropout</span> | Whether dropout replay is active. |
| <span style="color:#c77dff;">has_attn_bias</span> | Whether additive bias is active. |

Generated entries may include <span style="color:#c77dff;">nheads_q</span> and <span style="color:#c77dff;">nheads_kv</span> for logging and JSON metadata. Runtime lookup can prefer an exact recorded head layout when one is available, but it does not require the head layout to match. Batch size and head count commonly vary across workloads, so FFPA reuses the same launch config across compatible mask/dropout/causal/kernel variants instead of missing the persistent config because <span style="color:#c77dff;">Hq</span> or <span style="color:#c77dff;">Hkv</span> changed.

Configs generated before these variant fields existed are treated as no-mask,
no-dropout entries. They can still satisfy baseline requests, while <span style="color:#c77dff;">has_attn_bias</span> and <span style="color:#c77dff;">has_dropout</span> continue to prevent semantically different kernel variants from being mixed.

After variant filtering, FFPA chooses the nearest persisted head dimension. Ties
prefer the larger candidate. Examples:

| Runtime <span style="color:#c77dff;">D</span> | Persisted <span style="color:#c77dff;">D</span> |
| --- | --- |
| 384 | 320 |
| 448 | 512 |
| 900 | 1024 |

For sequence length, FFPA chooses the smallest persisted sequence length that is
greater than or equal to the runtime value. If the runtime value is larger than
all persisted values, FFPA uses the largest available persisted value. Examples:

| Runtime <span style="color:#c77dff;">seqlen</span> | Persisted <span style="color:#c77dff;">seqlen</span> |
| --- | --- |
| 3000 | 4096 |
| 32768 | 8192 or 16384, depending on what was generated |

### Debug Persistent Lookup

Set <span style="color:#c77dff;">FFPA_LOGGER_LEVEL=DEBUG</span> when you want to verify that runtime calls are
using persistent tuned configs instead of falling back to the built-in defaults:

```bash
FFPA_LOGGER_LEVEL=DEBUG \
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke \
python examples/perf.py --case decode-attn --backend ffpa-triton
```

On repeated runtime lookup hits, FFPA logs the kernel name and sanitized launch config selected from the in-process persistent config cache. The message uses <span style="color:#c77dff;">debug_once</span> semantics, so the same cache-hit/config line is emitted once per process instead of repeating on every attention call.

## Development Smoke Tests

Full autotune generation can take a long time. Use <span style="color:#c77dff;">FFPA_AUTOTUNE_MAX_CONFIGS</span> to cap the number of shapes during development:

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

To compare persistent tuned configs against the built-in fallback launch defaults without removing the JSON, force runtime lookup to bypass persisted entries:

```bash
CUDA_VISIBLE_DEVICES=0 \
FFPA_TUNED_CONFIG_DIR=/tmp/ffpa-config-smoke \
FFPA_SKIP_PERSISIT_TUNED_CONFIG=1 \
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

3. Commit the generated JSON under <span style="color:#c77dff;">src/ffpa_attn/triton/configs/</span>.
4. Run normal workloads with runtime autotune disabled, which is the default:

```python
out = ffpa_attn_func(q, k, v)
```

If the runtime shape is outside the generated grid, FFPA uses the nearest compatible persisted config. If the device JSON is missing, malformed, or does not contain a compatible entry, FFPA silently keeps the existing built-in launch defaults.

## Current Scope and Limitations

The first persistent-config generator focuses on the common no-bias and no-dropout path. The JSON schema already records <span style="color:#c77dff;">bias_grad</span>, <span style="color:#c77dff;">has_dropout</span>, and <span style="color:#c77dff;">grad_v_storage_dtype</span>, so bias-gradient and dropout-specific configs can be added later without changing the runtime lookup design.

<span style="color:#c77dff;">decode_dq_reduce</span> and key-bias gradient reduction are not currently autotuned by FFPA, so they keep their fixed launch parameters. Persistent configs are device-specific. Do not reuse a JSON generated on one GPU class as a performance baseline for a different GPU class.
