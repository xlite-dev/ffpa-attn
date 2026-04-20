# Configurable Environment Variables

This document summarizes all configurable environment variables in **ffpa-attn**, which control the kernel build (target SM list, headdim coverage, ccache shim) and the runtime kernel selection knobs (MMA accumulator dtype, SMEM swizzle, prefetch / persist policies, launch grid layout). Most runtime knobs are **boolean integers** parsed from `0` / `1`.

## Build-time environment variables

These are read once during `pip install .` / `python setup.py build_ext` and decide which translation units are generated and how nvcc is invoked.

- <span style="color:#c77dff;">FFPA_BUILD_ARCH</span>, default `""` (current device SM), Comma/semicolon/space separated list of target CUDA SM architectures. Accepts numeric SMs (e.g. `"80,89,90"`) or aliases (`maxwell`, `pascal`, `volta`, `turing`, `ampere`, `ada`, `hopper`, `blackwell`, `blackwell_geforce`). When empty, falls back to the currently visible CUDA device's compute capability.
- <span style="color:#c77dff;">FFPA_SKIP_CUDA_EXT</span>, default `False (0)`, Skip the CUDA extension build entirely and install only the Python wrapper. Used by ReadTheDocs and the `check-mkdocs` GitHub workflow where no nvcc is available; mkdocstrings can still import `ffpa_attn` to extract docstrings.
- <span style="color:#c77dff;">FFPA_NVCC_THREADS</span>, default `4`, nvcc intra-TU parallelism (`--threads N`). With the per-headdim TU split, the outer `MAX_JOBS` already drives many nvcc processes in parallel, so keeping `--threads` small avoids oversubscription. Set to `1` to disable intra-TU threading entirely; larger values only help when `MAX_JOBS` is small.
- <span style="color:#c77dff;">FFPA_PTXAS_VERBOSE</span>, default `False (0)`, Emit ptxas verbose info (register / SMEM usage). Off by default because it produces tens of MB of log output; enable only for tuning.
- <span style="color:#c77dff;">FFPA_DEV_HEADDIMS</span>, default `""`, Development-time headdim subset override. Comma/space separated list of headdims (e.g. `"256,512"`) that replaces the full generated set for fast iteration. Empty (default) means use the full set decided by `ENABLE_FFPA_ALL_HEADDIM`.
- <span style="color:#c77dff;">ENABLE_FFPA_ALL_STAGES</span>, default `True (1)`, When `1`, generate kernels for all multi-stage variants (stages `1~4`); when `0`, only stages `1~2` are generated. Reducing this shortens build time at the cost of fewer schedule choices at runtime.
- <span style="color:#c77dff;">ENABLE_FFPA_ALL_HEADDIM</span>, default `True (1)`, When `1`, headdims range from `32` to `1024` with step `32` (`range(32, 1024, 32)`); when `0`, headdims range from `256` to `1024` with step `64` (`range(256, 1024, 64)`).
- <span style="color:#c77dff;">MAX_JOBS</span>, default `min(nproc, 32)` via `tools/build_fast.sh`, Outer build parallelism passed to setuptools. The fast-build wrapper auto-caps at 32; for plain `python setup.py build_ext`, set it explicitly (e.g. `MAX_JOBS=32`).

### `tools/build_fast.sh`-only variables

These only affect the ccache-based fast-build wrapper.

- <span style="color:#c77dff;">FFPA_CLEAN</span>, default `0`, When `1`, removes `build/`, `dist/`, `ffpa_attn.egg-info/`, compiled `*.so`, and generated `csrc/cuffpa/generated/*.{cu,h}` before rebuilding.
- <span style="color:#c77dff;">FFPA_BUILD_IN_SHM</span>, default `0`, When `1`, symlinks `build/` into `/dev/shm/ffpa-build-$USER` (tmpfs) for IO-bound machines.
- <span style="color:#c77dff;">CCACHE_MAXSIZE</span>, default `20G`, Cap of the ccache storage used by the nvcc shim.
- <span style="color:#c77dff;">CCACHE_DIR</span>, default `~/.ccache`, ccache storage directory.
- <span style="color:#c77dff;">NVCC_REAL</span>, default `$CUDA_HOME/bin/nvcc`, Override the path to the real nvcc (advanced; rarely needed).

## Runtime kernel-selection environment variables

These are read by `env.py` and gate which generated kernel template is dispatched at import / call time. All are booleans (`0` / `1`) unless noted.

### MMA accumulator dtype

- <span style="color:#c77dff;">ENABLE_FFPA_FORCE_QK_F16</span>, default `False (0)`, Force `Q@K^T` MMA accumulator to FP16 within the FFPA Acc-F32 kernels. Enables the mixed mode `Q@K^T MMA Acc F16 + P@V MMA Acc F32`.
- <span style="color:#c77dff;">ENABLE_FFPA_FORCE_PV_F16</span>, default `False (0)`, Force `P@V` MMA accumulator to FP16 within the FFPA Acc-F32 kernels. Enables the mixed mode `Q@K^T MMA Acc F32 + P@V MMA Acc F16`.

### Prefetch & SMEM-share policies

- <span style="color:#c77dff;">ENABLE_FFPA_PREFETCH_QKV</span>, default `True (1)`, Prefetch QKV at the appropriate time point. Typical boost is `5%~10%`.
- <span style="color:#c77dff;">ENABLE_FFPA_QKV_SMEM_SHARE</span>, default `False (0)`, Use a shared QKV SMEM policy. Off by default because separate buffers overlap better with MMA / g2s; turn on for low-SRAM devices.

### SMEM swizzle (vs. padding) for Q / K / V

- <span style="color:#c77dff;">ENABLE_FFPA_SMEM_SWIZZLE_Q</span>, default `True (1)`, `True`: bank-conflict-free Q SMEM via swizzle. `False`: bank-conflict-free via padding.
- <span style="color:#c77dff;">ENABLE_FFPA_SMEM_SWIZZLE_K</span>, default `True (1)`, Same as above for K SMEM.
- <span style="color:#c77dff;">ENABLE_FFPA_SMEM_SWIZZLE_V</span>, default `True (1)`, Same as above for V SMEM.

### Persistent g2s / s2r loads

- <span style="color:#c77dff;">ENABLE_FFPA_PERSIST_Q_G2S</span>, default `True (1)`, Persistently keep Q in SMEM via g2s for headdim `<= 320`. Trades more SRAM for fewer global loads while keeping register usage stable.
- <span style="color:#c77dff;">ENABLE_FFPA_PERSIST_KV_G2S</span>, default `True (1)`, Persistently keep KV in SMEM via g2s for headdim `<= 256`. When enabled, FFPA auto-uses the FlashAttention attention-level tiling for `headdim <= 256` and the FFPA fine-grained MMA-level tiling for `headdim > 256`.
- <span style="color:#c77dff;">ENABLE_FFPA_PERSIST_Q_S2R</span>, default `False (0)`, Persistently load Q s2r for headdim `< 512` to reduce Q g2s/s2r IO while preserving O(1) SRAM complexity. Adds register pressure as headdim grows; weigh register usage vs. IO reduction before enabling.
- <span style="color:#c77dff;">ENABLE_FFPA_PERSIST_V_S2R</span>, default `True (1)`, Persistently load V s2r for the small-d kernel only (more registers).

### Pipelining and launch grid layout

- <span style="color:#c77dff;">ENABLE_FFPA_REGISTERS_PIPE_KV</span>, default `False (0)`, Use register-level ping-pong double buffers for `ldmatrix` / MMA computation overlap.
- <span style="color:#c77dff;">ENABLE_FFPA_LAUNCH_GRID_DNHB</span>, default `False (0)`, When `1`, launch with `grid(N/Br, H, B)`; when `0`, launch with `grid(N/Br, B * H)`.

## Key notes

1. **Boolean parsing**: all `ENABLE_FFPA_*` and `FFPA_PTXAS_VERBOSE` are parsed from integer values: `1` = `True` (enabled), `0` = `False` (disabled).
2. **Build-time vs. runtime separation**: `ENABLE_FFPA_ALL_STAGES` and `ENABLE_FFPA_ALL_HEADDIM` decide which TUs are generated at build time — changing them requires rebuilding (`FFPA_CLEAN=1 bash tools/build_fast.sh`). All other `ENABLE_FFPA_*` knobs are read at runtime and can be toggled without rebuilding.
3. **Default arch inference**: when `FFPA_BUILD_ARCH` is unset, `setup.py` queries the currently visible CUDA device's compute capability. On CI / RTD runners with no GPU, set `FFPA_SKIP_CUDA_EXT=1` (or set `FFPA_BUILD_ARCH` explicitly).
4. **MMA Acc combinations**: `ENABLE_FFPA_FORCE_QK_F16` and `ENABLE_FFPA_FORCE_PV_F16` are mutually exclusive options for the FFPA Acc-F32 kernels — enable at most one to choose the mixed-precision split.
5. **Recommended for tuning**: enable `FFPA_PTXAS_VERBOSE=1` (build-time) and start from the defaults; toggle one runtime knob at a time and measure with `bench/bench_ffpa_attn.py` to attribute the speedup or regression.
