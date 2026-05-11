#!/usr/bin/env bash
# Fast build wrapper for ffpa-attn. Combines the build-speed optimizations:
#
#   1. ccache-wrapped nvcc shim (caches .cu TUs across clean rebuilds).
#   2. ccache for host g++ via CC/CXX.
#   3. MAX_JOBS auto-sized to physical cores (capped at 32) unless preset.
#   4. FFPA_NVCC_THREADS=4 by default (the env default).
#   5. Optional tmpfs build dir when FFPA_BUILD_IN_SHM=1.
#   6. Pre-set FFPA_BUILD_ARCH to the current device SM when unset.
#
# Compatible with the PEP 621 pyproject.toml packaging: by default the script
# bypasses pip's build isolation and invokes setup.py directly for the in-place
# CUDA extension build. Set ``FFPA_EDITABLE=1`` to register the package as an
# editable install while reusing the same build environment. For an isolated
# PEP 517 wheel use ``pip wheel . --no-build-isolation`` instead.
#
# Usage:
#   bash tools/build_fast.sh                   # incremental in-place build
#   FFPA_EDITABLE=1 bash tools/build_fast.sh   # editable install + extension build
#   FFPA_CLEAN=1 bash tools/build_fast.sh      # rm -rf build/ + rebuild
#   FFPA_BUILD_IN_SHM=1 bash tools/build_fast.sh
#   bash tools/build_fast.sh bdist_wheel       # PEP 517-compatible wheel
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Optional clean (must run BEFORE we materialize the shadow CUDA_HOME under
# build/, otherwise the shim directory is wiped).
if [[ "${FFPA_CLEAN:-0}" == "1" ]]; then
  echo "[build_fast] FFPA_CLEAN=1 -> removing build/ and *.so"
  rm -rf build/ dist/ src/ffpa_attn.egg-info/ __pycache__
  rm -f pyffpa_cuda*.so src/ffpa_attn/_C*.so
  find csrc/cuffpa/generated -maxdepth 1 -type f \( -name '*.cu' -o -name '*.h' \) -delete 2>/dev/null || true
fi

# 1+2. ccache shim for nvcc + host compiler wrapping.
# torch.utils.cpp_extension resolves nvcc as ``$CUDA_HOME/bin/nvcc`` directly
# (not via PATH). We therefore materialize a shadow CUDA_HOME that reuses
# the real toolkit's layout but replaces ``bin/nvcc`` with our ccache shim.
if command -v ccache >/dev/null 2>&1; then
  REAL_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  if [[ ! -x "$REAL_CUDA_HOME/bin/nvcc" ]]; then
    echo "[build_fast] real nvcc not found under $REAL_CUDA_HOME/bin; disabling ccache for nvcc." >&2
  else
    SHADOW_CUDA="$REPO_DIR/build/.ccache_cuda_home"
    mkdir -p "$SHADOW_CUDA/bin"
    # Symlink every top-level entry from real CUDA_HOME except bin/ itself.
    for entry in "$REAL_CUDA_HOME"/*; do
      name="$(basename "$entry")"
      [[ "$name" == "bin" ]] && continue
      ln -sfn "$entry" "$SHADOW_CUDA/$name"
    done
    # Shadow bin/: symlink every tool except nvcc, then install our wrapper.
    for entry in "$REAL_CUDA_HOME/bin"/*; do
      name="$(basename "$entry")"
      [[ "$name" == "nvcc" ]] && continue
      ln -sfn "$entry" "$SHADOW_CUDA/bin/$name"
    done
    chmod +x "$REPO_DIR/tools/nvcc"
    cp -f "$REPO_DIR/tools/nvcc" "$SHADOW_CUDA/bin/nvcc"
    export NVCC_REAL="$REAL_CUDA_HOME/bin/nvcc"
    export CUDA_HOME="$SHADOW_CUDA"
    export CUDA_PATH="$SHADOW_CUDA"
    export PATH="$SHADOW_CUDA/bin:$PATH"
    echo "[build_fast] nvcc ccache shim active: CUDA_HOME=$CUDA_HOME (real nvcc=$NVCC_REAL)"
  fi
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
  echo "[build_fast] ccache enabled for nvcc"
else
  echo "[build_fast] ccache not found; host+nvcc caching disabled." >&2
fi

# 3. MAX_JOBS auto-size (capped at 32).
if [[ -z "${MAX_JOBS:-}" ]]; then
  NCORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 8)"
  if (( NCORES > 32 )); then NCORES=32; fi
  export MAX_JOBS="$NCORES"
fi

# 4. nvcc intra-TU threads (respect existing env, otherwise keep env.py default).
export FFPA_NVCC_THREADS="${FFPA_NVCC_THREADS:-4}"

# 5. Optional tmpfs build dir.
if [[ "${FFPA_BUILD_IN_SHM:-0}" == "1" ]]; then
  SHM_BUILD="/dev/shm/ffpa-build-$USER"
  mkdir -p "$SHM_BUILD"
  if [[ -e build && ! -L build ]]; then rm -rf build; fi
  ln -sfn "$SHM_BUILD" build
  echo "[build_fast] build/ -> $SHM_BUILD (tmpfs)"
fi

# 6. Default FFPA_BUILD_ARCH = current device SM (env.py already handles this,
# but we print it for visibility).
if [[ -z "${FFPA_BUILD_ARCH:-}" ]]; then
  echo "[build_fast] FFPA_BUILD_ARCH unset; env.py will infer from current CUDA device."
fi

echo "[build_fast] MAX_JOBS=$MAX_JOBS  FFPA_NVCC_THREADS=$FFPA_NVCC_THREADS"
T0=$(date +%s)
if [[ "${FFPA_EDITABLE:-0}" == "1" ]]; then
  echo "[build_fast] editable mode: python -m pip install -e . --no-build-isolation --no-deps"
  python -m pip install -e . --no-build-isolation --no-deps "$@"
else
  python setup.py build_ext --inplace "$@"
fi
T1=$(date +%s)
echo "[build_fast] total elapsed: $((T1-T0))s"

# -----------------------------------------------------------------------------
# Usage guide
# -----------------------------------------------------------------------------
# Measured on L20 (66 TUs = 25 headdims x 2 dtype + dispatch TU):
#   baseline cold (MAX_JOBS=32, no ccache) : ~207s
#   ccache cold   (first-time populate)    : ~214s
#   ccache warm   (clean rebuild, 65/65 hit): ~23s         (~9x speedup)
#   subset cold   (FFPA_DEV_HEADDIMS=256,512): ~48s
#
# Core mechanism
#   - tools/nvcc is a ccache shim that execs `ccache <real_nvcc> "$@"`.
#   - torch's CUDAExtension resolves nvcc as `$CUDA_HOME/bin/nvcc` (not via
#     PATH), so we materialize a shadow CUDA_HOME under
#     build/.ccache_cuda_home/: every top-level entry from the real CUDA
#     toolkit is symlinked in, and only `bin/nvcc` is replaced by our shim.
#   - ccache caches on preprocessed TU content + nvcc argv, so `rm -rf build/`
#     + rebuild with identical flags hits ~100% after the first populate.
#
# Common invocations
#   # Daily development (MAX_JOBS auto = min(nproc, 32), --threads=4, warm cache)
#   bash tools/build_fast.sh
#
#   # First-time editable install: later Python-only edits need no rebuild
#   FFPA_EDITABLE=1 bash tools/build_fast.sh
#
#   # Full clean rebuild (after touching launch_templates.cuh / ffpa_attn_fwd.cuh)
#   FFPA_CLEAN=1 bash tools/build_fast.sh
#
#   # Fast iteration: only build a headdim subset
#   FFPA_CLEAN=1 FFPA_DEV_HEADDIMS=256,512 bash tools/build_fast.sh
#
#   # Multi-SM build (aliases or numeric SMs both accepted)
#   FFPA_BUILD_ARCH="ampere,ada" bash tools/build_fast.sh
#   FFPA_BUILD_ARCH="80,89,90"   bash tools/build_fast.sh
#
#   # tmpfs build dir for IO-bound machines
#   FFPA_BUILD_IN_SHM=1 bash tools/build_fast.sh
#
#   # Dump ptxas register/smem usage (for tuning)
#   FFPA_PTXAS_VERBOSE=1 bash tools/build_fast.sh
#
#   # Pass-through args to setup.py build_ext / pip install -e
#   bash tools/build_fast.sh --verbose
#
# Editable mode notes
#   - Use FFPA_EDITABLE=1 once to install the package as editable.
#   - After that, Python-only changes under src/ffpa_attn/ are picked up
#     immediately without rerunning this script.
#   - C++/CUDA source changes still require rerunning this script.
#
# Environment knobs honored by this script / env.py
#   MAX_JOBS           : outer build parallelism (default = min(nproc, 32))
#   FFPA_NVCC_THREADS  : nvcc intra-TU threads (default 4; was hard-coded 8)
#   FFPA_PTXAS_VERBOSE : 1 -> enable `--ptxas-options=-v -Xptxas -v` (default 0)
#   FFPA_DEV_HEADDIMS  : explicit headdim subset, e.g. "256,512" (default empty)
#   FFPA_BUILD_ARCH    : SM list or aliases; empty -> current device SM
#   FFPA_EDITABLE      : 1 -> `python -m pip install -e . --no-build-isolation --no-deps`
#   FFPA_CLEAN         : 1 -> rm build/, *.so, csrc/cuffpa/generated/*.{cu,h}
#   FFPA_BUILD_IN_SHM  : 1 -> symlink build/ to /dev/shm/ffpa-build-$USER
#   CCACHE_MAXSIZE     : ccache size cap (default 20G)
#   CCACHE_DIR         : ccache storage dir (default ~/.ccache)
#   NVCC_REAL          : override path to the real nvcc (advanced)
#
# Troubleshooting
#   - `nvcc fatal: Unknown option '-diag-suppress 177'` => ensure env.py
#     splits it into two argv entries (already fixed).
#   - First clean build doesn't speed up: that's expected — it populates the
#     cache. The second clean build should hit 100% and finish in ~20-30s.
#   - After upgrading CUDA major version: run `ccache -C` to drop stale hits.
#   - DO NOT wrap CC/CXX with ccache: torch.cpp_extension checks the compiler
#     name and will refuse because libtorch wasn't built with ccache.
# -----------------------------------------------------------------------------
