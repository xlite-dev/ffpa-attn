#!/usr/bin/env bash
# Fast build wrapper for ffpa-attn. Combines the build-speed optimizations:
#
#   1. ccache-wrapped nvcc shim (caches .cu TUs across clean rebuilds).
#   2. ccache for host g++ via CC/CXX.
#   3. MAX_JOBS auto-sized to physical cores (capped at 32) unless preset.
#   4. FFPA_NVCC_THREADS=4 by default (the env default).
#   5. Optional tmpfs build dir (--shm / FFPA_BUILD_IN_SHM=1).
#   6. Pre-set FFPA_BUILD_ARCH to the current device SM when unset.
#
# Compatible with the PEP 621 pyproject.toml packaging: by default the script
# bypasses pip's build isolation and invokes setup.py directly for the in-place
# CUDA extension build. Pass --editable to register the package as an editable
# install while reusing the same build environment. For an isolated PEP 517
# wheel use ``pip wheel . --no-build-isolation`` instead.
#
# Usage (run with --help for all flags; flags map onto the FFPA_* /
# ENABLE_FFPA_* env vars and override same-named env vars):
#   bash tools/build_fast.sh                                   # ext=cuda default
#   bash tools/build_fast.sh --arch sm_120f --ext all --editable --headdim all --jobs 32
#   bash tools/build_fast.sh --arch sm_89,sm_120f              # multi-arch
#   bash tools/build_fast.sh --clean --headdim 256,512         # fast iteration
#   bash tools/build_fast.sh bdist_wheel                       # PEP 517-compatible wheel
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

usage() {
  cat <<EOF
Usage: bash tools/build_fast.sh [flags] [pass-through args]

Flags (override same-named env vars; env reference: docs/env.md):
  --arch <spec>        FFPA_BUILD_ARCH: SM list. Full names (sm_120f, sm_89,
                       sm_90a), short forms (120f, 89), or aliases (ada,
                       hopper, blackwell_geforce). Comma-separate or repeat
                       the flag for multiple archs, e.g. --arch sm_89,sm_120f
                       or --arch sm_89 --arch sm_120f.
  --ext <exts>         CUDA extension switches (ENABLE_FFPA_CUDA_IMPL /
                       CUTE_EXT / TMA_EXT):
                         all    cuda + cute + tma
                         none   all off (Triton/CuTeDSL-python only build)
                         <csv>  subset of cuda,cute,tma (cute/tma imply cuda)
                       Default without --ext or env: ENABLE_FFPA_CUDA_IMPL=1.
                       TMA auto-disables when every target arch is sm<90.
  --headdim <list|all> FFPA_DEV_HEADDIMS subset, e.g. 256,512; 'all' (or
                       omitting the flag) builds the full headdim set.
  --editable           FFPA_EDITABLE=1: pip install -e instead of build_ext.
  -j, --jobs N         MAX_JOBS outer build parallelism (default min(nproc,32)).
  --clean              FFPA_CLEAN=1: rm build/, *.so, generated TUs first.
  --shm                FFPA_BUILD_IN_SHM=1: build dir on tmpfs.
  --nvcc-threads N     FFPA_NVCC_THREADS intra-TU nvcc threads (default 4).
  --ptxas-verbose      FFPA_PTXAS_VERBOSE=1: dump register/smem usage.
  --dry-run            print resolved env + command, exit before side effects.
  -h, --help           show this help.

Unknown args pass through to 'setup.py build_ext --inplace' or
'pip install -e .', e.g.: bash tools/build_fast.sh bdist_wheel --verbose
EOF
  exit 0
}

fail_usage() {
  echo "[build_fast] error: $1" >&2
  echo "Run with --help for usage." >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || fail_usage "flag '$1' requires a value"
}

# CLI parsing: known flags map onto FFPA_*/ENABLE_FFPA_* env vars; unknown
# args are passed through to setup.py build_ext / pip install -e.
PASS_ARGS=()
DRY_RUN=0
ARCH_SET=0
EXT_SET=0
HEADDIM_SET=0
EXT_CUDA=0
EXT_CUTE=0
EXT_TMA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      require_value "$@"
      # repeatable: multiple --arch flags accumulate comma-separated
      if [[ "$ARCH_SET" == "1" ]]; then
        export FFPA_BUILD_ARCH="$FFPA_BUILD_ARCH,$2"
      else
        export FFPA_BUILD_ARCH="$2"
        ARCH_SET=1
      fi
      shift 2 ;;
    --ext)
      require_value "$@"
      EXT_SET=1
      IFS=', ' read -r -a _ext_toks <<< "$2"
      for tok in "${_ext_toks[@]}"; do
        tok="${tok,,}"
        [[ -z "$tok" ]] && continue
        case "$tok" in
          all)  EXT_CUDA=1; EXT_CUTE=1; EXT_TMA=1 ;;
          none) EXT_CUDA=0; EXT_CUTE=0; EXT_TMA=0 ;;
          cuda) EXT_CUDA=1 ;;
          cute) EXT_CUTE=1 ;;
          tma)  EXT_TMA=1 ;;
          *) fail_usage "invalid --ext token '$tok' (allowed: all, none, cuda, cute, tma)" ;;
        esac
      done
      shift 2 ;;
    --headdim)
      require_value "$@"
      HEADDIM_SET=1
      if [[ "${2,,}" == "all" ]]; then
        unset FFPA_DEV_HEADDIMS
      else
        export FFPA_DEV_HEADDIMS="$2"
      fi
      shift 2 ;;
    -j|--jobs)
      require_value "$@"
      [[ "$2" =~ ^[0-9]+$ ]] || fail_usage "--jobs expects a positive integer, got '$2'"
      export MAX_JOBS="$2"
      shift 2 ;;
    --nvcc-threads)
      require_value "$@"
      [[ "$2" =~ ^[0-9]+$ ]] || fail_usage "--nvcc-threads expects a positive integer, got '$2'"
      export FFPA_NVCC_THREADS="$2"
      shift 2 ;;
    --editable)      export FFPA_EDITABLE=1; shift ;;
    --clean)         export FFPA_CLEAN=1; shift ;;
    --shm)           export FFPA_BUILD_IN_SHM=1; shift ;;
    --ptxas-verbose) export FFPA_PTXAS_VERBOSE=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage ;;
    *)               PASS_ARGS+=("$1"); shift ;;
  esac
done

# headdim omitted -> default to the full headdim set.
if [[ "$HEADDIM_SET" == "0" ]]; then
  unset FFPA_DEV_HEADDIMS
fi

# Resolve --ext into ENABLE_FFPA_* switches (cute/tma live inside the _C ext).
if [[ "$EXT_SET" == "1" ]]; then
  if [[ "$EXT_CUTE" == "1" || "$EXT_TMA" == "1" ]]; then
    EXT_CUDA=1
  fi
  export ENABLE_FFPA_CUDA_IMPL="$EXT_CUDA"
  export ENABLE_FFPA_CUTE_EXT="$EXT_CUTE"
  export ENABLE_FFPA_TMA_EXT="$EXT_TMA"
  unset ENABLE_FFPA_FWD_CUDA_IMPL
  if [[ "$EXT_CUTE" == "1" && "$EXT_TMA" == "0" ]]; then
    echo "[build_fast] warning: cute without tma only enables the SM80 cute path;" \
         "the SM120 cute kernel needs --ext cute,tma (or all)." >&2
  fi
elif [[ -z "${ENABLE_FFPA_CUDA_IMPL:-}" ]]; then
  if [[ -z "${ENABLE_FFPA_FWD_CUDA_IMPL:-}" ]]; then
    echo "[build_fast] no --ext flag or ENABLE_FFPA_CUDA_IMPL env; defaulting ENABLE_FFPA_CUDA_IMPL=1"
    export ENABLE_FFPA_CUDA_IMPL=1
  elif [[ "${ENABLE_FFPA_TMA_EXT:-0}" == "1" || "${ENABLE_FFPA_CUTE_EXT:-0}" == "1" ]]; then
    echo "[build_fast] ENABLE_FFPA_CUTE_EXT/TMA_EXT set without ENABLE_FFPA_CUDA_IMPL; forcing ENABLE_FFPA_CUDA_IMPL=1"
    export ENABLE_FFPA_CUDA_IMPL=1
  fi
fi

# TMA kernels need sm>=90 (native/tma.cuh is guarded by __CUDA_ARCH__>=900)
# and the macro is global across all -gencode passes, so auto-disable
# ENABLE_FFPA_TMA_EXT whenever any target arch is below 90.
if [[ "${ENABLE_FFPA_TMA_EXT:-0}" == "1" ]]; then
  _arch_raw="${FFPA_BUILD_ARCH:-}"
  if [[ -z "$_arch_raw" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    [[ -n "${_cc:-}" ]] && _arch_raw="$_cc"
  fi
  _any_below_90=0
  if [[ -n "$_arch_raw" ]]; then
    IFS=',; ' read -r -a _arch_toks <<< "$_arch_raw"
    for tok in "${_arch_toks[@]}"; do
      tok="${tok,,}"
      tok="${tok%+ptx}"
      tok="${tok#sm_}"
      tok="${tok#compute_}"
      tok="${tok//./}"
      case "$tok" in
        maxwell) tok=50 ;; pascal) tok=60 ;; volta) tok=70 ;; turing) tok=75 ;;
        ampere) tok=80 ;; ada) tok=89 ;; hopper) tok=90 ;;
        blackwell) tok=100 ;; blackwell_geforce) tok=120 ;;
      esac
      num="${tok%%[!0-9]*}"
      [[ -z "$num" ]] && continue
      if (( 10#$num < 90 )); then _any_below_90=1; break; fi
    done
  fi
  if [[ "$_any_below_90" == "1" ]]; then
    echo "[build_fast] target archs include sm<90; disabling ENABLE_FFPA_TMA_EXT" \
         "(TMA needs sm>=90; cute falls back to the SM80 cp.async path)"
    export ENABLE_FFPA_TMA_EXT=0
  fi
fi

# MAX_JOBS auto-size (capped at 32) unless preset by --jobs or env.
if [[ -z "${MAX_JOBS:-}" ]]; then
  NCORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 8)"
  if (( NCORES > 32 )); then NCORES=32; fi
  export MAX_JOBS="$NCORES"
fi

# nvcc intra-TU threads (respect --nvcc-threads/env, else env.py default 4).
export FFPA_NVCC_THREADS="${FFPA_NVCC_THREADS:-4}"

# Resolved configuration; --dry-run exits here before any side effect.
if [[ "${FFPA_EDITABLE:-0}" == "1" ]]; then
  BUILD_CMD="python -m pip install -e . --no-build-isolation --no-deps"
else
  BUILD_CMD="python setup.py build_ext --inplace"
fi
echo "[build_fast] ENABLE_FFPA_CUDA_IMPL=${ENABLE_FFPA_CUDA_IMPL:-0}  ENABLE_FFPA_CUTE_EXT=${ENABLE_FFPA_CUTE_EXT:-0}  ENABLE_FFPA_TMA_EXT=${ENABLE_FFPA_TMA_EXT:-0}"
echo "[build_fast] FFPA_BUILD_ARCH=${FFPA_BUILD_ARCH:-<auto from current device>}  FFPA_DEV_HEADDIMS=${FFPA_DEV_HEADDIMS:-<full set>}  FFPA_EDITABLE=${FFPA_EDITABLE:-0}"
echo "[build_fast] MAX_JOBS=$MAX_JOBS  FFPA_NVCC_THREADS=$FFPA_NVCC_THREADS"
echo "[build_fast] command: $BUILD_CMD${PASS_ARGS[*]:+ ${PASS_ARGS[*]}}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[build_fast] dry-run: exiting before any clean/build side effects."
  exit 0
fi

# third_party/cutlass headers are required includes; sync the submodule if missing.
if [[ ! -e third_party/cutlass/include ]]; then
  echo "[build_fast] third_party/cutlass missing; initializing submodule"
  git submodule update --init --recursive third_party/cutlass
fi

# Optional clean (must run BEFORE we materialize the shadow CUDA_HOME under
# build/, otherwise the shim directory is wiped).
if [[ "${FFPA_CLEAN:-0}" == "1" ]]; then
  echo "[build_fast] FFPA_CLEAN=1 -> removing build/ and *.so"
  rm -rf build/ dist/ src/ffpa_attn.egg-info/ __pycache__
  rm -f pyffpa_cuda*.so src/ffpa_attn/_C*.so
  find csrc/cuffpa/generated -maxdepth 1 -type f \( -name '*.cu' -o -name '*.h' \) -delete 2>/dev/null || true
fi

# ccache shim for nvcc + host compiler wrapping.
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

# Optional tmpfs build dir.
if [[ "${FFPA_BUILD_IN_SHM:-0}" == "1" ]]; then
  SHM_BUILD="/dev/shm/ffpa-build-$USER"
  mkdir -p "$SHM_BUILD"
  if [[ -e build && ! -L build ]]; then rm -rf build; fi
  ln -sfn "$SHM_BUILD" build
  echo "[build_fast] build/ -> $SHM_BUILD (tmpfs)"
fi

T0=$(date +%s)
if [[ "${FFPA_EDITABLE:-0}" == "1" ]]; then
  echo "[build_fast] editable mode: $BUILD_CMD"
  python -m pip install -e . --no-build-isolation --no-deps ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
else
  python setup.py build_ext --inplace ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
fi
T1=$(date +%s)
echo "[build_fast] total elapsed: $((T1-T0))s"

# Usage guide
#
# CLI flags (override same-named env vars; see docs/env.md for the full table):
#   bash tools/build_fast.sh --arch sm_120f --ext all --editable --headdim all --jobs 32
#     == FFPA_BUILD_ARCH=sm_120f ENABLE_FFPA_CUDA_IMPL=1 ENABLE_FFPA_CUTE_EXT=1 \
#        ENABLE_FFPA_TMA_EXT=1 FFPA_EDITABLE=1 MAX_JOBS=32 bash tools/build_fast.sh
#
#   bash tools/build_fast.sh                      # incremental build; note the
#                                                 # default now sets ENABLE_FFPA_CUDA_IMPL=1
#   bash tools/build_fast.sh --ext none           # Triton/CuTeDSL-python only
#   bash tools/build_fast.sh --clean              # rm build/ + rebuild
#   bash tools/build_fast.sh --headdim 256,512    # headdim subset, fast iteration
#   bash tools/build_fast.sh --shm                # tmpfs build dir
#   bash tools/build_fast.sh --ptxas-verbose      # dump register/smem usage
#   bash tools/build_fast.sh --dry-run            # resolved env + command only
#   bash tools/build_fast.sh bdist_wheel          # pass-through to setup.py/pip
#
# Measured on L20 (66 TUs = 25 headdims x 2 dtype + dispatch TU):
#   baseline cold (MAX_JOBS=32, no ccache) : ~207s
#   ccache cold   (first-time populate)    : ~214s
#   ccache warm   (clean rebuild, 65/65 hit): ~23s         (~9x speedup)
#   subset cold   (--headdim 256,512)      : ~48s
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
# Editable mode notes
#   - Use --editable once to install the package as editable.
#   - After that, Python-only changes under src/ffpa_attn/ are picked up
#     immediately without rerunning this script.
#   - C++/CUDA source changes still require rerunning this script.
#
# Troubleshooting
#   - `nvcc fatal: Unknown option '-diag-suppress 177'` => ensure env.py
#     splits it into two argv entries (already fixed).
#   - First clean build doesn't speed up: that's expected — it populates the
#     cache. The second clean build should hit 100% and finish in ~20-30s.
#   - After upgrading CUDA major version: run `ccache -C` to drop stale hits.
#   - DO NOT wrap CC/CXX with ccache: torch.cpp_extension checks the compiler
#     name and will refuse because libtorch wasn't built with ccache.
