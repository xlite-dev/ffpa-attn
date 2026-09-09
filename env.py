import os
import re
import shutil

import torch

# Alias table mirroring cache-dit/setup.py::CUDA_ARCH_ALIASES so users may set
# FFPA_BUILD_ARCH to either numeric SMs or architecture names.
_ARCH_ALIASES = {
  "maxwell": "50",
  "pascal": "60",
  "volta": "70",
  "turing": "75",
  "ampere": "80",
  "ada": "89",
  "hopper": "90",
  "blackwell": "100",
  "blackwell_geforce":
  "120f",  # sm_120f need for TMA & setmaxnreg instructions.
}


class ENV(object):
  # ENVs for FFPA kernels compiling

  # Project dir, path to faster-prefill-attention
  PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

  # Enable all multi stages kernels or not, if True (1~N) else (1~2), default True.
  # FFPA_BUILD_MAX_STAGES controls N, default 8.
  ENABLE_FFPA_ALL_STAGES = bool(
    int(os.environ.get("ENABLE_FFPA_ALL_STAGES", 1))
  )

  # Maximum cp.async pipeline stages to generate at build time.
  # Controls the range dispatched in generated TUs and the static_assert
  # bounds in prefill.cuh. Default 4 (stages 1-4); >4 rarely pays off.
  FFPA_BUILD_MAX_STAGES = int(os.environ.get("FFPA_BUILD_MAX_STAGES", 4))

  # Pipeline stages subset to compile: csv (e.g. "2,3") or "all"
  # (= 1..FFPA_BUILD_MAX_STAGES). Highest priority, overrides
  # ENABLE_FFPA_ALL_STAGES. When unset but ENABLE_FFPA_ALL_STAGES is
  # explicitly set, the legacy semantics apply ("1" -> 1..max, "0" ->
  # [1, 2]); when both are unset the default is "2,3" (out-of-set runtime
  # requests are clamped by the generated wrapper, so s1 needs no TU).
  FFPA_BUILD_STAGES = os.environ.get("FFPA_BUILD_STAGES", "")

  # Enable all headdims for FFPA kernels or not, default False.
  # True, headdim will range from 64 to 1024 with step = 64, range(64, 1024, 64)
  # False, headdim defaults to the fixed set 64,128,192,256,320,512 (same as
  # `build_fast.sh --headdim default`). Pass other headdims via FFPA_DEV_HEADDIMS
  # / `build_fast.sh --headdim`.
  ENABLE_FFPA_ALL_HEADDIM = bool(
    int(os.environ.get("ENABLE_FFPA_ALL_HEADDIM", 0))
  )

  # Enable fp16 MMA acc (acc=0 / CUDABackend(acc="f16")) kernels. Default False:
  # the fp16-acc path is rarely used and roughly doubles the fp16 TU count, so
  # it is opt-in. When disabled env.py omits the ffpa_attn_fwd_fp16f16* TUs and
  # the C++/Python dispatch raises a clear "rebuild with ENABLE_FFPA_F16_ACC=1"
  # error. The kernel template code itself is unchanged (gating is at the
  # generation/dispatch layer only).
  ENABLE_FFPA_F16_ACC = bool(int(os.environ.get("ENABLE_FFPA_F16_ACC", 0)))

  # Enable force Q@K^T use fp16 as MMA Acc dtype for FFPA Acc F32 kernels, default False.
  # FFPA Acc F32 kernels MMA Acc = Mixed Q@K^T MMA Acc F16 + P@V MMA Acc F32.
  ENABLE_FFPA_FORCE_QK_F16 = bool(
    int(os.environ.get("ENABLE_FFPA_FORCE_QK_F16", 0))
  )

  # Enable TMA+MMA warp-specialised extension for sm_90+ (opt-in, default off).
  # When enabled, the build compiles the SM120 TMA kernel and dispatches to it
  # at runtime on TMA-capable devices. Requires CUDA Toolkit >= 13.0.
  ENABLE_FFPA_TMA_EXT = bool(int(os.environ.get("ENABLE_FFPA_TMA_EXT", 0)))

  # Enable CuTe C++ kernel extension for sm_120+ (opt-in, default off).
  # Requires ENABLE_FFPA_TMA_EXT=1 AND ENABLE_FFPA_CUTE_EXT=1 to activate
  # the cute-based kernel path. Uses cutlass headers from third_party/cutlass.
  ENABLE_FFPA_CUTE_EXT = bool(int(os.environ.get("ENABLE_FFPA_CUTE_EXT", 0)))

  # Enable force P@V use fp16 as MMA Acc dtype, for FFPA cc F32 kernels, default False.
  # FFPA Acc F32 kernels MMA Acc = Mixed Q@K^T MMA Acc F32 + P@V MMA Acc F16.
  ENABLE_FFPA_FORCE_PV_F16 = bool(
    int(os.environ.get("ENABLE_FFPA_FORCE_PV_F16", 0))
  )

  # Enable FFPA Prefetch QKV at the Appropriate Time Point, default True, boost 5%~10%.
  ENABLE_FFPA_PREFETCH_QKV = bool(
    int(os.environ.get("ENABLE_FFPA_PREFETCH_QKV", 1))
  )

  # Enable QKV smem shared policy, default False (perfered for MMA & g2s overlap).
  # Please, set it as True if you want to run FFPA on low SRAM device.
  ENABLE_FFPA_QKV_SMEM_SHARE = bool(
    int(os.environ.get("ENABLE_FFPA_QKV_SMEM_SHARE", 0))
  )

  # Enable smem swizzle for Q, default True. True: bank conflicts free for Q smem
  # via swizzle; False: bank conflicts free for Q smem via padding.
  ENABLE_FFPA_SMEM_SWIZZLE_Q = bool(
    int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_Q", 1))
  )

  # Enable smem swizzle for K, default True. True: bank conflicts free for K smem
  # via swizzle; False: bank conflicts free for K smem via padding.
  ENABLE_FFPA_SMEM_SWIZZLE_K = bool(
    int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_K", 1))
  )

  # Enable smem swizzle for V, now default True. True: bank conflicts free for V smem
  # via swizzle; False: bank conflicts free for V smem via padding.
  ENABLE_FFPA_SMEM_SWIZZLE_V = bool(
    int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_V", 1))
  )

  # Persist load Q g2s for headdim <= 320 && stages < 3, more SRAM. May not suitable
  # for headdim > 320 due to the SRAM pressure.
  ENABLE_FFPA_PERSIST_Q_G2S = bool(
    int(os.environ.get("ENABLE_FFPA_PERSIST_Q_G2S", 1))
  )

  # Persist load Q from s2r for headdim < 512 to reduce Q from g2s and s2r IO access,
  # but still keep O(1) SRAM complexity. Default value is False. This option will
  # introduce more registers for Q frags as the headdim becomes larger. We should
  # choose to enable it or not according to the balance between register usage and
  # IO access reduction.
  ENABLE_FFPA_PERSIST_Q_S2R = bool(
    int(os.environ.get("ENABLE_FFPA_PERSIST_Q_S2R", 0))
  )

  # Registers Ping pong double buffers for ldmatrix & mma computation overlapping.
  ENABLE_FFPA_REGISTERS_PIPE_KV = bool(
    int(os.environ.get("ENABLE_FFPA_REGISTERS_PIPE_KV", 0))
  )

  # if True: grid(N/Br, H, B) else: grid(N/Br, B * H)
  ENABLE_FFPA_LAUNCH_GRID_DNHB = bool(
    int(os.environ.get("ENABLE_FFPA_LAUNCH_GRID_DNHB", 0))
  )

  # Enable legacy native CUDA kernel generation/compilation. Defaults to
  # disabled so the package can be built and used in Triton-only mode.
  # For development/validation, explicitly build with:
  #   export ENABLE_FFPA_CUDA_IMPL=1
  # ``ENABLE_FFPA_FWD_CUDA_IMPL`` is accepted as a temporary compatibility
  # alias for older scripts. Native CUDA backward is no longer generated.
  ENABLE_FFPA_CUDA_IMPL = bool(
    int(
      os.environ.get(
        "ENABLE_FFPA_CUDA_IMPL", os.environ.get("ENABLE_FFPA_FWD_CUDA_IMPL", 0)
      )
    )
  )

  # --- Build-time tuning knobs ---------------------------------------------
  # Target CUDA SM architectures to compile for. When empty the current
  # device's capability is used. Accepts a comma/semicolon/space separated
  # list of either numeric SMs (e.g. "80,89,90") or aliases (e.g.
  # "ampere,ada,hopper"). Mirrors cache-dit's FFPA_BUILD_ARCH / TORCH_CUDA_
  # ARCH_LIST handling so power users can pin a specific arch set.
  FFPA_BUILD_ARCH = os.environ.get("FFPA_BUILD_ARCH", "")

  # nvcc intra-TU parallelism. With the per-headdim TU split, the outer
  # ``MAX_JOBS`` already drives many nvcc processes in parallel, so keeping
  # ``--threads`` small (default 4) avoids oversubscription. Set to 1 to
  # disable intra-TU threading entirely; larger values only help when
  # ``MAX_JOBS`` is small.
  FFPA_NVCC_THREADS = int(os.environ.get("FFPA_NVCC_THREADS", 4))

  # Emit ptxas verbose info (register / smem usage). Off by default because
  # it produces tens of MB of log output and is only useful for tuning.
  FFPA_PTXAS_VERBOSE = bool(int(os.environ.get("FFPA_PTXAS_VERBOSE", 0)))

  # Development-time headdim subset override. Comma/space separated list of
  # headdims (e.g. ``256,512``) that replaces the full generated set for
  # fast iteration. Empty (default) means use the full set from
  # ``ENABLE_FFPA_ALL_HEADDIM``.
  FFPA_DEV_HEADDIMS = os.environ.get("FFPA_DEV_HEADDIMS", "")

  # Compile per-family debug dispatch knobs (e.g. FFPA_FP8_FORCE_KERNEL,
  # FFPA_DROPOUT_BITMAP_DISABLE). Comma separated subset of {fp16,fp8,fp4}
  # or "all"; each enabled family expands to -DENABLE_FFPA_<FAM>_BUILD_DEBUG.
  # Empty (default) strips every debug getenv branch from the production
  # build, which also removes the FORCE_KERNEL dual template instantiation.
  ENABLE_FFPA_BUILD_DEBUG = os.environ.get("ENABLE_FFPA_BUILD_DEBUG", "")

  @classmethod
  def project_dir(cls):
    return cls.PROJECT_DIR

  @classmethod
  def get_build_arch_list(cls):
    """Resolve the SM targets for the current build.

    Priority order: explicit ``FFPA_BUILD_ARCH`` env var first, then fall
    back to the current visible CUDA device's compute capability.

    :returns: De-duplicated list of numeric SM strings (e.g. ``['89']``).
    :raises RuntimeError: if ``FFPA_BUILD_ARCH`` parses to an empty list,
        or if it is unset and no visible CUDA device is available to
        infer the target arch.
    """
    raw = cls.FFPA_BUILD_ARCH
    if raw.strip():
      archs = []
      for tok in re.split(r"[;,\s]+", raw):
        norm = tok.strip().lower()
        if not norm:
          continue
        norm = norm.removesuffix("+ptx")
        norm = norm.removeprefix("sm_").removeprefix("compute_")
        norm = norm.replace(".", "")
        norm = _ARCH_ALIASES.get(norm, norm)
        if norm not in archs:
          archs.append(norm)
      if not archs:
        raise RuntimeError(
          f"FFPA_BUILD_ARCH={raw!r} parsed to an empty arch list."
        )
      return archs
    # No explicit list -> use the current device's SM capability.
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
      raise RuntimeError(
        "FFPA_BUILD_ARCH is unset and no visible CUDA device is available "
        "to infer the target arch. Set FFPA_BUILD_ARCH=<sm list>, e.g. 80,89,90."
      )
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())
    arch = f"{cap[0]}{cap[1]}"
    # sm_90a/100a/120a: the 'a' suffix enables arch-specific instructions
    # (TMA, WGMMA, etc.) that are unavailable in the base ISA.
    if arch in ("90", "100"):
      arch += "a"
    if arch in ("120"):
      arch += "f"  # sm_120f is required for setmaxnreg instruction
    return [arch]

  @classmethod
  def enable_all_mutistages(cls):
    return cls.ENABLE_FFPA_ALL_STAGES

  @classmethod
  def enable_all_headdim(cls):
    return cls.ENABLE_FFPA_ALL_HEADDIM

  @classmethod
  def enable_force_pv_fp16(cls):
    return cls.ENABLE_FFPA_FORCE_PV_F16

  @classmethod
  def enable_force_qk_fp16(cls):
    return cls.ENABLE_FFPA_FORCE_QK_F16

  @classmethod
  def enable_prefetch_qkv(cls):
    return cls.ENABLE_FFPA_PREFETCH_QKV

  @classmethod
  def enable_qkv_smem_share(cls):
    return cls.ENABLE_FFPA_QKV_SMEM_SHARE

  @classmethod
  def enable_smem_swizzle_q(cls):
    return cls.ENABLE_FFPA_SMEM_SWIZZLE_Q

  @classmethod
  def enable_smem_swizzle_k(cls):
    return cls.ENABLE_FFPA_SMEM_SWIZZLE_K

  @classmethod
  def enable_smem_swizzle_v(cls):
    return cls.ENABLE_FFPA_SMEM_SWIZZLE_V

  @classmethod
  def enable_persist_q_g2s(cls):
    return cls.ENABLE_FFPA_PERSIST_Q_G2S

  @classmethod
  def enable_persist_q_s2r(cls):
    return cls.ENABLE_FFPA_PERSIST_Q_S2R

  @classmethod
  def enable_registers_pipe_kv(cls):
    return cls.ENABLE_FFPA_REGISTERS_PIPE_KV

  @classmethod
  def enable_launch_grid_dnhb(cls):
    return cls.ENABLE_FFPA_LAUNCH_GRID_DNHB

  @classmethod
  def enable_fwd_cuda_impl(cls):
    return cls.ENABLE_FFPA_CUDA_IMPL

  @classmethod
  def enable_cuda_impl(cls):
    return cls.ENABLE_FFPA_CUDA_IMPL

  @classmethod
  def enable_bwd_cuda_impl(cls):
    return False

  @classmethod
  def enable_tma_ext(cls):
    return cls.ENABLE_FFPA_TMA_EXT

  @classmethod
  def enable_cute_ext(cls):
    return cls.ENABLE_FFPA_CUTE_EXT

  @classmethod
  def enable_f16_acc(cls):
    return cls.ENABLE_FFPA_F16_ACC

  @classmethod
  def build_debug_families(cls):
    """Parse ``ENABLE_FFPA_BUILD_DEBUG`` into a set of family tokens.

    :returns: Subset of ``{'fp16', 'fp8', 'fp4'}``; ``all`` expands to the
        full set, empty (default) to the empty set.
    :raises RuntimeError: if the value contains an unknown family token.
    """
    toks = {
      t.strip().lower()
      for t in re.split(r"[;,\s]+", cls.ENABLE_FFPA_BUILD_DEBUG) if t.strip()
    }
    unknown = toks - {"fp16", "fp8", "fp4", "all"}
    if unknown:
      raise RuntimeError(
        f"ENABLE_FFPA_BUILD_DEBUG={cls.ENABLE_FFPA_BUILD_DEBUG!r} contains "
        f"unknown families {sorted(unknown)}, expected fp16/fp8/fp4/all."
      )
    if "all" in toks:
      return {"fp16", "fp8", "fp4"}
    return toks

  @classmethod
  def enable_build_debug(cls, family: str) -> bool:
    """Whether debug dispatch knobs are compiled for one kernel family.

    :param family: One of ``'fp16'``, ``'fp8'``, ``'fp4'``.
    :returns: True when ``ENABLE_FFPA_BUILD_DEBUG`` selects this family.
    """
    return family in cls.build_debug_families()

  @classmethod
  def env_cuda_cflags(cls):
    extra_env_cflags = []
    if cls.enable_all_mutistages():
      extra_env_cflags.append("-DENABLE_FFPA_ALL_STAGES")
    extra_env_cflags.append(
      f"-DFFPA_BUILD_MAX_STAGES={cls.FFPA_BUILD_MAX_STAGES}"
    )
    if cls.enable_all_headdim():
      extra_env_cflags.append("-DENABLE_FFPA_ALL_HEADDIM")
    if cls.enable_force_qk_fp16():
      extra_env_cflags.append("-DENABLE_FFPA_FORCE_QK_F16")
    if cls.enable_force_pv_fp16():
      extra_env_cflags.append("-DENABLE_FFPA_FORCE_PV_F16")
    if cls.enable_prefetch_qkv():
      extra_env_cflags.append("-DENABLE_FFPA_PREFETCH_QKV")
    if cls.enable_qkv_smem_share():
      extra_env_cflags.append("-DENABLE_FFPA_QKV_SMEM_SHARE")
    if cls.enable_smem_swizzle_q():
      extra_env_cflags.append("-DENABLE_FFPA_SMEM_SWIZZLE_Q")
    if cls.enable_smem_swizzle_k():
      extra_env_cflags.append("-DENABLE_FFPA_SMEM_SWIZZLE_K")
    if cls.enable_smem_swizzle_v():
      extra_env_cflags.append("-DENABLE_FFPA_SMEM_SWIZZLE_V")
    if cls.enable_persist_q_g2s():
      extra_env_cflags.append("-DENABLE_FFPA_PERSIST_Q_G2S")
    if cls.enable_persist_q_s2r():
      extra_env_cflags.append("-DENABLE_FFPA_PERSIST_Q_S2R")
    if cls.enable_registers_pipe_kv():
      extra_env_cflags.append("-DENABLE_FFPA_REGISTERS_PIPE_KV")
    if cls.enable_launch_grid_dnhb():
      extra_env_cflags.append("-DENABLE_FFPA_LAUNCH_GRID_DNHB")
    if cls.enable_cuda_impl():
      extra_env_cflags.append("-DENABLE_FFPA_CUDA_IMPL")
    if cls.enable_f16_acc():
      extra_env_cflags.append("-DENABLE_FFPA_F16_ACC")
    if cls.enable_tma_ext():
      extra_env_cflags.append("-DENABLE_FFPA_TMA_EXT")
    if cls.enable_cute_ext():
      extra_env_cflags.append("-DENABLE_FFPA_CUTE_EXT")
    for family in ("fp16", "fp8", "fp4"):
      if cls.enable_build_debug(family):
        extra_env_cflags.append(f"-DENABLE_FFPA_{family.upper()}_BUILD_DEBUG")

    # Debug/profiling pass-through: extra -D defines for nvcc.
    for d in os.environ.get("FFPA_NVCC_DEFINES", "").split(","):
      d = d.strip()
      if d:
        extra_env_cflags.append(f"-D{d}")
    return extra_env_cflags

  @classmethod
  def extra_gcc_flags(cls):
    extra_gcc_flags = ["-O3", "-std=c++20"]
    if cls.enable_cuda_impl():
      extra_gcc_flags.append("-DENABLE_FFPA_CUDA_IMPL")
    if cls.enable_f16_acc():
      extra_gcc_flags.append("-DENABLE_FFPA_F16_ACC")
    # Expose TMA/CUTE ext macros to the .cc pybind TU (ffpa_api.cc) so the
    # CUDA_CUTE_TMA_AVAILABLE attr guard reflects the actual build config.
    if cls.enable_tma_ext():
      extra_gcc_flags.append("-DENABLE_FFPA_TMA_EXT")
    if cls.enable_cute_ext():
      extra_gcc_flags.append("-DENABLE_FFPA_CUTE_EXT")
    return extra_gcc_flags

  @classmethod
  def list_ffpa_env(cls):

    def formatenv(name, value):
      try:
        print(
          f"{name:<30}: {str(value):<5} -> command:"
          f" export {name}={int(value)}"
        )
      except Exception:
        print(f"{name:<30}: {value}")

    _logging_msg("FFPA-ATTN ENVs")
    formatenv("PROJECT_DIR", cls.project_dir())
    formatenv("FFPA_BUILD_ARCH", ",".join(cls.get_build_arch_list()))
    formatenv("FFPA_NVCC_THREADS", cls.FFPA_NVCC_THREADS)
    formatenv("FFPA_PTXAS_VERBOSE", cls.FFPA_PTXAS_VERBOSE)
    formatenv(
      "FFPA_DEV_HEADDIMS", cls.FFPA_DEV_HEADDIMS or (
        "range(64, 1024, 64)"
        if cls.enable_all_headdim() else "64,128,192,256,320,512"
      )
    )
    formatenv("ENABLE_FFPA_ALL_STAGES", cls.enable_all_mutistages())
    formatenv("FFPA_BUILD_MAX_STAGES", cls.FFPA_BUILD_MAX_STAGES)
    formatenv(
      "FFPA_BUILD_STAGES",
      ",".join(str(s) for s in cls._enabled_stages()),
    )
    formatenv("ENABLE_FFPA_ALL_HEADDIM", cls.enable_all_headdim())
    formatenv("ENABLE_FFPA_F16_ACC", cls.enable_f16_acc())
    formatenv("ENABLE_FFPA_PREFETCH_QKV", cls.enable_prefetch_qkv())
    formatenv("ENABLE_FFPA_FORCE_QK_F16", cls.enable_force_qk_fp16())
    formatenv("ENABLE_FFPA_FORCE_PV_F16", cls.enable_force_pv_fp16())
    formatenv("ENABLE_FFPA_PERSIST_Q_G2S", cls.enable_persist_q_g2s())
    formatenv("ENABLE_FFPA_PERSIST_Q_S2R", cls.enable_persist_q_s2r())
    formatenv("ENABLE_FFPA_QKV_SMEM_SHARE", cls.enable_qkv_smem_share())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_Q", cls.enable_smem_swizzle_q())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_K", cls.enable_smem_swizzle_k())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_V", cls.enable_smem_swizzle_v())
    formatenv("ENABLE_FFPA_REGISTERS_PIPE_KV", cls.enable_registers_pipe_kv())
    formatenv("ENABLE_FFPA_LAUNCH_GRID_DNHB", cls.enable_launch_grid_dnhb())
    formatenv("ENABLE_FFPA_CUDA_IMPL", cls.enable_cuda_impl())
    formatenv("ENABLE_FFPA_TMA_EXT", cls.enable_tma_ext())
    formatenv("ENABLE_FFPA_CUTE_EXT", cls.enable_cute_ext())
    formatenv(
      "ENABLE_FFPA_BUILD_DEBUG",
      ",".join(sorted(cls.build_debug_families())) or "none",
    )
    _logging_msg()

  @staticmethod
  def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
      device_name += " WSL2"
    return device_name

  @staticmethod
  def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())

  @classmethod
  def get_enabled_headdims(cls):
    """Return the list of headdims enabled for the current build configuration.

    Priority order: ``FFPA_DEV_HEADDIMS`` (explicit subset for fast
    iteration) -> ``ENABLE_FFPA_ALL_HEADDIM`` (multiples of 64 in
    ``[64, 1024]``) -> default (the fixed set 64,128,192,256,320,512,
    same as ``build_fast.sh --headdim default``).

    :returns: Sorted list of ``int`` headdim values.
    :raises RuntimeError: if ``FFPA_DEV_HEADDIMS`` parses to an empty list.
    """
    raw = cls.FFPA_DEV_HEADDIMS.strip()
    if raw:
      subset = []
      for tok in re.split(r"[;,\s]+", raw):
        if not tok:
          continue
        d = int(tok)
        if d not in subset:
          subset.append(d)
      if not subset:
        raise RuntimeError(
          f"FFPA_DEV_HEADDIMS={raw!r} parsed to an empty list."
        )
      return sorted(subset)
    if cls.enable_all_headdim():
      return list(range(64, 1025, 64))
    return [64, 128, 192, 256, 320, 512]

  @classmethod
  def generated_sources_dir(cls):
    return os.path.join(cls.project_dir(), "csrc", "cuffpa", "generated")

  @staticmethod
  def _write_file(path: str, content: str):
    """Write ``content`` to ``path``, creating parent dirs as needed."""
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)

  @classmethod
  def _render_tu_naming_doc(cls) -> str:
    """Markdown reference for every generated TU filename under generated/."""
    return """\
# Generated TU naming reference

Every file under `csrc/cuffpa/generated/` is auto-generated by `env.py`
(wiped and rewritten per build config). This note decodes the fields.

## Wrapper / family TUs

| Pattern | Meaning |
|---|---|
| `fwd_decls.h` | stage-entry declarations shared by the wrapper TUs |
| `fwd_dispatch.cu` | top-level headdim/stage dispatch (the only public symbol table) |
| `fwd_{variant}_native_hdim{d}.cu` | lightweight wrapper: runtime `stages` -> stage entry `ffpa_attn_fwd_{variant}_d{d}_s{s}` (clamped to the compiled set) |
| `fwd_{variant}_native_hdim{d}_s{s}.cu` | native family TU: stage entry + `ffpa_fwd_native_sm80` / `ffpa_fwd_native_tma` instantiations |
| `fwd_{token}_cute_fp16_hdim{d}_s{s}.cu` | CuTe fp16 family TU (`ffpa_fwd_cute_fp16` / `_sm80` + hybrid stage-1 entries) |
| `fwd_{token}_cute_fp8_hdim{d}_s{s}.cu` | fp8 family TU: `ffpa_fwd_fp8` entry (wrapper shell only; kernel tables live in the variant TUs below) |
| `fwd_{token}_cute_fp4_hdim{d}.cu` | fp4 family TU: `ffpa_fwd_fp4` for every compiled stage (fp4 ignores kStage, one TU per (dtype, d)) |
| `fwd_cute_fp8_preprocess.cu/.cuh` | single definition site of `prepare_fp8_inputs`; the .cuh carries extern declarations for the family TUs |
| `fwd_cute_fp4_preprocess.cu/.cuh` | single definition site of the dtype-agnostic `launch_fp4_quant_*` helpers |
| `fwd_cute_fp8_variants.cuh` | extern-template table of every fp8 variant entry (included from the tail of `launch/cute_fp8.cuh`) |
| `fwd_cute_fp4_variants.cuh` | extern-template table of every fp4 variant entry (included from the tail of `launch/cute_fp4.cuh`) |
| `fwd_cute_fp16_variants.cuh` | extern-template table of every fp16 variant entry (included from the tail of `launch/cute_fp16.cuh`) |

`{variant}` is the input/accumulator combo: `fp16f16`, `fp16f32`, `bf16f32`.
`{token}` is the input dtype token: `fp16` or `bf16` (fp16f16/fp16f32 share
the `fp16` TUs). `{d}` = head dim, `{s}` = pipeline stages (kStage, 2/3).

## fp8 variant TUs (one kernel table per TU)

Pattern: `fwd_{token}_cute_fp8_{impl}_hdim{d}_s{s}_q{q}_b{b}m{m}f{f}.cu`

Example: `fwd_fp16_cute_fp8_persist_hdim128_s3_q1_b1m3f0.cu` = fp16 input,
persist-D kernel, D=128, 3 stages, int8 QK MMA, with attn_bias, resident
row-vector bias mode, 2-byte bias elements.

| Field | Values | Meaning |
|---|---|---|
| `{impl}` | `persist` / `split` / `m4n2` | kernel family: persist-D, split-D M8N1, split-D M4N2 TiledMMA |
| `q{q}` | `0` / `1` | `kQKInt8`: QK^T MMA type - 1 = int8 MMA + int32 acc, 0 = fp8 MMA |
| `b{b}` | `0` / `1` | `kBiasOn`: 1 = an attn_bias tensor is present (kernel compiled with bias plumbing) |
| `m{m}` | `0..3` | `kBiasPlanMode`: bias tile mode - 0 = gmem-direct fallback (no TMA), 1 = dense [kBr,kBc] TMA tile, 2 = row-broadcast TMA ([1,Nkv]), 3 = resident row vector in smem |
| `f{f}` | `0` / `1` | `kBias4BytesPerElem`: bias element width - 1 = 4 bytes (fp32), 0 = 2 bytes (fp16/bf16) |

The runtime wrapper computes the final plan once (single-source
`ffpa::fp8_{impl}_bias_plan`) and dispatches to the matching compile-time
tag, so each TU instantiates exactly one kernel table. Not every
(impl, m) combination exists: the plan demotes per impl/D (e.g. split_d
never yields mode 1; D>=512 never yields mode 2) - see `_fp8_impl_variants`
in `env.py` for the exact production windows.

## fp4 variant TUs (one kernel table per TU)

Pattern: `fwd_{token}_cute_fp4_{impl}_hdim{d}_p{p}_b{b}m{m}f{f}.cu`

Example: `fwd_fp16_cute_fp4_persist_hdim128_p1_b1m3f0.cu` = fp16 input,
persist-D NVFP4 kernel, D=128, MXFP8 PV MMA, with attn_bias, resident
row-vector bias mode, 2-byte bias elements. (No `{s}` field: fp4
launchers fix their stage count, so one TU per (dtype, d, tag).)

| Field | Values | Meaning |
|---|---|---|
| `{impl}` | `persist` / `split` / `m4n2` | kernel family: persist-D (D<=256), split-D M8N1 (256<D<768), split-D M4N2 (768<=D<=1024) |
| `p{p}` | `0` / `1` | `kPvMxfp8`: PV MMA dtype - 1 = MXFP8 PV (persist D<=192 and split only; m4n2 is NVFP4-only), 0 = NVFP4 PV |
| `b{b}` | `0` / `1` | `kBiasOn`: 1 = an attn_bias tensor is present (kernel compiled with bias plumbing) |
| `m{m}` | `0..3` | `kBiasPlanMode`: bias tile mode - 0 = gmem-direct fallback (no TMA), 1 = dense [kBr,kBc] TMA tile, 2 = row-broadcast TMA ([1,Nkv]), 3 = resident row vector in smem |
| `f{f}` | `0` / `1` | `kBias4BytesPerElem`: bias element width - 1 = 4 bytes (fp32), 0 = 2 bytes (fp16/bf16) |

Same single-source dispatch contract as fp8 (`ffpa::fp4_{impl}_bias_plan`),
with two fp4 specifics: the m4n2 plan pins mode 0 in regular builds
(PC-0-5; mode 2 survives only in ENABLE_FFPA_BUILD_DEBUG=fp4 builds behind
FFPA_BIAS_TILE_KEEP), and the persist plan keeps the mode-1 dense tile
(m4n2 demotes it). See `_fp4_impl_variants` in `env.py`.

## fp16 variant TUs (one kernel table per TU)

Pattern: `fwd_{token}_cute_fp16_{impl}_hdim{d}_s{s}_b{b}m{m}f{f}r{r}.cu`

Example: `fwd_fp16_cute_fp16_split_hdim512_s3_b1m3f1r1.cu` = fp16 input,
split-D kernel, D=512, 3 stages, with attn_bias, resident row-vector bias
mode, 4-byte (fp32) bias elements, compiled with dropout plumbing.

| Field | Values | Meaning |
|---|---|---|
| `{impl}` | `persist` / `split` / `m4n2` | kernel family: persist-D (D%32==0, 32<=D<=256, no mode 3), split-D (D%32==0, 128<D<768), split-D M4N2 (D%64==0, 768<=D<=1024) |
| `b{b}` | `0` / `1` | `kBiasOn`: 1 = an attn_bias tensor is present (kernel compiled with bias plumbing) |
| `m{m}` | `0..3` | `kBiasPlanMode`: bias tile mode - 0 = gmem-direct fallback (no TMA), 1 = dense [kBr,kBc] TMA tile, 2 = row-broadcast TMA ([1,Nkv]), 3 = resident row vector in smem |
| `f{f}` | `0` / `1` | `kBias4BytesPerElem`: bias element width - 1 = 4 bytes (fp32), 0 = 2 bytes (fp16/bf16) |
| `r{r}` | `0` / `1` | `kHasDropout`: 1 = compiled with dropout plumbing (Philox keep-mask, optionally the smem bitmap path), dispatched when dropout_p > 0 |

Same single-source dispatch contract as fp8/fp4
(`ffpa::fp16_{impl}_bias_plan`). Unlike fp8/fp4, dropout is a template
axis here (the bitmap path is compile-time), so every (b, m, f) tag pairs
with r in {0, 1}. See `_fp16_impl_variants` in `env.py`.
"""

  @classmethod
  def generate_split_headdim_sources(cls, build_pkg: bool = False):
    """Generate per-(variant, headdim, stage) TUs under ``csrc/cuffpa/generated/``.

    Layout (variant ∈ {fp16f16 (only with ENABLE_FFPA_F16_ACC), fp16f32,
    bf16f32}; dtype token ∈ {fp16, bf16}):

    - ``fwd_<variant>_native_hdim{d}.cu``: lightweight wrapper TU. Includes
      only ``fwd_decls.h`` (NOT ``launch/router.cuh``); dispatches on ``stages``
      to the per-stage symbols ``ffpa_attn_fwd_<variant>_d{d}_s{s}``,
      clamping out-of-set requests to the nearest compiled stage
      (s > max -> max, s < min -> min). Keeps the original dispatch symbol
      name so ``fwd_dispatch.cu`` / ``ffpa_api.cc`` are untouched.
    - ``fwd_<variant>_native_hdim{d}_s{s}.cu``: native family TU
      (``dispatch/native_fp16.cuh`` + ``launch/router.cuh``): the variant's
      stage entry (a single ``launch_ffpa_attn_fwd_template``
      instantiation routing to the family entries) plus the explicit
      instantiation of ``ffpa_fwd_native_sm80`` / ``ffpa_fwd_native_tma``
      with the variant-dependent QK/PV constexprs.
    - ``fwd_<dtype>_cute_fp16_hdim{d}_s{s}.cu`` (ENABLE_FFPA_CUTE_EXT):
      cute_fp16 family TU (``dispatch/cute_fp16.cuh``): explicit instantiation
      of ``ffpa_fwd_cute_fp16``, ``ffpa_fwd_cute_fp16_sm80`` and the hybrid
      stage-1 entries (kPersistMaxD 224/256); keyed by dtype so fp16f16
      and fp16f32 share one __half TU (duplicate explicit instantiation
      would be a compile error).
    - ``fwd_<dtype>_cute_fp8_hdim{d}_s{s}.cu`` (ENABLE_FFPA_TMA_EXT):
      cute_fp8 family TU (``dispatch/cute_fp8.cuh``): explicit
      instantiation of ``ffpa_fwd_fp8``.
    - ``fwd_<dtype>_cute_fp4_hdim{d}.cu`` (ENABLE_FFPA_TMA_EXT): cute_fp4
      family TU (``dispatch/cute_fp4.cuh``): a single TU per (dtype, d)
      instantiating
      ``ffpa_fwd_fp4`` for every compiled stage (the fp4 entry ignores
      kStage, so the kernel templates codegen once inside the TU).
    - ``fwd_cute_fp8_preprocess.cu`` / ``fwd_cute_fp4_preprocess.cu``
      (+ same-name ``.cuh`` declaration tables, ENABLE_FFPA_TMA_EXT):
      the single definition sites for the stage/dtype-independent
      preprocessing chains (``prepare_fp8_inputs`` and the dtype-agnostic
      ``launch_fp4_quant_*`` helpers); family TUs see extern-template
      declarations from the ``.cuh`` and skip re-instantiation.

    The generated dir is wiped and rewritten on every call so stale files
    from a previous config never leak into the build. It is gitignored.

    :param build_pkg: When ``True``, emit a per-call summary line via
        ``_logging_msg`` (suitable for the ``setup.py`` invocation).
    :returns: List of generated file paths (decls header, wrappers, stage
        TUs, dispatch TU).
    """
    gen_dir = cls.generated_sources_dir()
    headdims = cls.get_enabled_headdims()
    generated = []
    fwd_generated_count = 0

    if cls.enable_fwd_cuda_impl():
      # Wipe stale generated files from any prior config before regenerating.
      shutil.rmtree(gen_dir, ignore_errors=True)
      os.makedirs(gen_dir, exist_ok=True)

      decls_path = os.path.join(gen_dir, "fwd_decls.h")
      cls._write_file(decls_path, cls._render_decls_header(headdims))
      generated.append(decls_path)

      for name, content in cls._iter_generated_tus(headdims):
        path = os.path.join(gen_dir, name)
        cls._write_file(path, content)
        generated.append(path)

      dispatch_path = os.path.join(gen_dir, "fwd_dispatch.cu")
      cls._write_file(dispatch_path, cls._render_dispatch_tu(headdims))
      generated.append(dispatch_path)
      fwd_generated_count = sum(1 for p in generated if p.endswith(".cu"))

      # Not a build input: a naming reference for the generated TUs.
      cls._write_file(
        os.path.join(gen_dir, "tu_naming.md"), cls._render_tu_naming_doc()
      )

    if build_pkg:
      _logging_msg(
        f"Generated {fwd_generated_count} CUDA TUs under {gen_dir}",
        sep="",
        mode="left",
      )

    return generated

  # constexpr prefix lines selecting kMmaAccFloat32QK / kMmaAccFloat32PV per
  # variant. fp16f32 keeps the FORCE_{QK,PV}_F16 compile-time hooks for parity
  # with the legacy single-TU behaviour.
  _FP16F16_PREFIX = [
    "  constexpr int kMmaAccFloat32QK = 0;",
    "  constexpr int kMmaAccFloat32PV = 0;",
  ]
  _FP16F32_PREFIX = [
    "#ifdef ENABLE_FFPA_FORCE_QK_F16",
    "  constexpr int kMmaAccFloat32QK = 0;",
    "#else",
    "  constexpr int kMmaAccFloat32QK = 1;",
    "#endif",
    "#ifdef ENABLE_FFPA_FORCE_PV_F16",
    "  constexpr int kMmaAccFloat32PV = 0;",
    "#else",
    "  constexpr int kMmaAccFloat32PV = 1;",
    "#endif",
  ]
  _BF16F32_PREFIX = [
    "  constexpr int kMmaAccFloat32QK = 1;",
    "  constexpr int kMmaAccFloat32PV = 1;",
  ]

  @classmethod
  def _enabled_stages(cls):
    """Return the stage values to instantiate for the current build config.

    Priority: explicit ``FFPA_BUILD_STAGES`` (csv subset or ``all`` =
    ``1..FFPA_BUILD_MAX_STAGES``) first; else an explicitly set
    ``ENABLE_FFPA_ALL_STAGES`` keeps the legacy semantics (``1`` ->
    ``1..max``, ``0`` -> ``[1, 2]``); both unset -> the ``[2, 3]``
    default. Out-of-set runtime requests are clamped by the generated
    wrapper (nearest compiled stage), so s1 no longer needs a TU.

    :returns: Sorted list of ``int`` stage values.
    :raises RuntimeError: if ``FFPA_BUILD_STAGES`` parses to an empty list
      or contains a stage outside ``[1, FFPA_BUILD_MAX_STAGES]``.
    """
    raw = cls.FFPA_BUILD_STAGES.strip()
    if raw:
      if raw.lower() == "all":
        return list(range(1, cls.FFPA_BUILD_MAX_STAGES + 1))
      stages = []
      for tok in re.split(r"[;,\s]+", raw):
        if not tok:
          continue
        s = int(tok)
        if not 1 <= s <= cls.FFPA_BUILD_MAX_STAGES:
          raise RuntimeError(
            f"FFPA_BUILD_STAGES={raw!r}: stage {s} out of range "
            f"[1, {cls.FFPA_BUILD_MAX_STAGES}]."
          )
        if s not in stages:
          stages.append(s)
      if not stages:
        raise RuntimeError(
          f"FFPA_BUILD_STAGES={raw!r} parsed to an empty stage list."
        )
      return sorted(stages)
    if "ENABLE_FFPA_ALL_STAGES" in os.environ:
      if cls.enable_all_mutistages():
        return list(range(1, cls.FFPA_BUILD_MAX_STAGES + 1))
      return [1, 2]
    return [2, 3]

  @classmethod
  def _enabled_variants(cls):
    """Return ``(variant, t_in, constexpr_prefix)`` tuples to generate.

    fp16f16 is prepended only when ``ENABLE_FFPA_F16_ACC`` is on; the fp16f32
    / bf16f32 paths are always generated.

    :raises RuntimeError: if ``ENABLE_FFPA_F16_ACC`` is combined with both
      ``ENABLE_FFPA_FORCE_QK_F16`` and ``ENABLE_FFPA_FORCE_PV_F16`` (the
      docs mark them mutually exclusive; with both forced, fp16f32 and
      fp16f16 collapse to the same QK=0/PV=0 template ids and the native
      family TUs would emit duplicate explicit instantiations).
    """
    if (
      cls.enable_f16_acc() and cls.ENABLE_FFPA_FORCE_QK_F16
      and cls.ENABLE_FFPA_FORCE_PV_F16
    ):
      raise RuntimeError(
        "ENABLE_FFPA_F16_ACC cannot be combined with both "
        "ENABLE_FFPA_FORCE_QK_F16 and ENABLE_FFPA_FORCE_PV_F16: fp16f16 and "
        "fp16f32 would resolve to identical template ids (QK=0, PV=0)."
      )
    variants = [
      ("fp16f32", "__half", cls._FP16F32_PREFIX),
      ("bf16f32", "__nv_bfloat16", cls._BF16F32_PREFIX),
    ]
    if cls.enable_f16_acc():
      variants.insert(0, ("fp16f16", "__half", cls._FP16F16_PREFIX))
    return variants

  @staticmethod
  def _arg_lines(with_stages: bool) -> list:
    lines = [
      "    torch::Tensor Q,",
      "    torch::Tensor K,",
      "    torch::Tensor V,",
      "    torch::Tensor O,",
      "    torch::Tensor attn_bias,",
      "    torch::Tensor softmax_lse,",
    ]
    if with_stages:
      lines.append("    int stages,")
    lines += [
      "    int causal,",
      "    double softmax_scale,",
      "    double dropout_p,",
      "    int64_t philox_seed,",
      "    int64_t philox_offset,",
      "    bool fp8_smooth_k,",
      "    bool fp8_smooth_v,",
      "    int64_t fp8_q_quant_method,",
      "    int64_t fp8_k_quant_method,",
      "    int64_t fp8_v_quant_method,",
      "    int64_t fp8_pv_acc_type,",
      "    int64_t fp8_qk_mm_type,",
      "    bool fp8_hybrid,",
      "    int64_t fp8_hybrid_n_early,",
      "    bool fp4_hybrid,",
      "    int64_t fp4_hybrid_n_early,",
      "    bool fp8_hadamard,",
      "    bool fp4_hadamard,",
      "    int64_t fp4_pv_mm_type,",
      "    bool fp4_smooth_v)",
    ]
    return lines

  @staticmethod
  def _decl(symbol: str, with_stages: bool) -> str:
    args = (
      "torch::Tensor Q, torch::Tensor K, torch::Tensor V, "
      "torch::Tensor O, torch::Tensor attn_bias, torch::Tensor softmax_lse"
    )
    if with_stages:
      args += ", int stages"
    args += (
      ", int causal, double softmax_scale, double dropout_p, "
      "int64_t philox_seed, int64_t philox_offset, bool fp8_smooth_k, "
      "bool fp8_smooth_v, int64_t fp8_q_quant_method, "
      "int64_t fp8_k_quant_method, int64_t fp8_v_quant_method, "
      "int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type, bool fp8_hybrid, "
      "int64_t fp8_hybrid_n_early, bool fp4_hybrid, "
      "int64_t fp4_hybrid_n_early, bool fp8_hadamard, bool fp4_hadamard, "
      "int64_t fp4_pv_mm_type, bool fp4_smooth_v"
    )
    return f"void {symbol}({args});"

  @classmethod
  def _signature(cls, symbol: str, with_stages: bool) -> str:
    head = [f"void {symbol}("] + cls._arg_lines(with_stages)
    head[-1] = head[-1] + " {"
    return "\n".join(head)

  @classmethod
  def _render_decls_header(cls, headdims):
    variants = cls._enabled_variants()
    stages = cls._enabled_stages()
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#pragma once",
      "#include <torch/types.h>",
      "",
    ]
    for variant, _, _ in variants:
      # Explicit-head_dim dispatch helper (pad path entry; only compiled
      # headdims are cases, others throw).
      lines.append(
        cls._decl(f"ffpa_attn_fwd_{variant}_d", True)[:-2] + ", int d);"
      )
      for d in headdims:
        lines.append(cls._decl(f"ffpa_attn_fwd_{variant}_d{d}", True))
        for s in stages:
          lines.append(cls._decl(f"ffpa_attn_fwd_{variant}_d{d}_s{s}", False))
    lines.append("")
    return "\n".join(lines)

  @classmethod
  def _render_wrapper_dispatch(cls, variant: str, d: int) -> str:
    """Render the ``if (stages == s) {...}`` chain calling per-stage symbols.

    Out-of-set requests clamp to the nearest compiled stage: ``stages >
    max(S)`` -> ``max(S)`` (the last exact branch renders as ``>=``), and
    anything below ``min(S)`` falls through to ``min(S)``. This replaces
    the legacy "always fall back to s1" chain, so s1 needs no dedicated
    TU unless it is in the compiled set.
    """
    stages = sorted(cls._enabled_stages())
    call = (
      "Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, "
      "dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v, "
      "fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method, "
      "fp8_pv_acc_type, fp8_qk_mm_type, fp8_hybrid, fp8_hybrid_n_early, "
      "fp4_hybrid, fp4_hybrid_n_early, fp8_hadamard, fp4_hadamard, "
      "fp4_pv_mm_type, fp4_smooth_v"
    )
    if len(stages) == 1:
      return f"  ffpa_attn_fwd_{variant}_d{d}_s{stages[0]}({call});\n"
    lines = []
    for i, s in enumerate(stages):
      kw = "if" if i == 0 else "else if"
      op = ">=" if s == stages[-1] else "=="
      lines.append(f"  {kw} (stages {op} {s}) {{")
      lines.append(f"    ffpa_attn_fwd_{variant}_d{d}_s{s}({call});")
      lines.append("  }")
    lines.append("  else {")
    lines.append(f"    ffpa_attn_fwd_{variant}_d{d}_s{stages[0]}({call});")
    lines.append("  }")
    return "\n".join(lines) + "\n"

  @classmethod
  def _render_wrapper_tu(cls, variant: str, d: int) -> str:
    """Lightweight wrapper TU: only ``fwd_decls.h``, stage if/else dispatch."""
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      '#include "fwd_decls.h"\n\n' +
      cls._signature(f"ffpa_attn_fwd_{variant}_d{d}", True) + "\n" +
      cls._render_wrapper_dispatch(variant, d) + "}\n"
    )

  # Filename token per input dtype for the dtype-keyed family TUs
  # (cute_fp16/cute_fp8/cute_fp4). fp16f16 and fp16f32 share the __half TU.
  _DTYPE_TOKENS = {
    "__half": "fp16",
    "__nv_bfloat16": "bf16",
  }

  @classmethod
  def _iter_generated_tus(cls, headdims):
    """Yield ``(filename, content)`` for every generated forward TU.

    Wrappers / native family TUs (stage entry + native sm80/tma) are
    keyed by (variant, d, s) - the native entries take the
    variant-dependent QK/PV constexprs. cute_fp16/cute_fp8/cute_fp4
    family TUs are keyed by dtype only (a duplicate explicit
    instantiation across TUs is a compile error) and are macro-gated:
    CUTE ext for cute_fp16, TMA ext for cute_fp8/cute_fp4. The
    fp4 entry ignores kStage, so one TU per (dtype, d) covers all
    compiled stages.
    """
    stages = cls._enabled_stages()
    variants = cls._enabled_variants()
    if cls.enable_tma_ext():
      # Shared fp8 preprocessing TU: defines the prepare_fp8_inputs
      # instances once; the same header carries extern-template
      # declarations for every family TU (see launch/cute_fp8.cuh).
      yield (
        "fwd_cute_fp8_preprocess.cuh",
        cls._render_fp8_preprocess_instances(
          headdims, [t_in for _, t_in, _ in variants]
        ),
      )
      yield ("fwd_cute_fp8_preprocess.cu", cls._render_fp8_preprocess_tu())
      # Shared fp8 variant header + one TU per (dtype, d, s, tag) kernel
      # table: the family TUs' s2/s3 single-TU time collapses into small
      # parallel variants (see launch/cute_fp8.cuh wrapper dispatch).
      # fp16f16/fp16f32 share the __half variant TUs (same dedup as the
      # family TUs; a duplicate explicit instantiation is a compile error).
      dtypes = list(dict.fromkeys(t_in for _, t_in, _ in variants))
      yield (
        "fwd_cute_fp8_variants.cuh",
        cls._render_fp8_variants_header(headdims, dtypes),
      )
      for t_in, impl, func, d, s, tag in cls._iter_fp8_variant_combos(
        headdims, dtypes
      ):
        q, b, m, f = tag
        yield (
          f"fwd_{cls._DTYPE_TOKENS[t_in]}_cute_fp8_{impl}_hdim{d}_s{s}"
          f"_q{q}_b{b}m{m}f{f}.cu",
          cls._render_fp8_variant_tu(t_in, impl, func, d, s, tag),
        )
      # Shared fp4 quantize TU: the dtype-agnostic launch_ helpers are
      # identical across the fp16/bf16 family TUs; defined once here and
      # extern-declared for the family TUs (see launch/cute_fp4.cuh).
      yield (
        "fwd_cute_fp4_preprocess.cuh",
        cls._render_fp4_preprocess_instances(headdims),
      )
      yield ("fwd_cute_fp4_preprocess.cu", cls._render_fp4_preprocess_tu())
      # Shared fp4 variant header + one TU per (dtype, d, tag) kernel
      # table (no stage axis: the fp4 launchers fix their stage count);
      # the family TUs' single-TU time collapses into small parallel
      # variants (see launch/cute_fp4.cuh wrapper dispatch).
      yield (
        "fwd_cute_fp4_variants.cuh",
        cls._render_fp4_variants_header(headdims, dtypes),
      )
      for t_in, impl, func, d, tag in cls._iter_fp4_variant_combos(
        headdims, dtypes
      ):
        p, b, m, f = tag
        yield (
          f"fwd_{cls._DTYPE_TOKENS[t_in]}_cute_fp4_{impl}_hdim{d}"
          f"_p{p}_b{b}m{m}f{f}.cu",
          cls._render_fp4_variant_tu(t_in, impl, func, d, tag),
        )
      # Shared fp16 variant header + one TU per (dtype, d, s, tag): the
      # cute_fp16 family TUs' single-TU kernel table (every impl, every
      # tag, incl. the sm80 entry) collapses into small parallel variants
      # (see launch/cute_fp16.cuh wrapper dispatch). Needs both exts (the
      # per-impl headers are double-gated); the family TUs alone are
      # CUTE_EXT-only.
      if cls.enable_cute_ext():
        yield (
          "fwd_cute_fp16_variants.cuh",
          cls._render_fp16_variants_header(headdims, dtypes),
        )
        for t_in, impl, func, d, s, tag in cls._iter_fp16_variant_combos(
          headdims, dtypes
        ):
          b, m, f, r = tag
          yield (
            f"fwd_{cls._DTYPE_TOKENS[t_in]}_cute_fp16_{impl}_hdim{d}_s{s}"
            f"_b{b}m{m}f{f}r{r}.cu",
            cls._render_fp16_variant_tu(t_in, impl, func, d, s, tag),
          )
    for d in headdims:
      for variant, t_in, prefix in variants:
        yield (
          f"fwd_{variant}_native_hdim{d}.cu",
          cls._render_wrapper_tu(variant, d),
        )
        for s in stages:
          yield (
            f"fwd_{variant}_native_hdim{d}_s{s}.cu",
            cls._render_native_family_tu(variant, t_in, prefix, d, s),
          )
      seen_dtypes = set()
      for _, t_in, _ in variants:
        if t_in in seen_dtypes:
          continue
        seen_dtypes.add(t_in)
        token = cls._DTYPE_TOKENS[t_in]
        if cls.enable_cute_ext():
          for s in stages:
            yield (
              f"fwd_{token}_cute_fp16_hdim{d}_s{s}.cu",
              cls._render_cute_fp16_family_tu(t_in, d, s),
            )
        if cls.enable_tma_ext():
          for s in stages:
            yield (
              f"fwd_{token}_cute_fp8_hdim{d}_s{s}.cu",
              cls._render_fp8_family_tu(t_in, d, s),
            )
          yield (
            f"fwd_{token}_cute_fp4_hdim{d}.cu",
            cls._render_fp4_family_tu(t_in, d, stages),
          )

  @classmethod
  def _render_native_family_tu(
    cls, variant: str, t_in: str, prefix: list, d: int, s: int
  ) -> str:
    """Native family TU: the variant's stage entry + native sm80/tma.

    The stage entry (ffpa_attn_fwd_{variant}_d{d}_s{s}, a
    launch_ffpa_attn_fwd_template instantiation routing to the family
    entries) and the native sm80/tma explicit instantiations share the
    variant prefix lines (namespace-scope constexpr QK/PV), so they live
    in one TU; these strong symbols bind to the same template ids the
    routing layer references under any FORCE_{QK,PV}_F16 config.
    """
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "launch/router.cuh"',
      '#include "dispatch/native_fp16.cuh"',
      "using namespace ffpa;",
      "",
    ]
    lines += [ln.lstrip() for ln in prefix]
    lines.append("")
    lines.append(cls._signature(f"ffpa_attn_fwd_{variant}_d{d}_s{s}", False))
    lines.append(
      f"  launch_ffpa_attn_fwd_template<{t_in}, {d}, kMmaAccFloat32QK, "
      f"kMmaAccFloat32PV, {s}>(Q, K, V, O, attn_bias, softmax_lse, causal, "
      "softmax_scale, dropout_p, philox_seed, philox_offset, fp8_smooth_k, "
      "fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method, "
      "fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type, fp8_hybrid, "
      "fp8_hybrid_n_early, fp4_hybrid, fp4_hybrid_n_early, fp8_hadamard, "
      "fp4_hadamard, fp4_pv_mm_type, fp4_smooth_v);"
    )
    lines.append("}")
    lines.append("")
    lines.append(
      f"template void ffpa::ffpa_fwd_native_sm80<{t_in}, {d}, "
      f"kMmaAccFloat32QK, kMmaAccFloat32PV, {s}>"
      "(const ffpa::FfpaFwdParams&);"
    )
    lines.append(
      f"template void ffpa::ffpa_fwd_native_tma<{t_in}, {d}, "
      f"kMmaAccFloat32QK, kMmaAccFloat32PV, {s}>"
      "(const ffpa::FfpaFwdParams&);"
    )
    return "\n".join(lines) + "\n"

  @classmethod
  def _render_cute_fp16_family_tu(cls, t_in: str, d: int, s: int) -> str:
    """CuTe fp16 family TU: cute_fp16 sm120/sm80 entries + hybrid stage-1."""
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "dispatch/cute_fp16.cuh"',
      "",
      f"template void ffpa::ffpa_fwd_cute_fp16<{t_in}, {d}, {s}>"
      "(const ffpa::FfpaFwdParams&);",
      f"template void ffpa::ffpa_fwd_cute_fp16_sm80<{t_in}, {d}, {s}>"
      "(const ffpa::FfpaFwdParams&);",
      f"template void ffpa::ffpa_fwd_fp16_stage1<{t_in}, {d}, {s}, 224>"
      "(const ffpa::FfpaFwdParams&);",
      f"template void ffpa::ffpa_fwd_fp16_stage1<{t_in}, {d}, {s}, 256>"
      "(const ffpa::FfpaFwdParams&);",
    ]
    return "\n".join(lines) + "\n"

  @classmethod
  def _render_fp8_family_tu(cls, t_in: str, d: int, s: int) -> str:
    """FP8 family TU: explicit instantiation of the CUTE_TMA_FP8 entry."""
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      '#include "dispatch/cute_fp8.cuh"\n\n'
      f"template void ffpa::ffpa_fwd_fp8<{t_in}, {d}, {s}>"
      "(const ffpa::FfpaFwdParams&);\n"
    )

  @classmethod
  def _fp8_variant_blocks(cls, d: int):
    """(kBr, kBc) of the fp8 launcher owning head_dim d.

    Mirrors the constexpr choices at the top of each launcher in
    launch/cute_fp8.cuh: persist_d (D<=224) shrinks kBc above D=128,
    split_d (D<768) keeps 128/128, m4n2 (D>=768) drops to 64/64.
    """
    if d <= 224:
      return 128, (128 if d <= 128 else 64)
    if d < 768:
      return 128, 128
    return 64, 64

  @classmethod
  def _fp8_impl_variants(cls, d: int):
    """(impl key, variant entry func, tag tuples) owning head_dim d.

    Mirrors the routing windows in dispatch/cute_fp8.cuh (keep in sync):
    regular builds route by constexpr D (persist_d D%32==0 and 32<=D<=224;
    split_d D%64==0 and 224<D<768; m4n2 D%64==0 and 768<=D<=1024), so only
    the routed impl's tags get variant TUs. Debug builds additionally
    instantiate both split wrappers across the FFPA_FP8_FORCE_KERNEL A/B
    window (224<D<=1024), so both families get TUs there. Tag tuple =
    (kQKInt8, kBiasOn, kBiasPlanMode, kBias4BytesPerElem).
    """
    debug_fp8 = cls.enable_build_debug("fp8")
    out = []
    if d % 32 == 0 and 32 <= d <= 224:
      tags = [
        (0, 0, 0),
        (1, 2, 1),
        (1, 2, 0),
        (1, 3, 0),
        (1, 0, 0),
      ]
      out.append(("persist", "launch_cute_fwd_persist_d_fp8_sm120_v", tags))
    if d % 64 == 0 and 224 < d <= 1024:
      routed_split = 224 < d < 768
      routed_m4n2 = d >= 768
      if routed_split or debug_fp8:
        tags = [(0, 0, 0), (1, 0, 0)]
        if d < 512:
          tags += [(1, 2, 1), (1, 2, 0)]
        out.append(("split", "launch_cute_fwd_split_d_fp8_sm120_v", tags))
      if routed_m4n2 or debug_fp8:
        tags = [
          (0, 0, 0),
          (1, 1, 1),
          (1, 1, 0),
          (1, 2, 1),
          (1, 2, 0),
          (1, 3, 1),
          (1, 3, 0),
          (1, 0, 0),
        ]
        out.append(("m4n2", "launch_cute_fwd_split_d_m4n2_fp8_sm120_v", tags))
    return out

  @classmethod
  def _iter_fp8_variant_combos(cls, headdims, dtypes):
    """Yield (t_in, impl, func, d, s, (q, b, m, f)) for every fp8 variant."""
    for t_in in dtypes:
      for d in headdims:
        for impl, func, tags in cls._fp8_impl_variants(d):
          for s in cls._enabled_stages():
            for q in (0, 1):
              for b, m, f in tags:
                yield t_in, impl, func, d, s, (q, b, m, f)

  # 19-arg signature shared by every fp8 variant entry (Q, K, V, O,
  # attn_bias, softmax_lse, causal, softmax_scale, dropout_p, philox_seed,
  # philox_offset, fp8_smooth_k, fp8_smooth_v, q/k/v quant, pv_acc,
  # q_start_row, fp8_hadamard). Keep in sync with launch/cute_fp8.cuh.
  _FP8_VARIANT_SIG = (
    "(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
    "torch::Tensor, torch::Tensor, int, double, double, int64_t, int64_t, "
    "bool, bool, int64_t, int64_t, int64_t, int64_t, int, bool)"
  )

  @classmethod
  def _render_fp8_variants_header(cls, headdims, dtypes) -> str:
    """Shared fp8 variant header: extern declarations only.

    The variant TUs include the per-impl headers
    (launch/cute_fp8_{persist_d,split_d,split_d_m4n2}.cuh) directly and
    compile exactly one kernel table. Every family TU includes this
    header from the tail of launch/cute_fp8.cuh and gets extern-template
    declarations for the whole table, which suppresses re-instantiation
    the whole table, which suppresses re-instantiation of the variant
    entries (and keeps FFPA_FP8_FORCE_KERNEL A/B combos - absent from
    the table - on the implicit-instantization path).
    """
    decls = []
    for t_in, _, func, d, s, (q, b, m, f) in cls._iter_fp8_variant_combos(
      headdims, dtypes
    ):
      decls.append(
        f"extern template void {func}<{t_in}, {d}, {s}, "
        f"{'true' if q else 'false'}, {b}, {m}, {f}>"
        f"{cls._FP8_VARIANT_SIG};"
      )
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#ifndef FFPA_GENERATED_FWD_CUTE_FP8_VARIANTS_CUH_",
      "#define FFPA_GENERATED_FWD_CUTE_FP8_VARIANTS_CUH_",
      "// Declarations only: included from the tail of",
      "// launch/cute_fp8.cuh, after the variant templates are defined.",
    ]
    lines += decls
    lines += ["#endif", ""]
    return "\n".join(lines)

  # impl token (TU naming / _fp8_impl_variants) -> per-impl header stem
  _FP8_IMPL_HEADER = {
    "persist": "persist_d",
    "split": "split_d",
    "m4n2": "split_d_m4n2",
  }

  @classmethod
  def _render_fp8_variant_tu(
    cls, t_in: str, impl: str, func: str, d: int, s: int, tag: tuple
  ) -> str:
    """Single-variant TU: one kernel table per translation unit."""
    q, b, m, f = tag
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      f'#include "launch/cute_fp8_{cls._FP8_IMPL_HEADER[impl]}.cuh"\n\n'
      f"template void {func}<{t_in}, {d}, {s}, "
      f"{'true' if q else 'false'}, {b}, {m}, {f}>"
      f"{cls._FP8_VARIANT_SIG};\n"
    )

  @classmethod
  def _fp4_impl_variants(cls, d: int):
    """(impl key, variant entry func, tag tuples) owning head_dim d.

    Mirrors the routing windows in dispatch/cute_fp4.cuh (keep in sync):
    persist_d owns D%64==0 and 64<=D<=256 (MXFP8 PV only D<=192),
    split_d owns D%64==0 and 256<D<768 (PV dual-open), m4n2 owns
    D%64==0 and 768<=D<=1024 (NVFP4-only PV, so pv stays 0). Regular
    builds pin m4n2 to mode 0 (PC-0-5); debug builds
    (ENABLE_FFPA_BUILD_DEBUG=fp4) add the mode-2 KEEP tags behind
    FFPA_BIAS_TILE_KEEP. Tag tuple = (kPvMxfp8, kBiasOn, kBiasPlanMode,
    kBias4BytesPerElem).
    """
    debug_fp4 = cls.enable_build_debug("fp4")
    out = []
    if d % 64 == 0 and 64 <= d <= 256:
      for pv in ((0, 1) if d <= 192 else (0, )):
        out.append((
          "persist",
          "launch_cute_fwd_persist_d_fp4_sm120_v",
          [
            (pv, 0, 0, 0),
            (pv, 1, 1, 1),
            (pv, 1, 1, 0),
            (pv, 1, 2, 1),
            (pv, 1, 2, 0),
            (pv, 1, 3, 1),
            (pv, 1, 3, 0),
            (pv, 1, 0, 0),
          ],
        ))
    if d % 64 == 0 and 256 < d < 768:
      for pv in (0, 1):
        out.append((
          "split",
          "launch_cute_fwd_split_d_fp4_sm120_v",
          [
            (pv, 0, 0, 0),
            (pv, 1, 2, 1),
            (pv, 1, 2, 0),
            (pv, 1, 3, 0),
            (pv, 1, 0, 0),
          ],
        ))
    if d % 64 == 0 and 768 <= d <= 1024:
      tags = [(0, 0, 0, 0), (0, 1, 0, 0)]
      if debug_fp4:
        tags += [(0, 1, 2, 1), (0, 1, 2, 0)]
      out.append(("m4n2", "launch_cute_fwd_split_d_m4n2_fp4_sm120_v", tags))
    return out

  @classmethod
  def _iter_fp4_variant_combos(cls, headdims, dtypes):
    """Yield (t_in, impl, func, d, (p, b, m, f)) for every fp4 variant."""
    for t_in in dtypes:
      for d in headdims:
        for impl, func, tags in cls._fp4_impl_variants(d):
          for tag in tags:
            yield t_in, impl, func, d, tag

  # 11-arg signature shared by every fp4 variant entry (Q, K, V, O,
  # attn_bias, softmax_lse, causal, softmax_scale, q_start_row,
  # fp4_hadamard, fp4_smooth_v). Keep in sync with launch/cute_fp4.cuh.
  _FP4_VARIANT_SIG = (
    "(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
    "torch::Tensor, torch::Tensor, int, double, int, bool, bool)"
  )

  @classmethod
  def _render_fp4_variants_header(cls, headdims, dtypes) -> str:
    """Shared fp4 variant header: extern declarations only.

    Variant TUs include the per-impl headers
    (launch/cute_fp4_{persist_d,split_d,split_d_m4n2}.cuh) directly and
    compile exactly one kernel table; every family TU includes this
    header from the tail of launch/cute_fp4.cuh and gets extern-template
    declarations that suppress variant re-instantiation.
    """
    decls = []
    for t_in, _, func, d, (p, b, m, f
                           ) in cls._iter_fp4_variant_combos(headdims, dtypes):
      decls.append(
        f"extern template void {func}<{t_in}, {d}, "
        f"{'true' if p else 'false'}, {b}, {m}, {f}>"
        f"{cls._FP4_VARIANT_SIG};"
      )
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#ifndef FFPA_GENERATED_FWD_CUTE_FP4_VARIANTS_CUH_",
      "#define FFPA_GENERATED_FWD_CUTE_FP4_VARIANTS_CUH_",
      "// Declarations only: included from the tail of",
      "// launch/cute_fp4.cuh, after the variant templates are defined.",
    ]
    lines += decls
    lines += ["#endif", ""]
    return "\n".join(lines)

  # impl token (TU naming / _fp4_impl_variants) -> per-impl header stem
  _FP4_IMPL_HEADER = {
    "persist": "persist_d",
    "split": "split_d",
    "m4n2": "split_d_m4n2",
  }

  @classmethod
  def _render_fp4_variant_tu(
    cls, t_in: str, impl: str, func: str, d: int, tag: tuple
  ) -> str:
    """Single-variant TU: one kernel table per translation unit."""
    p, b, m, f = tag
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      f'#include "launch/cute_fp4_{cls._FP4_IMPL_HEADER[impl]}.cuh"\n\n'
      f"template void {func}<{t_in}, {d}, "
      f"{'true' if p else 'false'}, {b}, {m}, {f}>"
      f"{cls._FP4_VARIANT_SIG};\n"
    )

  @classmethod
  def _fp16_impl_variants(cls, d: int):
    """(impl key, variant entry func, tag tuples) owning head_dim d.

    Mirrors the routing windows in dispatch/cute_fp16.cuh (keep in sync):
    persist_d owns D%32==0 and 32<=D<=256 (the direct dispatch route caps
    at 128; the fp8/fp4 hybrid stage-1 entries extend it to 224/256) and
    has no mode 3; split_d owns D%32==0 above 128 ((32,64) chunks for
    D%64==0 below 768, (32,32) otherwise — derived in
    _fp16_split_chunks); m4n2 owns D%64==0 and 768<=D<=1024. Tag tuple =
    (kBiasOn, kBiasPlanMode, kBias4BytesPerElem); the kHasDropout axis
    expands in _iter_fp16_variant_combos.
    """
    out = []
    if d % 32 == 0 and 32 <= d <= 256:
      out.append((
        "persist",
        "launch_cute_fwd_persist_d_sm120_v",
        [
          (0, 0, 0),
          (1, 1, 1),
          (1, 1, 0),
          (1, 2, 1),
          (1, 2, 0),
          (1, 0, 0),
        ],
      ))
    if d > 128 and d < 768 and d % 32 == 0:
      out.append((
        "split",
        "launch_cute_fwd_split_d_sm120_v",
        [
          (0, 0, 0),
          (1, 1, 1),
          (1, 1, 0),
          (1, 2, 1),
          (1, 2, 0),
          (1, 3, 1),
          (1, 3, 0),
          (1, 0, 0),
        ],
      ))
    if d % 64 == 0 and 768 <= d <= 1024:
      out.append((
        "m4n2",
        "launch_cute_fwd_split_d_m4n2_sm120_v",
        [
          (0, 0, 0),
          (1, 1, 1),
          (1, 1, 0),
          (1, 2, 1),
          (1, 2, 0),
          (1, 3, 1),
          (1, 3, 0),
          (1, 0, 0),
        ],
      ))
    return out

  @classmethod
  def _fp16_split_chunks(cls, d: int):
    """(kQKDChunk, kVDChunk) of the split_d launcher owning head_dim d."""
    return (32, 64) if d % 64 == 0 else (32, 32)

  @classmethod
  def _iter_fp16_variant_combos(cls, headdims, dtypes):
    """Yield (t_in, impl, func, d, s, (b, m, f, r)) for every fp16 variant."""
    for t_in in dtypes:
      for d in headdims:
        for impl, func, tags in cls._fp16_impl_variants(d):
          for s in cls._enabled_stages():
            for b, m, f in tags:
              for r in (0, 1):
                yield t_in, impl, func, d, s, (b, m, f, r)

  # 11-arg signature shared by every fp16 variant entry (Q, K, V, O,
  # attn_bias, softmax_lse, causal, softmax_scale, dropout_p, philox_seed,
  # philox_offset). Keep in sync with launch/cute_fp16.cuh.
  _FP16_VARIANT_SIG = (
    "(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
    "torch::Tensor, torch::Tensor, int, double, double, int64_t, int64_t)"
  )

  @classmethod
  def _render_fp16_variants_header(cls, headdims, dtypes) -> str:
    """Shared fp16 variant header: extern declarations only.

    Variant TUs include the per-impl headers
    (launch/cute_fp16_{persist_d,split_d,split_d_m4n2}.cuh) directly and
    compile exactly one kernel table; every family TU includes this
    header from the tail of launch/cute_fp16.cuh and gets extern-template
    declarations that suppress variant re-instantiation.
    """
    decls = []
    for t_in, impl, func, d, s, (
      b,
      m,
      f,
      r,
    ) in cls._iter_fp16_variant_combos(headdims, dtypes):
      if impl == "split":
        qc, vc = cls._fp16_split_chunks(d)
        args = f"<{t_in}, {d}, {s}, {qc}, {vc}, {b}, {m}, {f}, {r}>"
      else:
        args = f"<{t_in}, {d}, {s}, {b}, {m}, {f}, {r}>"
      decls.append(f"extern template void {func}{args}{cls._FP16_VARIANT_SIG};")
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#ifndef FFPA_GENERATED_FWD_CUTE_FP16_VARIANTS_CUH_",
      "#define FFPA_GENERATED_FWD_CUTE_FP16_VARIANTS_CUH_",
      "// Declarations only: included from the tail of",
      "// launch/cute_fp16.cuh, after the variant templates are defined.",
    ]
    lines += decls
    lines += ["#endif", ""]
    return "\n".join(lines)

  # impl token (TU naming / _fp16_impl_variants) -> per-impl header stem
  _FP16_IMPL_HEADER = {
    "persist": "persist_d",
    "split": "split_d",
    "m4n2": "split_d_m4n2",
  }

  @classmethod
  def _render_fp16_variant_tu(
    cls, t_in: str, impl: str, func: str, d: int, s: int, tag: tuple
  ) -> str:
    """Single-variant TU: one kernel table per translation unit."""
    b, m, f, r = tag
    if impl == "split":
      qc, vc = cls._fp16_split_chunks(d)
      args = f"<{t_in}, {d}, {s}, {qc}, {vc}, {b}, {m}, {f}, {r}>"
    else:
      args = f"<{t_in}, {d}, {s}, {b}, {m}, {f}, {r}>"
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      f'#include "launch/cute_fp16_{cls._FP16_IMPL_HEADER[impl]}.cuh"\n\n'
      f"template void {func}{args}{cls._FP16_VARIANT_SIG};\n"
    )

  @classmethod
  def _render_fp8_preprocess_instances(cls, headdims, dtypes) -> str:
    """Explicit-instantiation table for ffpa_fp8::prepare_fp8_inputs.

    The preprocess TU (fwd_cute_fp8_preprocess.cu) defines FFPA_FP8_
    PREPROCESS_TU and emits the definitions; every other TU including
    this header gets the matching extern-template declarations, so the
    s2/s3 family TUs stop re-instantiating the quantize kernel family.
    """
    sig = (
      "(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,"
      " const ffpa_fp8::Fp8InputLayout&, const ffpa_fp8::Fp8InputLayout&,"
      " const ffpa_fp8::Fp8InputLayout&, int, int, int, int, int, int,"
      " int, int, int, bool, bool, bool, bool, float, bool, cudaStream_t)"
    )
    combos = []
    seen = set()
    debug_fp8 = cls.enable_build_debug("fp8")
    for t_in in dtypes:
      for d in headdims:
        br, bc = cls._fp8_variant_blocks(d)
        blocks = [(br, bc)]
        if debug_fp8 and 224 < d <= 1024:
          # FFPA_FP8_FORCE_KERNEL A/B instantiates both split_d and m4n2
          # in one TU; the forced variant's blocks are absent from the
          # dispatch route, so emit both or the TU silently falls back to
          # implicit instantiation (compile-time only, no correctness
          # impact).
          blocks.append((64, 64) if (br, bc) == (128, 128) else (128, 128))
        for b in blocks:
          for qk in ("false", "true"):
            key = (t_in, b[0], b[1], d, qk)
            if key not in seen:
              seen.add(key)
              combos.append(
                f"ffpa_fp8::prepare_fp8_inputs"
                f"<{t_in}, {b[0]}, {b[1]}, {d}, {qk}>"
              )
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#include \"cute/fp8/prepare_inputs.cuh\"",
      "",
      "#ifdef FFPA_FP8_PREPROCESS_TU",
    ]
    lines += [
      f"template ffpa_fp8::Fp8QuantizedInputs {c}{sig};" for c in combos
    ]
    lines += ["#else"]
    lines += [
      f"extern template ffpa_fp8::Fp8QuantizedInputs {c}{sig};" for c in combos
    ]
    lines += ["#endif", ""]
    return "\n".join(lines)

  @classmethod
  def _render_fp8_preprocess_tu(cls) -> str:
    """Preprocess TU: the single definition site for the table above."""
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      "#define FFPA_FP8_PREPROCESS_TU\n"
      '#include "generated/fwd_cute_fp8_preprocess.cuh"\n'
    )

  # dtype-agnostic fp4 quantize launchers (template <int kHeadDim>, the
  # half/bf16 kernels are picked at runtime inside), mirrored from
  # cute/fp4/quantize_fp4.cuh. The fp4 family TUs are keyed by dtype, so
  # these identical instantiations would otherwise be codegen'd twice.
  # (signature, max head_dim or None); keep in sync with the headers.
  _FP4_PREPROCESS_LAUNCHERS = [
    (
      "launch_fp4_quant_q_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "const torch::Tensor&, long, bool)",
      None,
    ),
    (
      "launch_fp4_quant_k_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "const torch::Tensor&, long, bool)",
      None,
    ),
    (
      "launch_fp4_quant_vt_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "long, const torch::Tensor&)",
      None,
    ),
    (
      "launch_mxfp8_quant_vt_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "long, const torch::Tensor&)",
      None,
    ),
    (
      "launch_fp4_quant_q_wht_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "const torch::Tensor&, long)",
      "pow2",
    ),
    (
      "launch_fp4_quant_k_wht_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "const torch::Tensor&, long)",
      "pow2",
    ),
    (
      "launch_fp4_q_block_mean_sm120",
      "(const torch::Tensor&, torch::Tensor&)",
      None,
    ),
    (
      "launch_fp4_quant_qkv_fused_sm120",
      "(const torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "torch::Tensor&, torch::Tensor&, torch::Tensor&, "
      "const torch::Tensor&, const torch::Tensor&, torch::Tensor&, "
      "torch::Tensor&, const torch::Tensor&, const torch::Tensor&, "
      "torch::Tensor&, torch::Tensor&, long, long, bool, bool)",
      128,
    ),
  ]

  @classmethod
  def _render_fp4_preprocess_instances(cls, headdims) -> str:
    """Explicit-instantiation table for the dtype-agnostic fp4 launchers.

    Same dual-mode pattern as the fp8 table: the preprocess TU
    (fwd_cute_fp4_preprocess.cu) defines FFPA_FP4_PREPROCESS_TU and emits the
    definitions; the fp4 family TUs get extern-template declarations, so
    the half and bf16 family TUs stop codegen'ing the same quantize
    kernels twice.
    """
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "cute/fp4/quantize_fp4.cuh"',
      "",
      "#ifdef FFPA_FP4_PREPROCESS_TU",
    ]
    for name, sig, d_filter in cls._FP4_PREPROCESS_LAUNCHERS:
      for d in headdims:
        if d_filter == "pow2" and d & (d - 1):
          continue
        if isinstance(d_filter, int) and d > d_filter:
          continue
        lines.append(f"template void ffpa_fp4::{name}<{d}>{sig};")
    lines += ["#else"]
    for name, sig, d_filter in cls._FP4_PREPROCESS_LAUNCHERS:
      for d in headdims:
        if d_filter == "pow2" and d & (d - 1):
          continue
        if isinstance(d_filter, int) and d > d_filter:
          continue
        lines.append(f"extern template void ffpa_fp4::{name}<{d}>{sig};")
    lines += ["#endif", ""]
    return "\n".join(lines)

  @classmethod
  def _render_fp4_preprocess_tu(cls) -> str:
    """Preprocess TU: the single definition site for the table above."""
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      "#define FFPA_FP4_PREPROCESS_TU\n"
      '#include "generated/fwd_cute_fp4_preprocess.cuh"\n'
    )

  @classmethod
  def _render_fp4_family_tu(cls, t_in: str, d: int, stages: list) -> str:
    """FP4 family TU: one TU per (dtype, d) covering every compiled stage.

    The fp4 launcher ignores kStage (fixed by traits), so all explicit
    instantiations live in a single TU and the kernel templates codegen
    once inside it. Exception: the hybrid path forwards kStage to the
    fp16 stage-1 entry, which the cute_fp16 family TU instantiates per stage.
    """
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "dispatch/cute_fp4.cuh"',
      "",
    ]
    for s in stages:
      lines.append(
        f"template void ffpa::ffpa_fwd_fp4<{t_in}, {d}, {s}>"
        "(const ffpa::FfpaFwdParams&);"
      )
    return "\n".join(lines) + "\n"

  @classmethod
  def _render_dispatch_tu(cls, headdims) -> str:
    # fp16f16 (acc=0) dispatch is only emitted when ENABLE_FFPA_F16_ACC is on.
    specs = [
      ("ffpa_attn_fwd_fp16f32", "torch::kHalf"),
      ("ffpa_attn_fwd_bf16f32", "torch::kBFloat16"),
    ]
    if cls.enable_f16_acc():
      specs.insert(0, ("ffpa_attn_fwd_fp16f16", "torch::kHalf"))

    call_args = (
      "Q, K, V, O, attn_bias, softmax_lse, stages, causal, softmax_scale, "
      "dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v, "
      "fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method, "
      "fp8_pv_acc_type, fp8_qk_mm_type, fp8_hybrid, fp8_hybrid_n_early, "
      "fp4_hybrid, fp4_hybrid_n_early, fp8_hadamard, fp4_hadamard, "
      "fp4_pv_mm_type, fp4_smooth_v"
    )

    out = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "logging.cuh"',
      '#include "fwd_decls.h"',
      "",
    ]
    for name, dtype in specs:
      # Explicit-head_dim dispatch helper: pad path calls this with the
      # padded D (Q.size(3) is the unpadded D_og, so the normal entry can't
      # be used). Only compiled headdims are cases; others throw cleanly.
      helper_sig = cls._signature(name + "_d", True)
      out.append(helper_sig[:-3] + ", int d) {")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(Q, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(K, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(V, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(O, {dtype})")
      out.append("  switch (d) {")
      for d in headdims:
        out.append(f"    case {d}: {name}_d{d}({call_args}); break;")
      out.append(
        '    default: throw std::runtime_error("headdim not support!");'
      )
      out.append("  }")
      out.append("}")
      out.append("")
      # Normal entry: derive head_dim from Q and reuse the helper.
      out.append(cls._signature(name, True))
      out.append(f"  {name}_d({call_args}, Q.size(3));")
      out.append("}")
      out.append("")
    return "\n".join(out) + "\n"

  @staticmethod
  def get_build_sources(build_pkg: bool = False):

    def csrc(sub_dir, filename):
      csrc_file = f"{ENV.project_dir()}/csrc/{sub_dir}/{filename}"
      if build_pkg:
        _logging_msg(f"csrc_file: {csrc_file}", sep="", mode="left")
      return csrc_file

    if build_pkg:
      _logging_msg()
    # Generate per-headdim TUs under csrc/cuffpa/generated/ and use them as
    # the actual build sources. The generated TUs include launch/router.cuh,
    # which in turn includes ffpa_attn_fwd.cuh. Splitting by headdim enables
    # MAX_JOBS to drive nvcc on many small files in parallel and cuts the build
    # time of the heavy launch_ffpa_attn_fwd_template instantiations.
    generated_files = ENV.generate_split_headdim_sources(build_pkg=build_pkg)
    generated_sources = [p for p in generated_files if p.endswith(".cu")]
    if build_pkg:
      for gs in generated_sources:
        _logging_msg(f"csrc_file: {gs}", sep="", mode="left")
    build_sources = [
      csrc("cuffpa", "ffpa_api.cc"),
    ] + generated_sources
    if build_pkg:
      _logging_msg()
    return build_sources

  @staticmethod
  def get_build_cuda_cflags(build_pkg: bool = False):
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    if os.environ.get("FFPA_LINEINFO", "0") == "1":
      extra_cuda_cflags.append("-lineinfo")
    extra_cuda_cflags.append("-std=c++20")
    extra_cuda_cflags.append("-Xcompiler")
    extra_cuda_cflags.append("-fPIC")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    extra_cuda_cflags.extend(ENV.env_cuda_cflags())
    extra_cuda_cflags.append(f"-I {ENV.project_dir()}/csrc/cuffpa")
    if ENV.enable_cute_ext():
      extra_cuda_cflags.append(
        f"-I {ENV.project_dir()}/third_party/cutlass/include"
      )
    extra_cuda_cflags.append("-diag-suppress")
    extra_cuda_cflags.append("177")
    extra_cuda_cflags.append("-diag-suppress")
    extra_cuda_cflags.append("1886")
    # 2908: deprecated implicit by-copy capture of "this" in third_party
    # CUTLASS headers; upstream code, not ours to patch.
    extra_cuda_cflags.append("-diag-suppress")
    extra_cuda_cflags.append("2908")
    # 3189-D: torch/python.h names a lambda parameter "module"; cudafe
    # parses it as an identifier under C++20 (harmless, upstream torch).
    extra_cuda_cflags.append("-Xcudafe")
    extra_cuda_cflags.append("--diag_suppress=3189")
    if ENV.FFPA_PTXAS_VERBOSE:
      extra_cuda_cflags.append("--ptxas-options=-v")
      extra_cuda_cflags.append("-Xptxas")
      extra_cuda_cflags.append("-v")
    else:
      extra_cuda_cflags.append("--ptxas-options=-O3")
    # NOTE: ptxas C7506 (setmaxnreg ignored on sm_120a) is an *info*-level
    # message that only appears under --ptxas-options=-v (FFPA_PTXAS_VERBOSE).
    # Normal builds are unaffected. ptxas --diag-suppress does not accept
    # info-level codes (only warning/error diag numbers), so it cannot be
    # suppressed via command-line flags. See sm120.cuh header for details.

    if ENV.FFPA_NVCC_THREADS > 1:
      extra_cuda_cflags.append(f"--threads={ENV.FFPA_NVCC_THREADS}")
    # Avoid None or empty str as flag or macro
    extra_cuda_cflags = [flag for flag in extra_cuda_cflags if flag]
    return extra_cuda_cflags

  @staticmethod
  def get_build_cflags():
    extra_cflags = []
    extra_cflags.append("-std=c++20")
    return extra_cflags

  @staticmethod
  def get_cuda_bare_metal_version(cuda_dir):
    # helper function to get cuda version
    import subprocess

    from packaging.version import parse

    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

  @staticmethod
  def build(verbose: bool = False):
    from torch.utils.cpp_extension import load

    if not ENV.enable_fwd_cuda_impl():
      raise RuntimeError(
        "CUDA kernels are disabled for this build. "
        "Rebuild with ENABLE_FFPA_CUDA_IMPL=1 to build ffpa_attn._C."
      )

    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    # Load the CUDA kernel as a python module
    _logging_msg(
      f"Loading ffpa_attn lib on device: {ENV.get_device_name()}, "
      f"capability: {ENV.get_device_capability()}, "
      f"Arch ENV: {torch_arch_list_env}"
    )
    return load(
      name="ffpa_attn._C",
      sources=ENV.get_build_sources(),
      extra_cuda_cflags=ENV.get_build_cuda_cflags(),
      extra_cflags=ENV.get_build_cflags(),
      verbose=verbose,
    )

  @staticmethod
  def load(force_build: bool = False, verbose: bool = False):
    use_ffpa_attn_package = False
    if not force_build:
      # check if can import ffpa_attn
      try:
        import ffpa_attn

        _logging_msg("Import ffpa_attn library done, use it!")
        use_ffpa_attn_package = True
        return ffpa_attn, use_ffpa_attn_package
      except Exception:
        _logging_msg("Can't import ffpa_attn, force build from sources")
        _logging_msg(
          "Also may need export LD_LIBRARY_PATH="
          "PATH-TO/torch/lib:$LD_LIBRARY_PATH"
        )
        ffpa_attn = ENV.build(verbose=verbose)
        use_ffpa_attn_package = False
        return ffpa_attn, use_ffpa_attn_package
    else:
      _logging_msg("Force ffpa_attn lib build from sources")
      ffpa_attn = ENV.build(verbose=verbose)
      use_ffpa_attn_package = False
      return ffpa_attn, use_ffpa_attn_package


def _logging_msg(
  m: str = "", sep: str = "-", mode: str = "center", width: int = 150
):
  res_len = width - len(m)
  if mode == "center":
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
  elif mode == "left":
    pretty_line = m + sep * res_len
  else:
    pretty_line = sep * res_len + m
  print(pretty_line)


if __name__ == "__main__":
  # Debug: show FFPA ENV information. run: python3 env.py
  ENV.list_ffpa_env()
