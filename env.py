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

  # Enable all headdims for FFPA kernels or not, default False.
  # True, headdim will range from 32 to 1024 with step = 32, range(32, 1024, 32)
  # False, headdim will range from 64 to 1024 with step = 64, range(64, 1024, 64)
  # (only multiples of 64). Pass other headdims via FFPA_DEV_HEADDIMS /
  # `build_fast.sh --headdim`.
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

    assert not all((cls.enable_persist_q_s2r(), cls.enable_persist_q_g2s())
                   ), "PERSIST_Q_G2S and PERSIST_Q_S2R can not both enabled."
    assert not all((cls.enable_qkv_smem_share(), cls.enable_persist_q_g2s())
                   ), "PERSIST_Q_G2S and QKV_SMEM_SHARE can not both enabled."
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
        "range(32, 1024, 32)"
        if cls.enable_all_headdim() else "range(64, 1024, 64)"
      )
    )
    formatenv("ENABLE_FFPA_ALL_STAGES", cls.enable_all_mutistages())
    formatenv("FFPA_BUILD_MAX_STAGES", cls.FFPA_BUILD_MAX_STAGES)
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
    iteration) -> ``ENABLE_FFPA_ALL_HEADDIM`` (multiples of 32 in
    ``[32, 1024]``) -> default (multiples of 64 in ``[256, 1024]``).

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
      return list(range(32, 1025, 32))
    return list(range(64, 1025, 64))

  @classmethod
  def generated_sources_dir(cls):
    return os.path.join(cls.project_dir(), "csrc", "cuffpa", "generated")

  @staticmethod
  def _write_file(path: str, content: str):
    """Write ``content`` to ``path``, creating parent dirs as needed."""
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)

  @classmethod
  def generate_split_headdim_sources(cls, build_pkg: bool = False):
    """Generate per-(variant, headdim, stage) TUs under ``csrc/cuffpa/generated/``.

    Layout (variant ∈ {fp16f16 (only with ENABLE_FFPA_F16_ACC), fp16f32,
    bf16f32}):

    - ``fwd_<variant>_hdim{d}.cu``: lightweight wrapper TU. Includes only
      ``fwd_decls.h`` (NOT ``launch.cuh``); dispatches on ``stages`` to the
      per-stage symbols ``ffpa_attn_fwd_<variant>_d{d}_s{s}``. Keeps the
      original dispatch symbol name so ``fwd_dispatch.cu`` / ``ffpa_api.cc``
      are untouched.
    - ``fwd_<variant>_hdim{d}_s{s}.cu``: heavy TU. Includes ``launch.cuh``
      and contains a single ``launch_ffpa_attn_fwd_template`` instantiation
      per stage, so ``MAX_JOBS`` parallelism is no longer bottlenecked by a
      single TU serially compiling all stages.

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

      stages = cls._enabled_stages()
      variants = cls._enabled_variants()

      decls_path = os.path.join(gen_dir, "fwd_decls.h")
      cls._write_file(decls_path, cls._render_decls_header(headdims))
      generated.append(decls_path)

      for d in headdims:
        for variant, t_in, prefix in variants:
          wrapper_path = os.path.join(gen_dir, f"fwd_{variant}_hdim{d}.cu")
          cls._write_file(wrapper_path, cls._render_wrapper_tu(variant, d))
          generated.append(wrapper_path)
          for s in stages:
            stage_path = os.path.join(gen_dir, f"fwd_{variant}_hdim{d}_s{s}.cu")
            cls._write_file(
              stage_path, cls._render_stage_tu(variant, t_in, prefix, d, s)
            )
            generated.append(stage_path)
          fwd_generated_count += 1 + len(stages)

      dispatch_path = os.path.join(gen_dir, "fwd_dispatch.cu")
      cls._write_file(dispatch_path, cls._render_dispatch_tu(headdims))
      generated.append(dispatch_path)
      fwd_generated_count += 1

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

    ``ENABLE_FFPA_ALL_STAGES=1`` → ``1..FFPA_BUILD_MAX_STAGES``; ``=0`` →
    ``[1, 2]``. Stage 1 is always present (runtime fallback).
    """
    if cls.enable_all_mutistages():
      return list(range(1, cls.FFPA_BUILD_MAX_STAGES + 1))
    return [1, 2]

  @classmethod
  def _enabled_variants(cls):
    """Return ``(variant, t_in, constexpr_prefix)`` tuples to generate.

    fp16f16 is prepended only when ``ENABLE_FFPA_F16_ACC`` is on; the fp16f32
    / bf16f32 paths are always generated.
    """
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
      "    bool smooth_k)",
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
      "int64_t philox_seed, int64_t philox_offset, bool smooth_k"
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
      for d in headdims:
        lines.append(cls._decl(f"ffpa_attn_fwd_{variant}_d{d}", True))
        for s in stages:
          lines.append(cls._decl(f"ffpa_attn_fwd_{variant}_d{d}_s{s}", False))
    lines.append("")
    return "\n".join(lines)

  @classmethod
  def _render_wrapper_dispatch(cls, variant: str, d: int) -> str:
    """Render the ``if (stages == s) {...}`` chain calling per-stage symbols.

    Stage 1 is the fallback (covers ``stages == 1`` and any out-of-range
    value), mirroring the legacy dispatch semantics.
    """
    branches = [s for s in cls._enabled_stages() if s != 1]
    call = (
      "Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, "
      "dropout_p, philox_seed, philox_offset, smooth_k"
    )
    if not branches:
      return f"  ffpa_attn_fwd_{variant}_d{d}_s1({call});\n"
    lines = []
    for i, s in enumerate(branches):
      kw = "if" if i == 0 else "else if"
      lines.append(f"  {kw} (stages == {s}) {{")
      lines.append(f"    ffpa_attn_fwd_{variant}_d{d}_s{s}({call});")
      lines.append("  }")
    lines.append("  else {")
    lines.append(f"    ffpa_attn_fwd_{variant}_d{d}_s1({call});")
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

  @classmethod
  def _render_stage_tu(
    cls, variant: str, t_in: str, prefix: list, d: int, s: int
  ) -> str:
    """Heavy stage TU: one ``launch_ffpa_attn_fwd_template`` instantiation."""
    body = list(prefix)
    body.append(
      f"  launch_ffpa_attn_fwd_template<{t_in}, {d}, kMmaAccFloat32QK, "
      f"kMmaAccFloat32PV, {s}>(Q, K, V, O, attn_bias, softmax_lse, causal, "
      "softmax_scale, dropout_p, philox_seed, philox_offset, smooth_k);"
    )
    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      '#include "launch.cuh"\n'
      "using namespace ffpa;\n\n" +
      cls._signature(f"ffpa_attn_fwd_{variant}_d{d}_s{s}", False) + "\n" +
      "\n".join(body) + "\n}\n"
    )

  @classmethod
  def _render_dispatch_tu(cls, headdims) -> str:
    # fp16f16 (acc=0) dispatch is only emitted when ENABLE_FFPA_F16_ACC is on.
    specs = [
      ("ffpa_attn_fwd_fp16f32", "torch::kHalf"),
      ("ffpa_attn_fwd_bf16f32", "torch::kBFloat16"),
    ]
    if cls.enable_f16_acc():
      specs.insert(0, ("ffpa_attn_fwd_fp16f16", "torch::kHalf"))

    out = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "logging.cuh"',
      '#include "fwd_decls.h"',
      "",
    ]
    for name, dtype in specs:
      out.append(cls._signature(name, True))
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(Q, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(K, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(V, {dtype})")
      out.append(f"  CHECK_TORCH_TENSOR_DTYPE(O, {dtype})")
      out.append("  const int d = Q.size(3);")
      out.append("  switch (d) {")
      for d in headdims:
        out.append(
          f"    case {d}: {name}_d{d}(Q, K, V, O, attn_bias, softmax_lse, "
          "stages, causal, softmax_scale, dropout_p, philox_seed, "
          "philox_offset, smooth_k); break;"
        )
      out.append(
        '    default: throw std::runtime_error("headdim not support!");'
      )
      out.append("  }")
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
    # the actual build sources. The generated TUs include launch.cuh,
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
