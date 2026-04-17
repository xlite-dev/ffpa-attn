import os

import torch


class ENV(object):
  # ENVs for FFPA kernels compiling

  # Project dir, path to faster-prefill-attention
  PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

  # Enable debug mode for FFPA, fast build minimal kernels, default False.
  ENABLE_FFPA_DEBUG = bool(int(os.environ.get("ENABLE_FFPA_DEBUG", 0)))

  # Enable build FFPA kernels for Ada devices (sm89, L2O, 4090, etc),
  # default True.
  ENABLE_FFPA_ADA = bool(int(os.environ.get("ENABLE_FFPA_ADA", 1)))

  # Enable build FFPA kernels for Ampere devices (sm80, A30, A100, etc),
  # default True.
  ENABLE_FFPA_AMPERE = bool(int(os.environ.get("ENABLE_FFPA_AMPERE", 1)))

  # Enable build FFPA kernels for Hopper devices (sm90, H100, H20, etc),
  # default False.
  ENABLE_FFPA_HOPPER = bool(int(os.environ.get("ENABLE_FFPA_HOPPER", 0)))

  # Enable all multi stages kernels or not, if True (1~4) else (1~2), default True.
  ENABLE_FFPA_ALL_STAGES = bool(int(os.environ.get("ENABLE_FFPA_ALL_STAGES", 1)))

  # Enable all headdims for FFPA kernels or not, default False.
  # True, headdim will range from 32 to 1024 with step = 32, range(32, 1024, 32)
  # False, headdim will range from 256 to 1024 with step = 64, range(256, 1024, 64)
  ENABLE_FFPA_ALL_HEADDIM = bool(int(os.environ.get("ENABLE_FFPA_ALL_HEADDIM", 0)))

  # Enable force Q@K^T use fp16 as MMA Acc dtype for FFPA Acc F32 kernels, default False.
  # FFPA Acc F32 kernels MMA Acc = Mixed Q@K^T MMA Acc F16 + P@V MMA Acc F32.
  ENABLE_FFPA_FORCE_QK_F16 = bool(int(os.environ.get("ENABLE_FFPA_FORCE_QK_F16", 0)))

  # Enable force P@V use fp16 as MMA Acc dtype, for FFPA Acc F32 kernels, default False.
  # FFPA Acc F32 kernels MMA Acc = Mixed Q@K^T MMA Acc F32 + P@V MMA Acc F16.
  ENABLE_FFPA_FORCE_PV_F16 = bool(int(os.environ.get("ENABLE_FFPA_FORCE_PV_F16", 0)))

  # Enable FFPA Prefetch QKV at the Appropriate Time Point, default True, boost 5%~10%.
  ENABLE_FFPA_PREFETCH_QKV = bool(int(os.environ.get("ENABLE_FFPA_PREFETCH_QKV", 1)))

  # Enable QKV smem shared policy, default False (perfered for MMA & g2s overlap).
  # Please, set it as True if you want to run FFPA on low SRAM device.
  ENABLE_FFPA_QKV_SMEM_SHARE = bool(int(os.environ.get("ENABLE_FFPA_QKV_SMEM_SHARE", 0)))

  # Enable smem swizzle for Q, default True. True: bank conflicts free for Q smem
  # via swizzle; False: bank conflicts free for Q smem via padding.
  ENABLE_FFPA_SMEM_SWIZZLE_Q = bool(int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_Q", 1)))

  # Enable smem swizzle for K, default True. True: bank conflicts free for K smem
  # via swizzle; False: bank conflicts free for K smem via padding.
  ENABLE_FFPA_SMEM_SWIZZLE_K = bool(int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_K", 1)))

  # Enable smem swizzle for V, now default True. True: bank conflicts free for V smem
  # via swizzle; False: bank conflicts free for V smem via padding. FIXME(DefTruth):
  # swizzle V seems can not get good performance. why? Will enable it by default untill
  # I have fixed the performance issue. (Fixed)
  ENABLE_FFPA_SMEM_SWIZZLE_V = bool(int(os.environ.get("ENABLE_FFPA_SMEM_SWIZZLE_V", 1)))

  # Persist load Q g2s for headdim <= 320, more SRAM, but still keep register usage.
  ENABLE_FFPA_PERSIST_Q_G2S = bool(int(os.environ.get("ENABLE_FFPA_PERSIST_Q_G2S", 0)))

  # Persist load KV g2s for headdim <= 256, more SRAM. If True, auto use flash-attn
  # algo that tiling at attention level for headdim <= 256 and auto use ffpa-attn
  # fined-grain tiling at MMA level for headdim > 256.
  ENABLE_FFPA_PERSIST_KV_G2S = bool(int(os.environ.get("ENABLE_FFPA_PERSIST_KV_G2S", 0)))

  # Persist load Q from s2r for headdim < 512 to reduce Q from g2s and s2r IO access,
  # but still keep O(1) SRAM complexity. Default value is False. This option will
  # introduce more registers for Q frags as the headdim becomes larger. We should
  # choose to enable it or not according to the balance between register usage and
  # IO access reduction.
  ENABLE_FFPA_PERSIST_Q_S2R = bool(int(os.environ.get("ENABLE_FFPA_PERSIST_Q_S2R", 0)))

  # Persist V s2r only for small d kernel, more registers.
  ENABLE_FFPA_PERSIST_V_S2R = bool(int(os.environ.get("ENABLE_FFPA_PERSIST_V_S2R", ENABLE_FFPA_PERSIST_KV_G2S)))

  # Registers Ping pong double buffers for ldmatrix & mma computation overlapping.
  ENABLE_FFPA_REGISTERS_PIPE_KV = bool(int(os.environ.get("ENABLE_FFPA_REGISTERS_PIPE_KV", 0)))

  # if True: grid(N/Br, H, B) else: grid(N/Br, B * H)
  ENABLE_FFPA_LAUNCH_GRID_DNHB = bool(int(os.environ.get("ENABLE_FFPA_LAUNCH_GRID_DNHB", 0)))

  @classmethod
  def project_dir(cls):
    return cls.PROJECT_DIR

  @classmethod
  def enable_debug(cls):
    return cls.ENABLE_FFPA_DEBUG

  @classmethod
  def enable_ada(cls):
    return cls.ENABLE_FFPA_ADA

  @classmethod
  def enable_ampere(cls):
    return cls.ENABLE_FFPA_AMPERE

  @classmethod
  def enable_hopper(cls):
    return cls.ENABLE_FFPA_HOPPER

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
  def enable_persist_kv_g2s(cls):
    return cls.ENABLE_FFPA_PERSIST_KV_G2S

  @classmethod
  def enable_persist_q_s2r(cls):
    return cls.ENABLE_FFPA_PERSIST_Q_S2R

  @classmethod
  def enable_persist_v_s2r(cls):
    if cls.enable_persist_kv_g2s():
      return cls.ENABLE_FFPA_PERSIST_V_S2R
    return False

  @classmethod
  def enable_registers_pipe_kv(cls):
    return cls.ENABLE_FFPA_REGISTERS_PIPE_KV

  @classmethod
  def enable_launch_grid_dnhb(cls):
    return cls.ENABLE_FFPA_LAUNCH_GRID_DNHB

  @classmethod
  def env_cuda_cflags(cls):
    extra_env_cflags = []
    if cls.enable_debug():
      extra_env_cflags.append("-DENABLE_FFPA_DEBUG")
    if cls.enable_all_mutistages():
      extra_env_cflags.append("-DENABLE_FFPA_ALL_STAGES")
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
    if cls.enable_persist_kv_g2s():
      extra_env_cflags.append("-DENABLE_FFPA_PERSIST_KV_G2S")
    if cls.enable_persist_q_s2r():
      extra_env_cflags.append("-DENABLE_FFPA_PERSIST_Q_S2R")
    if cls.enable_persist_v_s2r():
      extra_env_cflags.append("-DENABLE_FFPA_PERSIST_V_S2R")
    if cls.enable_registers_pipe_kv():
      extra_env_cflags.append("-DENABLE_FFPA_REGISTERS_PIPE_KV")
    if cls.enable_launch_grid_dnhb():
      extra_env_cflags.append("-DENBALE_FFPA_LAUNCH_GRID_DNHB")

    if cls.enable_persist_kv_g2s():
      assert (cls.enable_persist_q_g2s()), "PERSIST_Q_G2S must be enable if PERSIST_KV_G2S is enabled."
      if cls.enable_qkv_smem_share():
        assert (cls.enable_persist_q_s2r()), "PERSIST_Q_S2R must be enable if QKV_SMEM_SHARE and "
        "PERSIST_KV_G2S are enabled."
    else:
      assert not all((cls.enable_persist_q_s2r(), cls.enable_persist_q_g2s())
                     ), "PERSIST_Q_G2S and PERSIST_Q_S2R can not both enabled."
      assert not all((cls.enable_qkv_smem_share(), cls.enable_persist_q_g2s())
                     ), "PERSIST_Q_G2S and QKV_SMEM_SHARE can not both enabled."
      assert not all((cls.enable_qkv_smem_share(), cls.enable_persist_kv_g2s())
                     ), "PERSIST_KV_G2S and QKV_SMEM_SHARE can not both enabled."
    return extra_env_cflags

  @classmethod
  def list_ffpa_env(cls):

    def formatenv(name, value):
      try:
        print(f"{name:<30}: {str(value):<5} -> command:"
              f" export {name}={int(value)}")
      except Exception:
        print(f"{name:<30}: {value}")

    pretty_print_line("FFPA-ATTN ENVs")
    formatenv("PROJECT_DIR", cls.project_dir())
    formatenv("ENABLE_FFPA_DEBUG", cls.enable_debug())
    formatenv("ENABLE_FFPA_ADA", cls.enable_ada())
    formatenv("ENABLE_FFPA_AMPERE", cls.enable_ampere())
    formatenv("ENABLE_FFPA_HOPPER", cls.enable_hopper())
    formatenv("ENABLE_FFPA_ALL_STAGES", cls.enable_all_mutistages())
    formatenv("ENABLE_FFPA_ALL_HEADDIM", cls.enable_all_headdim())
    formatenv("ENABLE_FFPA_PREFETCH_QKV", cls.enable_prefetch_qkv())
    formatenv("ENABLE_FFPA_FORCE_QK_F16", cls.enable_force_qk_fp16())
    formatenv("ENABLE_FFPA_FORCE_PV_F16", cls.enable_force_pv_fp16())
    formatenv("ENABLE_FFPA_PERSIST_Q_G2S", cls.enable_persist_q_g2s())
    formatenv("ENABLE_FFPA_PERSIST_KV_G2S", cls.enable_persist_kv_g2s())
    formatenv("ENABLE_FFPA_PERSIST_Q_S2R", cls.enable_persist_q_s2r())
    formatenv("ENABLE_FFPA_PERSIST_V_S2R", cls.enable_persist_v_s2r())
    formatenv("ENABLE_FFPA_QKV_SMEM_SHARE", cls.enable_qkv_smem_share())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_Q", cls.enable_smem_swizzle_q())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_K", cls.enable_smem_swizzle_k())
    formatenv("ENABLE_FFPA_SMEM_SWIZZLE_V", cls.enable_smem_swizzle_v())
    formatenv("ENABLE_FFPA_REGISTERS_PIPE_KV", cls.enable_registers_pipe_kv())
    formatenv("ENABLE_FFPA_LAUNCH_GRID_DNHB", cls.enable_launch_grid_dnhb())
    pretty_print_line()

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

        The set mirrors the ``DISPATCH_HEADDIM`` macro in ``launch_templates.cuh``:

        * ``ENABLE_FFPA_DEBUG``         -> minimal headdims for fast iteration.
        * ``ENABLE_FFPA_ALL_HEADDIM``   -> multiples of 32 in [32, 1024].
        * default                       -> multiples of 64 in [256, 1024].
        """
    if cls.enable_debug():
      return [32, 64, 128, 256, 320, 512, 1024]
    if cls.enable_all_headdim():
      return list(range(32, 1025, 32))
    return list(range(256, 1025, 64))

  @classmethod
  def generated_sources_dir(cls):
    return os.path.join(cls.project_dir(), "csrc", "cuffpa", "generated")

  @staticmethod
  def _write_if_changed(path: str, content: str):
    """Write ``content`` to ``path`` only if it differs from the current file.

        Avoids touching mtime when unchanged so ``setuptools`` / ``ninja`` can skip
        recompilation of TUs whose generated sources have not actually changed.
        """
    if os.path.exists(path):
      with open(path, "r", encoding="utf-8") as f:
        if f.read() == content:
          return
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)

  @classmethod
  def generate_split_headdim_sources(cls, build_pkg: bool = False):
    """Generate per-headdim ``.cu`` translation units under ``csrc/cuffpa/generated/``.

        Splitting the big ``DISPATCH_HEADDIM`` switch into one TU per headdim lets
        ``MAX_JOBS`` invoke nvcc on many files in parallel, dramatically reducing
        the wall-clock build time of the heavy ``launch_ffpa_mma_L1_template``
        instantiations.

        The generated files are committed to the repository (not ignored). On
        every build the generator refreshes their contents only when they would
        actually change, so under steady state this is a no-op and incremental
        builds stay fast.

        Returns the list of generated file paths (headers first, then ``.cu`` sources).
        """
    gen_dir = cls.generated_sources_dir()
    os.makedirs(gen_dir, exist_ok=True)

    headdims = cls.get_enabled_headdims()
    generated = []

    # ---- declarations header shared by per-D TUs + dispatch TU ----
    decls_path = os.path.join(gen_dir, "ffpa_attn_L1_decls.h")
    cls._write_if_changed(decls_path, cls._render_decls_header(headdims))
    generated.append(decls_path)

    # ---- one TU per (headdim, acc-dtype) ----
    for d in headdims:
      f16_path = os.path.join(gen_dir, f"ffpa_attn_L1_acc_f16_d{d}.cu")
      f32_path = os.path.join(gen_dir, f"ffpa_attn_L1_acc_f32_d{d}.cu")
      cls._write_if_changed(f16_path, cls._render_per_headdim_tu(d, "f16"))
      cls._write_if_changed(f32_path, cls._render_per_headdim_tu(d, "f32"))
      generated.append(f16_path)
      generated.append(f32_path)

    # ---- top-level dispatch TU: CHECK_* + switch(d) to per-D entry points ----
    dispatch_path = os.path.join(gen_dir, "ffpa_attn_L1_dispatch.cu")
    cls._write_if_changed(dispatch_path, cls._render_dispatch_tu(headdims))
    generated.append(dispatch_path)

    if cls.enable_debug() or build_pkg:
      pretty_print_line(
        f"Generated {len(headdims)} x 2 per-headdim TUs under {gen_dir}",
        sep="",
        mode="left",
      )

    return generated

  # -------------------- code generation helpers --------------------

  @staticmethod
  def _render_decls_header(headdims):
    lines = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      "#pragma once",
      "#include <torch/types.h>",
      "",
    ]
    for d in headdims:
      lines.append(
        f"void ffpa_mma_acc_f16_L1_d{d}(torch::Tensor Q, torch::Tensor K, "
        f"torch::Tensor V, torch::Tensor O, int stages);"
      )
      lines.append(
        f"void ffpa_mma_acc_f32_L1_d{d}(torch::Tensor Q, torch::Tensor K, "
        f"torch::Tensor V, torch::Tensor O, int stages);"
      )
    lines.append("")
    return "\n".join(lines)

  @staticmethod
  def _render_stage_body(d: int, qk: str, pv: str) -> str:
    """Render the stage-dispatch body at body scope (2-space indent).

        Preprocessor directives intentionally start at column 0 as required by
        strict compilers, while normal statements use the 2-space indent that
        matches the surrounding function body.
        """
    call = (f"launch_ffpa_mma_L1_template<{d}, {qk}, {pv}, "
            "{S}>(Q, K, V, O);")
    return (
      "#ifdef ENABLE_FFPA_ALL_STAGES\n"
      "  if (stages == 2) {\n"
      f"    {call.replace('{S}', '2')}\n"
      "  } else if (stages == 3) {\n"
      f"    {call.replace('{S}', '3')}\n"
      "  } else if (stages == 4) {\n"
      f"    {call.replace('{S}', '4')}\n"
      "  } else {\n"
      f"    {call.replace('{S}', '1')}\n"
      "  }\n"
      "#else\n"
      "  if (stages == 2) {\n"
      f"    {call.replace('{S}', '2')}\n"
      "  } else {\n"
      f"    {call.replace('{S}', '1')}\n"
      "  }\n"
      "#endif\n"
    )

  @classmethod
  def _render_per_headdim_tu(cls, d: int, acc: str) -> str:
    assert acc in ("f16", "f32")
    header = [
      "// AUTO-GENERATED by env.py. DO NOT EDIT.",
      '#include "launch_templates.cuh"',
      "using namespace ffpa;",
      "",
      f"void ffpa_mma_acc_{acc}_L1_d{d}(",
      "    torch::Tensor Q,",
      "    torch::Tensor K,",
      "    torch::Tensor V,",
      "    torch::Tensor O,",
      "    int stages) {",
    ]
    if acc == "f16":
      body_prefix = [
        "  constexpr int kMmaAccFloat32QK = 0;",
        "  constexpr int kMmaAccFloat32PV = 0;",
      ]
    else:
      body_prefix = [
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
    stage_body = cls._render_stage_body(d, "kMmaAccFloat32QK", "kMmaAccFloat32PV")
    return ("\n".join(header) + "\n" + "\n".join(body_prefix) + "\n" + stage_body + "}\n")

  @staticmethod
  def _render_dispatch_tu(headdims) -> str:

    def _cases(acc: str) -> str:
      return "\n".join(f"    case {d}: ffpa_mma_acc_{acc}_L1_d{d}"
                       "(Q, K, V, O, stages); break;" for d in headdims)

    def _fn(name: str, acc: str) -> str:
      return (
        f"void {name}(\n"
        "    torch::Tensor Q,\n"
        "    torch::Tensor K,\n"
        "    torch::Tensor V,\n"
        "    torch::Tensor O,\n"
        "    int stages) {\n"
        "  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)\n"
        "  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)\n"
        "  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)\n"
        "  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)\n"
        "  const int d = Q.size(3);\n"
        "  switch (d) {\n"
        f"{_cases(acc)}\n"
        '    default: throw std::runtime_error("headdim not support!");\n'
        "  }\n"
        "}\n"
      )

    return (
      "// AUTO-GENERATED by env.py. DO NOT EDIT.\n"
      '#include "cuffpa/logging.cuh"\n'
      '#include "ffpa_attn_L1_decls.h"\n'
      "\n" + _fn("ffpa_mma_acc_f16_L1", "f16") + "\n" + _fn("ffpa_mma_acc_f32_L1", "f32")
    )

  @staticmethod
  def get_build_sources(build_pkg: bool = False):

    def csrc(sub_dir, filename):
      csrc_file = f"{ENV.project_dir()}/csrc/{sub_dir}/{filename}"
      if ENV.enable_debug() or build_pkg:
        pretty_print_line(f"csrc_file: {csrc_file}", sep="", mode="left")
      return csrc_file

    if ENV.enable_debug() or build_pkg:
      pretty_print_line()
    # Generate per-headdim TUs under csrc/cuffpa/generated/ and use them as
    # the actual build sources. Splitting by headdim enables MAX_JOBS to
    # drive nvcc on many small files in parallel and cuts the build time
    # of the heavy launch_ffpa_mma_L1_template instantiations.
    generated_files = ENV.generate_split_headdim_sources(build_pkg=build_pkg)
    generated_sources = [p for p in generated_files if p.endswith(".cu")]
    if ENV.enable_debug() or build_pkg:
      for gs in generated_sources:
        pretty_print_line(f"csrc_file: {gs}", sep="", mode="left")
    build_sources = [csrc("pybind", "ffpa_attn_api.cc")] + generated_sources
    if ENV.enable_debug() or build_pkg:
      pretty_print_line()
    return build_sources

  @staticmethod
  def get_build_cuda_cflags(build_pkg: bool = False):
    device_name = ENV.get_device_name()

    def _specific_device_tag():
      if "L20" in device_name:
        return "L20"
      elif "4090" in device_name:
        return "4090"
      elif "3080" in device_name:
        return "3080"
      return None

    def _specific_device_macro():
      tag = _specific_device_tag()
      if tag is not None:
        return f"-DBUILD_FFPA_ATTN_MMA_{tag}"
      return None

    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-Xcompiler")
    extra_cuda_cflags.append("-fPIC")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    extra_cuda_cflags.append(_specific_device_macro())
    extra_cuda_cflags.extend(ENV.env_cuda_cflags())
    extra_cuda_cflags.append(f"-I {ENV.project_dir()}/include")
    extra_cuda_cflags.append(f"-I {ENV.project_dir()}/csrc/cuffpa")
    extra_cuda_cflags.append("-diag-suppress 177" if not build_pkg else "--ptxas-options=-v")
    extra_cuda_cflags.append("-Xptxas -v" if not build_pkg else "--ptxas-options=-O3")

    extra_cuda_cflags.append("--threads=8")
    # Avoid None or empty str as flag or macro
    extra_cuda_cflags = [flag for flag in extra_cuda_cflags if flag]
    return extra_cuda_cflags

  @staticmethod
  def get_build_cflags():
    extra_cflags = []
    extra_cflags.append("-std=c++17")
    return extra_cflags

  @staticmethod
  def get_cuda_bare_metal_version(cuda_dir):
    # helper function to get cuda version
    import subprocess

    from packaging.version import parse

    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

  @staticmethod
  def build_ffpa_from_sources(verbose: bool = False):
    from torch.utils.cpp_extension import load

    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    # Load the CUDA kernel as a python module
    pretty_print_line(
      f"Loading ffpa_attn lib on device: {ENV.get_device_name()}, "
      f"capability: {ENV.get_device_capability()}, "
      f"Arch ENV: {torch_arch_list_env}"
    )
    return load(
      name="pyffpa_cuda",
      sources=ENV.get_build_sources(),
      extra_cuda_cflags=ENV.get_build_cuda_cflags(),
      extra_cflags=ENV.get_build_cflags(),
      verbose=verbose,
    )

  @staticmethod
  def try_load_ffpa_library(force_build: bool = False, verbose: bool = False):
    use_ffpa_attn_package = False
    if not force_build:
      # check if can import ffpa_attn
      try:
        import ffpa_attn

        pretty_print_line("Import ffpa_attn library done, use it!")
        use_ffpa_attn_package = True
        return ffpa_attn, use_ffpa_attn_package
      except Exception:
        pretty_print_line("Can't import ffpa_attn, force build from sources")
        pretty_print_line("Also may need export LD_LIBRARY_PATH="
                          "PATH-TO/torch/lib:$LD_LIBRARY_PATH")
        ffpa_attn = ENV.build_ffpa_from_sources(verbose=verbose)
        use_ffpa_attn_package = False
        return ffpa_attn, use_ffpa_attn_package
    else:
      pretty_print_line("Force ffpa_attn lib build from sources")
      ffpa_attn = ENV.build_ffpa_from_sources(verbose=verbose)
      use_ffpa_attn_package = False
      return ffpa_attn, use_ffpa_attn_package


def pretty_print_line(m: str = "", sep: str = "-", mode: str = "center", width: int = 150):
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
