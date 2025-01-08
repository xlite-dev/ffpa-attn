import os


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


class ENV(object):
    # ENVs for pyffpa compiling

    # Project dir, path to faster-prefill-attention
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Enable all multi stages kernels or not (1~4), default False (1~2).
    ENABLE_FFPA_ALL_STAGES = bool(int(os.environ.get("ENABLE_FFPA_ALL_STAGES", 0)))

    # Enable all headdims for FFPA kernels or not, default False.
    # True, headdim will range from 32 to 1024 with step = 32, range(32, 1024, 32)
    # False, headdim will range from 256 to 1024 with step = 64, range(256, 1024, 64)
    ENABLE_FFPA_ALL_HEADDIM = bool(int(os.environ.get("ENABLE_FFPA_ALL_HEADDIM", 0)))

    # Enable build FFPA kernels for Ada devices (sm89, L2O, 4090, etc),
    # default True.
    ENABLE_FFPA_ADA = bool(int(os.environ.get("ENABLE_FFPA_ADA", 1)))

    # Enable build FFPA kernels for Ampere devices (sm80, A30, A100, etc),
    # default True.
    ENABLE_FFPA_AMPERE = bool(int(os.environ.get("ENABLE_FFPA_AMPERE", 1)))

    # Enable build FFPA kernels for Hopper devices (sm90, H100, H20, etc),
    # default False.
    ENABLE_FFPA_HOPPER = bool(int(os.environ.get("ENABLE_FFPA_HOPPER", 0)))

    # Enable debug mode for FFPA, fast build minimal kernels, default False.
    ENABLE_FFPA_DEBUG = bool(int(os.environ.get("ENABLE_FFPA_DEBUG", 0)))

    @classmethod
    def project_dir(cls):
        return cls.PROJECT_DIR

    @classmethod
    def enable_hopper(cls):
        return cls.ENABLE_FFPA_HOPPER

    @classmethod
    def enable_ampere(cls):
        return cls.ENABLE_FFPA_AMPERE

    @classmethod
    def enable_ada(cls):
        return cls.ENABLE_FFPA_ADA

    @classmethod
    def enable_all_mutistages(cls):
        return cls.ENABLE_FFPA_ALL_STAGES

    @classmethod
    def enable_all_headdim(cls):
        return cls.ENABLE_FFPA_ALL_HEADDIM

    @classmethod
    def enable_debug(cls):
        return cls.ENABLE_FFPA_DEBUG

    @classmethod
    def env_cuda_cflags(cls):
        extra_env_cflags = []
        if cls.enable_all_mutistages():
            extra_env_cflags.append("-DENABLE_FFPA_ALL_STAGES")
        if cls.enable_all_headdim():
            extra_env_cflags.append("-DENABLE_FFPA_ALL_HEADDIM")
        if cls.enable_debug():
            extra_env_cflags.append("-DENABLE_FFPA_DEBUG")
        return extra_env_cflags

    @classmethod
    def list_ffpa_env(cls):
        pretty_print_line("cuffpa-py ENVs")
        pretty_print_line(f"PROJECT_DIR: {cls.project_dir()}")
        pretty_print_line(f"ENABLE_FFPA_DEBUG: {cls.enable_debug()}")
        pretty_print_line(f"ENABLE_FFPA_ADA: {cls.enable_ada()}")
        pretty_print_line(f"ENABLE_FFPA_AMPERE: {cls.enable_ampere()}")
        pretty_print_line(f"ENABLE_FFPA_HOPPER: {cls.enable_hopper()}")
        pretty_print_line(f"ENABLE_FFPA_ALL_STAGES: {cls.enable_all_mutistages()}")
        pretty_print_line(f"ENABLE_FFPA_ALL_HEADDIM: {cls.enable_all_headdim()}")
        pretty_print_line("-")
