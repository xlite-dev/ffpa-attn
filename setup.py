"""ffpa-attn build script.

Most package metadata (name, version, dependencies, extras, URLs, etc.) lives
in ``pyproject.toml``. This ``setup.py`` exists only to drive the optional
``CUDAExtension`` build for the ``ffpa_attn._C`` C++/CUDA module via the
PyTorch build helpers.

Behavior:
- Default: build the CUDA extension via ``torch.utils.cpp_extension``.
- ``FFPA_SKIP_CUDA_EXT=1``: skip CUDA extension entirely (used by the
  ReadTheDocs build and by the docs CI runner, which have no CUDA toolchain
  but still need to import ``ffpa_attn`` so ``mkdocstrings`` can render API
  documentation from docstrings).
"""

import os
import sys
import warnings
from pathlib import Path

from setuptools import setup

warnings.filterwarnings("ignore")

# Ensure the project root (containing ``env.py``) is on ``sys.path`` so that
# the build backend (``setuptools.build_meta``) can import ``env`` even when
# pip invokes setup.py from a different working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))


def _env_flag(name: str) -> bool:
  return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


SKIP_CUDA_EXT = _env_flag("FFPA_SKIP_CUDA_EXT")

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_EXT:
  from env import ENV
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension

  ENV.list_ffpa_env()

  cc_flag = []
  for _sm in ENV.get_build_arch_list():
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{_sm},code=sm_{_sm}")

  ext_modules.append(
    CUDAExtension(
      # Package-internal C extension module; imported as ``ffpa_attn._C``.
      name="ffpa_attn._C",
      sources=[
        # Convert to repo-relative paths; setuptools rejects absolute paths
        # in ``sources`` for editable installs (``pip install -e .``).
        os.path.relpath(s, _ROOT) for s in ENV.get_build_sources(build_pkg=True)
      ],
      extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [flag for flag in (ENV.get_build_cuda_cflags(build_pkg=True) + cc_flag) if flag.strip()],
      },
      include_dirs=[
        Path(ENV.project_dir()) / "include",
      ],
    )
  )
  cmdclass["build_ext"] = BuildExtension

setup(
  ext_modules=ext_modules,
  cmdclass=cmdclass,
)
