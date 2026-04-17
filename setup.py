import warnings
from pathlib import Path

from env import ENV
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

warnings.filterwarnings("ignore")


def get_long_description():
  description = (Path(ENV.project_dir()) / "README.md").read_text(encoding="utf-8")
  return description


# package name managed by pip, which can be remove by `pip uninstall ffpa-attn -y`
PACKAGE_NAME = "ffpa-attn"

ext_modules = []
generator_flag = []
cc_flag = []

ENV.list_ffpa_env()

# Expand the FFPA_BUILD_ARCH list (or the current device's SM if unset) into
# nvcc -gencode flags. Replaces the old ENABLE_FFPA_{ADA,AMPERE,HOPPER}
# switches with a cache-dit-style arch list (see cache-dit/setup.py).
for _sm in ENV.get_build_arch_list():
  cc_flag.append("-gencode")
  cc_flag.append(f"arch=compute_{_sm},code=sm_{_sm}")

assert cc_flag is not None, "cc_flag can not be NoneType."

# cuda module
# may need export LD_LIBRARY_PATH=PATH-TO/torch/lib:$LD_LIBRARY_PATH
ext_modules.append(
  CUDAExtension(
    # package name for import
    name="pyffpa_cuda",
    sources=ENV.get_build_sources(build_pkg=True),
    extra_compile_args={
      # add c compile flags
      "cxx": ["-O3", "-std=c++17"] + generator_flag,
      # add nvcc compile flags
      "nvcc": [
        flag for flag in (ENV.get_build_cuda_cflags(build_pkg=True) + generator_flag + cc_flag)
        if flag.strip()  # <--- Filter out empty strings
      ],
    },
    include_dirs=[
      Path(ENV.project_dir()) / "include",
    ],
  )
)


def fetch_requirements():
  with open("requirements.txt") as f:
    reqs = f.read().strip().split("\n")
  return reqs


setup(
  name=PACKAGE_NAME,
  version="0.0.2.1",
  author="DefTruth",
  author_email="qyjdef@163.com",
  license="GNU General Public License v3.0",
  packages=find_packages(
    exclude=(
      "build",
      "dist",
      "include",
      "csrc",
      "tests",
      "bench",
      "tmp",
      "cuffpa_py.egg-info",
      "ffpa_attn.egg-info",
      "__pycache__",
      "third_party",
    )
  ),
  description="FFPA: Yet another Faster Flash Prefill Attention for large headdim, 1.8x~3x faster than SDPA EA.",
  long_description=get_long_description(),
  long_description_content_type="text/markdown",
  url="https://github.com/xlite/ffpa-attn.git",
  ext_modules=ext_modules,
  cmdclass={"build_ext": BuildExtension},
  python_requires=">=3.10",
  install_requires=fetch_requirements(),
  extras_require={
    "all": [],
    "dev": [
      "pre-commit",
      "packaging",
      "ninja",
    ],
  },
)
