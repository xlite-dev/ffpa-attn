from pathlib import Path

from env import ENV
from packaging.version import Version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension
import warnings
warnings.filterwarnings("ignore")


def get_long_description():
    description = (Path(ENV.project_dir()) / "README.md").read_text(encoding="utf-8")
    # replace relative repository path to absolute link to the release
    static_url = "https://github.com/DefTruth/cuffpa-py/blob/main"
    description = description.replace("docs/", f"{static_url}/docs/")
    return description


# package name managed by pip, which can be remove by `pip uninstall cuffpa-py -y`
PACKAGE_NAME = "cuffpa-py"

ext_modules = []
generator_flag = []
cc_flag = []

ENV.list_ffpa_env()

if ENV.enable_ampere():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")

if ENV.enable_ada():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_89,code=sm_89")

if ENV.enable_hopper():
    if CUDA_HOME is not None:
        _, bare_metal_version = ENV.get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

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
            "nvcc": ENV.get_build_cuda_cflags(build_pkg=True)
            + generator_flag
            + cc_flag,
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
    version="0.0.1",
    author="DefTruth",
    author_email="qyjdef@163.com",
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
            "__pycache__",
            "third_party",
        )
    ),
    description="FFPA: Yet another Faster Flash Prefill Attention for large headdim, ~1.5x faster than SDPA EA.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/cuffpa-py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=fetch_requirements(),
    extras_require={
        # optional dependencies, required by some features
        "all": [],
        # dev dependencies. Install them by `pip3 install 'akvattn[dev]'`
        "dev": [
            "pre-commit",
            "packaging",
            "ninja",
        ],
    },
)
