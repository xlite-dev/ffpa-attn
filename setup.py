import subprocess
from pathlib import Path

import torch
from env import ENV
from packaging.version import parse, Version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension


def get_long_description():
    description = (Path(ENV.project_dir()) / "README.md").read_text(encoding="utf-8")
    # replace relative repository path to absolute link to the release
    static_url = "https://github.com/DefTruth/faster-prefill-attention/blob/main"
    description = description.replace("docs/", f"{static_url}/docs/")
    return description


def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def get_build_sources():
    build_sources = [
        f"{ENV.project_dir()}/csrc/pybind/faster_prefill_attn_api.cc",
        f"{ENV.project_dir()}/csrc/deprecated/faster_prefill_attn_F16F16F16F16_L1.cu",
        f"{ENV.project_dir()}/csrc/deprecated/faster_prefill_attn_F32F16F16F32_L1.cu",
    ]
    return build_sources


def get_build_cuda_cflags(build_pkg: bool = False):
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    extra_cuda_cflags.append(
        "-diag-suppress 177" if not build_pkg else "--ptxas-options=-v"
    )
    extra_cuda_cflags.append("-Xptxas -v" if not build_pkg else "--ptxas-options=-O3")
    extra_cuda_cflags.extend(ENV.env_cuda_cflags())
    extra_cuda_cflags.append(f"-I {ENV.project_dir()}/include")
    return extra_cuda_cflags


# helper function to get cuda version
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


# package name managed by pip, which can be remove by `pip uninstall cuffpa-py -y`
PACKAGE_NAME = "cuffpa-py"

ext_modules = []
generator_flag = []
cc_flag = []

if ENV.enable_ampere():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")

if ENV.enable_ada():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_89,code=sm_89")

if ENV.enable_hopper():
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
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
        sources=get_build_sources(),
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": get_build_cuda_cflags(build_pkg=True) + generator_flag + cc_flag,
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
            "pyffpa.egg-info",
        )
    ),
    description="FFPA: Yet another Faster Flash Prefill Attention for large headdim, ~1.5x faster than SDPA EA.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/faster-prefill-attention",
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
