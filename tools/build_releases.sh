# Build release packages for ffpa-attn.
# Conda ENV list: py310, py311, py312, py313, py314

for env in py310 py311 py312 py313 py314; do
  conda activate $env && \
    export FFPA_BUILD_ARCH=80,89,90,100,120 && \
    bash tools/build_fast.sh bdist_wheel
done

# Usage:
# 1. Build the release packages by running this script.
# 2. The built wheel files will be located in the dist/ directory.
# 3. You can then upload the wheel files to PyPI using twine.
# Example build command:
# bash tools/build_releases.sh
# Example upload command:
# twine upload dist/*.whl
