path=$(cd `dirname $0`; pwd)
cd $path

# cpp & python format lint
sudo apt-get update
sudo apt-get install clang-format -y
pip install pre-commit
pip install yapf
pip install cpplint
pre-commit install -c ./.dev/.pre-commit-config.yaml # only lint for python
# pre-commit install -c ./.dev/.pre-commit-config-cpp.yaml # both python + cpp
