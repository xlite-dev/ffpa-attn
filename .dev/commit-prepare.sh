path=$(cd `dirname $0`; pwd)
cd $path

sudo apt-get install clang-format -y
pip install pre-commit
pip install yapf
pip install cpplint
pre-commit install -c ./.dev/.pre-commit-config.yaml
