python3 setup.py bdist_wheel && cd dist # build pyffpa from sources
python3 -m pip install pyffpa-*-linux_x86_64.whl # pip uninstall pyffpa -y
cd .. && rm -rf build *.egg-info
