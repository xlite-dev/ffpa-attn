rm -rf $(find . -name __pycache__)
python3 setup.py bdist_wheel && cd dist # build cuffpa-py from sources
python3 -m pip install cuffpa-py-*-linux_x86_64.whl # pip uninstall cuffpa-py -y
cd .. && rm -rf build *.egg-info 
rm -rf $(find . -name __pycache__)
