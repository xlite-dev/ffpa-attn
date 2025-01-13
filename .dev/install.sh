rm -rf $(find . -name __pycache__)
python3 setup.py bdist_wheel && cd dist 
python3 -m pip install *.whl 
cd .. && rm -rf build *.egg-info 
rm -rf $(find . -name __pycache__)
