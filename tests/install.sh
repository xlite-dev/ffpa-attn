python3 setup.py bdist_wheel && cd dist # build ffpa-attn from sources
python3 -m pip install ffpa_attn-*-linux_x86_64.whl # pip uninstall ffpa-attn -y 
cd .. && rm -rf build *.egg-info
