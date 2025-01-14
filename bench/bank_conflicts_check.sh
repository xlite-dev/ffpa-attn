cd tests
ncu \
  --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm \
  python3 test.py --B 1 --H 8 --N 1024 --D 320 --w 0 --i 1 --show-all
cd ..
