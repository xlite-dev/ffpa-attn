"""Entry point for ``python -m ffpa_attn.bench``.

Usage::

  CUDA_VISIBLE_DEVICES=0 python -m ffpa_attn.bench --help
  CUDA_VISIBLE_DEVICES=0 python -m ffpa_attn.bench --no-bwd --fwd-backend triton --tune fast
  CUDA_VISIBLE_DEVICES=0 python -m ffpa_attn.bench --fwd-backend cutedsl --bwd-backend cutedsl
"""

from .cli._bench import main

if __name__ == "__main__":
  main()
