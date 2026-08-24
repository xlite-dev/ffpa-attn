"""Hadamard Q/K pre-rotation smoke tests (fp8 + fp4 shared plumbing).

The rotation is mathematically exact (orthogonal WHT), so on randn inputs
the fp8/fp4 output error must stay in the same order with the switch on;
a wiring/indexing bug breaks orthogonality and blows the error up.
"""

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("FFPA_CUDA_ALLOW_SMALL_D", "1")

from ffpa_attn import ffpa_attn_func
from ffpa_attn.functional import CUDABackend


def _sm120() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


pytestmark = pytest.mark.skipif(not _sm120(), reason="requires an sm_120 GPU")


def _randn_qkv(D=128, N=4096):
  g = torch.Generator(device="cuda").manual_seed(0)
  mk = lambda *s: torch.randn(
    *s, device="cuda", dtype=torch.bfloat16, generator=g
  )
  return mk(1, 8, N, D), mk(1, 2, N, D), mk(1, 2, N, D)


def _ref(q, k, v):
  scale = q.shape[-1]**-0.5
  g = q.shape[1] // k.shape[1]
  return F.scaled_dot_product_attention(
    q.float(),
    k.repeat_interleave(g, 1).float(),
    v.repeat_interleave(g, 1).float(),
    scale=scale
  )


@pytest.mark.parametrize(
  "enable_fp8,enable_fp4", [(True, False), (False, True)]
)
def test_randn_invariance(enable_fp8, enable_fp4):
  q, k, v = _randn_qkv()
  ref = _ref(q, k, v)
  rel = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
  rels = []
  for hadamard in (False, True):
    b = CUDABackend(
      backward=False,
      enable_fp8=enable_fp8,
      enable_fp4=enable_fp4,
      fp8_hadamard=hadamard and enable_fp8,
      fp4_hadamard=hadamard and enable_fp4
    )
    o = ffpa_attn_func(q, k, v, enable_gqa=True, forward_backend=b)
    rels.append(rel(o))
  assert rels[1] < max(rels[0] * 3.0, 0.05), \
    f"hadamard broke orthogonality: off={rels[0]:.4f} had={rels[1]:.4f}"


def test_pad_head_dim_invariance():
  # D=120 dispatches to kHeadDim=128; the fp8 launcher must zero-pad V to
  # keep the shared D_og row stride exact.
  q, k, v = _randn_qkv(D=120)
  ref = _ref(q, k, v)
  rel = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
  rels = []
  for hadamard in (False, True):
    b = CUDABackend(backward=False, enable_fp8=True, fp8_hadamard=hadamard)
    o = ffpa_attn_func(q, k, v, enable_gqa=True, forward_backend=b)
    rels.append(rel(o))
  assert rels[1] < max(rels[0] * 3.0, 0.05), \
    f"pad-path corruption: off={rels[0]:.4f} had={rels[1]:.4f}"


def test_hadamard_requires_matching_backend():
  with pytest.raises(AssertionError):
    CUDABackend(backward=False, enable_fp8=True, fp4_hadamard=True)
  with pytest.raises(AssertionError):
    CUDABackend(backward=False, enable_fp4=True, fp8_hadamard=True)
