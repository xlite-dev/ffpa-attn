"""Unit tests for benchmark attention FLOPs and TFLOPS helpers."""

from examples._attention_flops import (
  attention_bwd_flops,
  attention_fwd_flops,
  attention_valid_pairs,
  format_tflops_short,
  tflops_from_ms,
)


def test_attention_valid_pairs_non_causal_and_cross_attn():
  assert attention_valid_pairs(1024, 8192, False) == 1024 * 8192
  assert attention_valid_pairs(1, 8192, False) == 8192


def test_attention_valid_pairs_square_causal_matches_triangle():
  assert attention_valid_pairs(8, 8, True) == 8 * 9 // 2
  assert attention_valid_pairs(8192, 8192, True) == 8192 * 8193 // 2


def test_attention_valid_pairs_decode_causal_keeps_full_kv_tail():
  assert attention_valid_pairs(1, 8192, True) == 8192
  assert attention_valid_pairs(4, 8, True) == (8 - 4 + 1) + (8 - 4 + 2) + (8 - 4 + 3) + (8 - 4 + 4)


def test_attention_flops_use_query_head_count_for_gqa():
  flops = attention_fwd_flops(batch=1, num_heads_q=32, nq=8192, nkv=8192, headdim=512, causal=False)
  expected = 4 * 1 * 32 * 512 * 8192 * 8192
  assert flops == expected


def test_attention_backward_flops_is_two_point_five_x_forward():
  fwd = attention_fwd_flops(batch=1, num_heads_q=32, nq=1024, nkv=8192, headdim=512, causal=False)
  bwd = attention_bwd_flops(batch=1, num_heads_q=32, nq=1024, nkv=8192, headdim=512, causal=False)
  assert bwd == fwd * 5 // 2


def test_tflops_from_ms_and_compact_formatting():
  assert tflops_from_ms(90 * 10**12, 1000.0) == 90.0
  assert format_tflops_short(90.0) == "90T"
  assert format_tflops_short(9.25) == "9.2T"
  assert format_tflops_short(0.256) == "0.26T"
  assert format_tflops_short(None) == "-"
