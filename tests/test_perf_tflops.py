"""Unit tests for benchmark attention FLOPs and TFLOPS helpers."""

import pytest
from types import SimpleNamespace

from examples._attn_flops import (
  attention_bwd_flops,
  attention_fwd_flops,
  attention_valid_pairs,
  format_tflops_short,
  tflops_from_ms,
)
from examples.perf import _resolve_directional_cli_flags


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


def test_perf_legacy_tma_ws_flags_map_to_both_directions():
  args = SimpleNamespace(
    enable_tma=True,
    enable_ws=True,
    enable_fwd_tma=False,
    enable_bwd_tma=False,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=False,
    enable_bwd_split_launch=False,
  )

  _resolve_directional_cli_flags(args)

  assert args.enable_fwd_tma is True
  assert args.enable_bwd_tma is True
  assert args.enable_fwd_ws is True
  assert args.enable_bwd_ws is True


def test_perf_forward_tma_flag_does_not_enable_backward_tma():
  args = SimpleNamespace(
    enable_tma=False,
    enable_ws=False,
    enable_fwd_tma=True,
    enable_bwd_tma=False,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=False,
    enable_bwd_split_launch=False,
  )

  _resolve_directional_cli_flags(args)

  assert args.enable_fwd_tma is True
  assert args.enable_bwd_tma is False


def test_perf_persist_dkdv_requires_backward_tma():
  args = SimpleNamespace(
    enable_tma=False,
    enable_ws=False,
    enable_fwd_tma=False,
    enable_bwd_tma=False,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=True,
    enable_bwd_split_launch=False,
  )

  with pytest.raises(SystemExit, match="requires --enable-bwd-tma"):
    _resolve_directional_cli_flags(args)


def test_perf_persist_dkdv_allows_backward_tma():
  args = SimpleNamespace(
    enable_tma=False,
    enable_ws=False,
    enable_fwd_tma=False,
    enable_bwd_tma=True,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=True,
    enable_bwd_split_launch=False,
  )

  _resolve_directional_cli_flags(args)

  assert args.enable_persist_dkdv is True
  assert args.enable_bwd_tma is True


def test_perf_bwd_split_launch_allows_without_backward_tma():
  args = SimpleNamespace(
    enable_tma=False,
    enable_ws=False,
    enable_fwd_tma=False,
    enable_bwd_tma=False,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=False,
    enable_bwd_split_launch=True,
  )

  _resolve_directional_cli_flags(args)

  assert args.enable_bwd_split_launch is True
  assert args.enable_bwd_tma is False


def test_perf_bwd_split_launch_allows_backward_tma():
  args = SimpleNamespace(
    enable_tma=False,
    enable_ws=False,
    enable_fwd_tma=False,
    enable_bwd_tma=True,
    enable_fwd_ws=False,
    enable_bwd_ws=False,
    enable_persist_dkdv=False,
    enable_bwd_split_launch=True,
  )

  _resolve_directional_cli_flags(args)

  assert args.enable_bwd_split_launch is True
  assert args.enable_bwd_tma is True
