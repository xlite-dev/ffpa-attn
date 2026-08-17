"""SM100 (Blackwell) CuTeDSL D512 tests — config, routing, dedicated path.

Two kinds of test live here and they are gated differently on purpose.

* The **configuration witnesses** are pure Python: they check that one
  executable configuration drives every tile, stage, TMEM and SMEM figure,
  and they run on any machine, so a laptop CI still catches a constant that
  drifts out of budget.
* The **device tests** carry ``@requires_sm100a``. A skipped run is *not*
  device evidence, so nothing here substitutes for running it on sm100a.

The routing tests assert on the *dispatch decision* rather than on a kernel
class name, so they keep proving the seam across refactors.
"""

import math
from contextlib import nullcontext
from typing import get_args

import cutlass
import cutlass.cute as cute
import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func, ffpa_attn_varlen_func
from ffpa_attn.cute import (
  _backward_impl_for_device,
  _cute_device_capability,
  _ffpa_attn_forward_cute,
  _forward_impl_for_device,
  _use_sm100_d512_specialized,
)
from ffpa_attn.cute import _ffpa_fwd_sm100
from ffpa_attn.cute._ffpa_fwd_sm100 import (
  SM100_D512_KERNEL_LIVE,
  _ffpa_attn_forward_sm100,
)
from ffpa_attn.cute._ffpa_bwd_sm100 import (
  SM100_D512_BWD_KERNEL_LIVE,
  _ffpa_attn_backward_sm100,
)
from ffpa_attn.cute._utils import (
  SM100_BWD_TILE_M,
  SM100_BWD_TILE_N_DKDV,
)
from ffpa_attn.cute._dk_d512_sm100 import FFPAAttnBwdDKSm100D512
from ffpa_attn.cute._dv_d512_sm100 import FFPAAttnBwdDVSm100D512
from ffpa_attn.cute._dq_d512_sm100 import FFPAAttnBwdDQSm100D512
from ffpa_attn.cute._ffpa_bwd_sm80 import _ffpa_attn_backward_sm80
from ffpa_attn.cute._ffpa_fwd_sm80 import _ffpa_attn_forward_sm80
from ffpa_attn.cute._fwd_d512_sm100 import (
  SM100_SMEM_CAPACITY_BYTES,
  SM100_TMEM_CAPACITY_COLUMNS,
  FFPAAttnFwdSm100D512,
  compile_key_fields,
)
from ffpa_attn.cute.utils import AuxData
from ffpa_attn.cute.utils.cute_dsl_utils import to_cute_tensor
from ffpa_attn.cute.utils.hd512_helpers import check_tmem_intervals
from ffpa_attn.cute.utils.named_barrier import NamedBarrierFwdSm100Hd512

SM100_D512 = 512


def _is_sm100a() -> bool:
  if not torch.cuda.is_available():
    return False
  if hasattr(torch.version, "hip") and torch.version.hip is not None:
    return False
  return torch.cuda.get_device_capability() == (10, 0)


requires_sm100a = pytest.mark.skipif(
  not _is_sm100a(),
  reason="SM100 D512 device tests require a compute capability 10.0 device",
)


def _make_kernel(**kwargs) -> FFPAAttnFwdSm100D512:
  params = dict(head_dim=SM100_D512, head_dim_v=SM100_D512)
  params.update(kwargs)
  return FFPAAttnFwdSm100D512(**params)


def _configured(**kwargs) -> FFPAAttnFwdSm100D512:
  """A kernel with ``_setup_attributes`` already run.

  The ring depths are decided there rather than in ``__init__`` (the S ring's
  depth is what the TMEM ledger check needs), and it takes no arguments, so a
  pure-Python witness can reach them without a device.
  """
  k = _make_kernel(**kwargs)
  k._setup_attributes()
  return k


def _tol(dtype: torch.dtype) -> dict[str, float]:
  if dtype == torch.float16:
    return {"atol": 2e-3, "rtol": 2e-3}
  return {"atol": 3e-3, "rtol": 3e-3}


def _cu(lengths) -> torch.Tensor:
  return torch.tensor(
    [0, *torch.tensor(lengths).cumsum(0).tolist()],
    device="cuda",
    dtype=torch.int32,
  )


def _qkv(b, s_q, s_k, h_q, h_kv, *, seed, dtype=torch.bfloat16):
  """q/dO at ``[b, s_q, h_q, D]`` and k/v at ``[b, s_k, h_kv, D]``.

  ``[b, s, h, d]`` is what the SM100 launchers take directly; the public
  entry's ``[B, H, N, D]`` is ``_qkv_bhnd`` and packed varlen is
  ``_packed_qkv``.
  """
  torch.manual_seed(seed)
  mk = lambda s, h: torch.randn(
    b, s, h, SM100_D512, device="cuda", dtype=torch.float32
  ).to(dtype)
  return mk(s_q, h_q), mk(s_k, h_kv), mk(s_k, h_kv), mk(s_q, h_q)


def _qkv_bhnd(b, h, n, d=SM100_D512, *, seed, dtype=torch.bfloat16, grad=False):
  """Three ``[B, H, N, D]`` tensors — the layout the public dense entry takes."""
  torch.manual_seed(seed)
  return [
    torch.randn(b, h, n, d, dtype=dtype, device="cuda", requires_grad=grad)
    for _ in range(3)
  ]


def _packed_qkv(lens, h, *, seed, grad=False):
  """Packed ``[total, h, D]`` q/k/v/dO and the prefix both sides share."""
  torch.manual_seed(seed)
  cu = _cu(lens)
  mk = lambda g: torch.randn(
    int(cu[-1]),
    h,
    SM100_D512,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=g
  )
  return cu, mk(grad), mk(grad), mk(grad), mk(False)


def _cosine(a, b) -> float:
  a, b = a.float().flatten(), b.float().flatten()
  if a.norm() == 0 and b.norm() == 0:
    return 1.0
  return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


# ---------------------------------------------------------------------------
# Configuration witnesses — no device required
#
# These assert on the kernel's *own* structures. An earlier revision kept a
# parallel model of the storage budget (``expected_smem_bytes`` and friends);
# it was removed with the kernel it modelled, because a second description of
# the same layout can agree with the test suite while disagreeing with the
# kernel, which is the failure this file exists to catch.
# ---------------------------------------------------------------------------


def test_config_tiles_stages_and_two_cta_pairing():
  """One executable configuration drives every tile, slice and stage count.

  ``cluster_shape_mn`` is fixed by ``__init__`` and is not a constructor
  parameter, so no supported call can ask for a different cluster.  What can
  still drift is the *relationship* between the per-CTA tile and the MMA tile
  the pair forms, so that is what is checked rather than the constants alone.
  The q/k/v ``stage`` counts index D/Dv slices rather than pipeline depth, so
  they equal the slice count; only ``qk_acc`` and ``mma_corr`` are temporal.
  """
  k = _configured(is_causal=True)
  assert k.cta_tiler == (64, 128, 512)
  assert k.qk_mma_tiler == k.pv_mma_tiler == (128, 128, 128)
  assert k.pv_block_tiler == (64, 128, 128)
  assert k.iterations_qk == k.iterations_pv == 4
  assert k.threads_per_cta == 384

  assert (k.q_stage, k.k_stage) == (k.iterations_qk, k.iterations_qk)
  assert k.v_stage == k.iterations_pv
  assert (k.qk_acc_stage, k.mma_corr_stage) == (2, 1)

  assert k.cta_group_size == 2
  assert k.cluster_shape_mn == (2, 1)
  assert k.qk_mma_tiler[0] == k.cta_group_size * k.cta_tiler[0]
  assert k.pv_mma_tiler[0] == k.cta_group_size * k.pv_block_tiler[0]
  # QK contracts all four 128-wide D slices; PV covers the full Dv in four.
  assert k.iterations_qk * k.qk_mma_tiler[2] == k.cta_tiler[2]
  assert k.iterations_pv * k.pv_mma_tiler[2] == k.cta_tiler[2]


def test_config_tmem_ledger_fits_the_allocation_and_its_check_bites():
  """One physical owner per TMEM column, inside what the kernel asks for.

  ``_setup_attributes`` already ran ``check_tmem_intervals`` on the real map,
  so disjointness is not restated here -- a second copy of that loop is the
  parallel model this file exists to catch.  What the check does *not* cover
  is the allocation: it bounds the ledger by the hardware capacity, not by
  ``tmem_alloc_cols``.  Reported per S-ring stage, so a ring made deeper
  without widening its region shows up as fewer intervals than there are
  stages -- the failure of this class that lowers cleanly, leaves SMEM
  unchanged, and races.
  """
  k = _configured()
  intervals = k.tmem_region_intervals(k.qk_acc_stage)
  assert len(intervals) == k.qk_acc_stage + k.iterations_pv
  assert min(start for start, _ in intervals.values()) >= 0
  assert max(stop for _, stop in intervals.values()) <= k.tmem_alloc_cols
  assert k.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

  # Without this the check above could be a tautology that never compares two
  # regions, or one that never compares a region against the addressable range.
  for region, message in (
    (intervals["S0"], "overlap"),
    ((SM100_TMEM_CAPACITY_COLUMNS, SM100_TMEM_CAPACITY_COLUMNS + 64),
     "outside"),
  ):
    with pytest.raises(AssertionError, match=message):
      check_tmem_intervals({**intervals, "O0": region})


def test_config_warp_roles_cover_every_warp():
  """Every warp of the CTA has exactly one named role.

  Warp 10 (V load) and the idle warp 11 form ``reg_trim_warp_ids``, the
  pre-dispatch 32-register trim set; that set is a register budget, not a
  role, so it is asserted separately from the role cover.
  """
  k = _make_kernel()
  warps_per_cta = k.threads_per_cta // 32
  roles = (
    *k.softmax_warp_ids,
    *k.correction_warp_ids,
    k.mma_warp_id,
    k.load_warp_id,
    k.v_load_warp_id,
    k.empty_warp_id,
  )
  assert sorted(roles) == list(range(warps_per_cta))
  assert k.reg_trim_warp_ids == (k.v_load_warp_id, k.empty_warp_id)


def test_config_named_barriers_match_the_participant_counts():
  """A barrier id is a contract with a specific participant count."""
  k = _make_kernel()
  assert k.tmem_alloc_barrier.barrier_id == NamedBarrierFwdSm100Hd512.TmemPtr
  assert k.tmem_alloc_barrier.num_threads == k.threads_per_cta
  assert k.softmax_pair_barrier.barrier_id == (
    NamedBarrierFwdSm100Hd512.SoftmaxPair
  )
  assert k.softmax_pair_barrier.num_threads == k.threads_per_softmax_group


def test_config_smem_ceiling_is_the_sm100_opt_in_limit():
  """227 KiB, the per-CTA opt-in dynamic shared-memory ceiling on SM100.

  The kernel's *actual* usage is not modelled here on purpose. A launch that
  asks for more than this ceiling fails at ``cudaFuncSetAttribute`` time, so
  every device test below is the budget witness; a second Python description
  of the storage layout could only drift away from the one that matters.
  """
  assert SM100_SMEM_CAPACITY_BYTES == 227 * 1024


# The dense-causal arms, read off the constructor's own ``Literal``. Restating
# the triple here would be the parallel model this file exists to catch: it
# could keep agreeing with itself while the kernel grew or lost an arm.
DENSE_CAUSAL_SCHED = get_args(
  FFPAAttnFwdSm100D512.__init__.__annotations__["dense_causal_sched"]
)
# An annotation that stopped being a ``Literal`` would empty the loop below
# rather than fail it, and one arm would make its distinct-key half vacuous.
assert "phase" in DENSE_CAUSAL_SCHED and len(DENSE_CAUSAL_SCHED) > 1, (
  f"dense_causal_sched arms degenerated to {DENSE_CAUSAL_SCHED!r}"
)


def test_config_compile_key_separates_every_static_axis():
  """Scheduler order, causal and varlen each fork the cache key."""
  base = compile_key_fields(_make_kernel(is_causal=True))
  for sched in DENSE_CAUSAL_SCHED:
    other = _make_kernel(is_causal=True, dense_causal_sched=sched)
    assert (compile_key_fields(other) == base) is (sched == "phase")
  assert len({
    compile_key_fields(_make_kernel(is_causal=c, is_varlen_q=v))
    for c in (False, True)
    for v in (False, True)
  }) == 4


@pytest.mark.parametrize(
  "kwargs,message",
  [
    (dict(head_dim=256, head_dim_v=256), r"\(512, 512\)"),
    (dict(is_local=True), "local attention"),
    (dict(pack_gqa=True), "pack_gqa"),
    (dict(is_split_kv=True), "SplitKV"),
    (dict(m_block_size=64), "tile_m=128"),
    (dict(dense_causal_sched="lpt"), "dense_causal_sched"),
  ],
)
def test_config_rejects_out_of_contract_construction(kwargs, message):
  with pytest.raises(AssertionError, match=message):
    _make_kernel(**kwargs)


# ---------------------------------------------------------------------------
# Routing and delegation — pure, so they can be checked for architectures we do
# not have, and for features the wrapper would reject before a device saw them.
# ---------------------------------------------------------------------------


def test_both_kernels_are_live():
  """Nothing below is conditional on the migration flags, so they are pinned.

  ``_sm100_d512_fallback_reason`` short-circuits to a single reason while the
  forward is not live, and every backward test presumes the dedicated pair
  exists.  Flipping either flag has to fail here rather than silently drain
  the meaning out of the rest of the file.
  """
  assert SM100_D512_KERNEL_LIVE and SM100_D512_BWD_KERNEL_LIVE


@pytest.mark.parametrize(
  "major,minor,head_dim,head_dim_v,expected",
  [
    (10, 0, 512, 512, True),
    # Exact specialisation, not a range: other Blackwell minors fall back
    # until they have their own device evidence.
    (10, 3, 512, 512, False),
    (11, 0, 512, 512, False),
    (12, 0, 512, 512, False),
    # Other head dims stay on the SM80 Split-D fallback.
    (10, 0, 256, 256, False),
    (10, 0, 576, 576, False),
    (10, 0, 512, 256, False),
    (10, 0, 256, 512, False),
    # Hopper keeps its own specialisation.
    (9, 0, 512, 512, False),
    (8, 0, 512, 512, False),
  ],
)
def test_sm100_routing_predicate(major, minor, head_dim, head_dim_v, expected):
  assert _use_sm100_d512_specialized(
    major, minor, head_dim, head_dim_v
  ) is expected


#: Every frozen-contract exclusion the SM100 wrapper names, plus the two calls
#: that are in contract. A shared boolean would make ``softcap`` and
#: ``score_mod`` indistinguishable in a log; the migration standard requires
#: the delegation to be auditable, so the reason string *is* the contract.
_FALLBACK_REASONS = [
  pytest.param({}, None, id="in-contract"),
  # ``requires_grad`` used to delegate. It no longer does: the dedicated
  # backward exists, and one predicate selects the pair.
  pytest.param({"requires_grad": True}, None, id="requires_grad"),
  # Packed varlen *is* in contract, but only with both prefixes: one alone
  # would leave the roles deriving trip counts from different lengths, which
  # is not a half-supported mode, it is a hang.
  pytest.param({"cu_seqlens_q": object()}, "varlen-one-sided", id="cu_q_only"),
  pytest.param({"cu_seqlens_k": object()}, "varlen-one-sided", id="cu_k_only"),
  pytest.param({"local": True}, "local-window", id="local"),
  pytest.param({"softcap": 30.0}, "softcap", id="softcap"),
  pytest.param({"score_mod": lambda *a: a}, "score_mod", id="score_mod"),
  pytest.param({"mask_mod": lambda *a: a}, "mask_mod", id="mask_mod"),
  pytest.param({"aux_tensors": [object()]}, "aux_tensors", id="aux_tensors"),
]


@pytest.mark.parametrize("kwargs,reason", _FALLBACK_REASONS)
def test_every_out_of_contract_feature_has_its_own_named_reason(kwargs, reason):
  """Each delegation is attributable to exactly one rule, or to none.

  Runs without a GPU: the predicate is pure, so ``meta`` tensors reach it.
  """
  q = torch.empty(1, 1, 1, SM100_D512, dtype=torch.bfloat16, device="meta")
  base = dict(head_dim=SM100_D512, head_dim_v=SM100_D512, requires_grad=False)
  base.update(kwargs)
  assert _ffpa_fwd_sm100._sm100_d512_fallback_reason(q, q, q, **base) == reason


# ---------------------------------------------------------------------------
# Dispatch seam — crosses the architecture-dispatch interface on a real device
# ---------------------------------------------------------------------------

#: ``(head_dim, head_dim_v, dedicated)``. The SM100 claim is exact, so every
#: other head dim the public interface accepts must reach SM80 Split-D.
_DISPATCH_DOMAIN = [
  (SM100_D512, SM100_D512, True),
  (320, 320, False),
  (576, 576, False),
  (1024, 1024, False),
  (512, 256, False),
]


@requires_sm100a
def test_device_capability_reports_sm100a():
  """The dispatch helper must read the minor, not just the major."""
  dev = torch.device("cuda", torch.cuda.current_device())
  assert _cute_device_capability(dev) == (10, 0)


@requires_sm100a
@pytest.mark.parametrize("head_dim,head_dim_v,dedicated", _DISPATCH_DOMAIN)
def test_dispatch_pairs_forward_and_backward_over_the_declared_domain(
  head_dim, head_dim_v, dedicated
):
  """Forward and backward are selected by one predicate, so they cannot split.

  A D512 sm100a call that took the dedicated forward and the SM80 Split-D
  backward would disagree on causal alignment for asymmetric sequence lengths
  and produce plausible, wrong gradients.  Asserted on the dispatch decision
  rather than on a kernel class name, and in both directions, so that neither
  a missed specialisation nor a widened one can pass.
  """
  dev = torch.device("cuda", torch.cuda.current_device())
  fwd = _forward_impl_for_device(dev, head_dim, head_dim_v)
  bwd = _backward_impl_for_device(dev, head_dim, head_dim_v)
  assert (fwd is _ffpa_attn_forward_sm100) is dedicated
  assert (bwd is _ffpa_attn_backward_sm100) is dedicated
  assert (fwd is _ffpa_attn_forward_sm80) is not dedicated
  assert (bwd is _ffpa_attn_backward_sm80) is not dedicated


@requires_sm100a
def test_sm80_fallback_is_not_reached_for_a_d512_forward(monkeypatch):
  """The dedicated path must not silently ride on the fallback."""

  def _fail(*args, **kwargs):
    raise AssertionError(
      "SM80 Split-D fallback was called for a D512 sm100a forward"
    )

  monkeypatch.setattr(_ffpa_fwd_sm100, "_ffpa_attn_forward_sm80", _fail)
  q, k, v = _qkv_bhnd(1, 2, 128, seed=0)
  out, _ = _ffpa_attn_forward_cute(
    q, k, v, 1.0 / math.sqrt(SM100_D512), False, return_lse=True
  )
  assert torch.isfinite(out).all()


@requires_sm100a
def test_sm80_fallback_is_not_reached_for_a_d512_training_call(monkeypatch):
  """The same for the pair: a training call must not reach the SM80 backward."""
  import ffpa_attn.cute as _cute

  def _fail(*args, **kwargs):
    raise AssertionError(
      "SM80 Split-D backward was called for a D512 sm100a training pass"
    )

  monkeypatch.setattr(_cute, "_ffpa_attn_backward_sm80", _fail)
  q, k, v = _qkv_bhnd(1, 2, 128, seed=0, grad=True)
  ffpa_attn_func(q, k, v, is_causal=True, backend="cutedsl").sum().backward()
  assert all(torch.isfinite(t.grad).all() for t in (q, k, v))


# ---------------------------------------------------------------------------
# The public D512 result must stay correct across the seam, and everything the
# SM100 path does not claim must keep working on the declared fallback.
# ---------------------------------------------------------------------------


@requires_sm100a
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [SM100_D512, 576], ids=["d512", "d576-fallback"])
def test_public_entry_matches_sdpa_across_the_seam(causal, d):
  """One public call, whichever side of the delegation currently owns it."""
  q, k, v = _qkv_bhnd(1, 2, 256, d, seed=0)
  scale = 1.0 / math.sqrt(d)
  out, lse = _ffpa_attn_forward_cute(q, k, v, scale, causal, return_lse=True)
  ref = F.scaled_dot_product_attention(
    q, k, v, dropout_p=0.0, is_causal=causal, scale=scale
  )
  assert lse.shape == (1, 2, 256)
  torch.testing.assert_close(out, ref, **_tol(torch.bfloat16))


@requires_sm100a
@pytest.mark.parametrize("seqlen_q, seqlen_k", [(1, 1), (1, 64), (64, 1)])
def test_minimum_legal_shapes(seqlen_q, seqlen_k):
  """The smallest shapes the contract admits, including a padded x-tail.

  ``seqlen_q < 64`` leaves the whole 64-row CTA tile partially out of range
  and the cluster-pair padding adds a second, entirely out-of-range tile.
  Both are masked per row rather than skipped, so the store guard -- not a
  pre-fill -- is what has to hold here.
  """
  torch.manual_seed(3)
  d = SM100_D512
  dtype = torch.bfloat16
  q = torch.randn(1, 1, seqlen_q, d, dtype=dtype, device="cuda")
  k, v = [
    torch.randn(1, 1, seqlen_k, d, dtype=dtype, device="cuda")
    for _ in range(2)
  ]
  scale = 1.0 / math.sqrt(d)

  out, lse = _ffpa_attn_forward_cute(q, k, v, scale, False, return_lse=True)
  ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
  assert lse.shape == (1, 1, seqlen_q)
  torch.testing.assert_close(out, ref, **_tol(dtype))


# ---------------------------------------------------------------------------
# Compile and cache behaviour of the wired path.
# ---------------------------------------------------------------------------


@requires_sm100a
def test_first_compile_then_cache_hit_on_the_same_key():
  """A second call with the same static shape must not compile again.

  The key comes from ``compile_key_fields()``, so this also pins that a
  purely dynamic difference (sequence length) does not fork the cache while
  a static one (causal) does.
  """
  cache = _ffpa_attn_forward_sm100.compile_cache.cache

  def _call(n, causal):
    q, k, v, _do = _qkv(1, n, n, 1, 1, seed=0)
    _ffpa_attn_forward_sm100(
      q, k, v, softmax_scale=1.0 / math.sqrt(SM100_D512), causal=causal
    )

  _call(128, False)
  after_first = len(cache)
  _call(128, False)
  assert len(cache) == after_first, "identical static config recompiled"
  _call(256, False)
  assert len(cache) == after_first, "sequence length must not fork the key"
  _call(128, True)
  assert len(cache) == after_first + 1, "causal is static and must fork"


@requires_sm100a
@pytest.mark.parametrize("causal", [False, True])
def test_lse_optional_does_not_change_o(causal):
  """Dropping LSE must change what is written, not how O is computed.

  ``mLSE is None`` is a separate constexpr specialisation: it skips the LSE
  store *and* one of the three paired-softmax barrier arrivals.  The rest of
  the mainloop must be untouched, so O has to come out bit-for-bit equal --
  an approximate match here would hide a reordered reduction.
  """
  q, k, v, _do = _qkv(1, 384, 384, 2, 2, seed=5)
  scale = 1.0 / math.sqrt(SM100_D512)
  with_lse, lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=causal, return_lse=True
  )
  without_lse, none_lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=causal, return_lse=False
  )
  assert lse is not None and none_lse is None
  assert torch.equal(with_lse, without_lse)


@requires_sm100a
def test_fake_mode_reports_shapes_without_reaching_the_sm100_launcher():
  """Fake mode must produce metadata only, at D512 like anywhere else.

  The op's ``register_fake`` meta implementation is what answers here, so the
  assertion that matters is not just the shape but that the SM100 compile
  cache did not grow -- i.e. no descriptor construction or launch happened
  behind a traced call.
  """
  from torch._subclasses.fake_tensor import FakeTensorMode

  b, h, n, d = 2, 4, 256, SM100_D512
  before = len(_ffpa_attn_forward_sm100.compile_cache.cache)
  with FakeTensorMode():
    q, k, v = [
      torch.empty(b, h, n, d, dtype=torch.bfloat16, device="cuda")
      for _ in range(3)
    ]
    out, lse = _ffpa_attn_forward_cute(
      q, k, v, 1.0 / math.sqrt(d), False, return_lse=True
    )
  assert out.shape == (b, h, n, d) and out.dtype == torch.bfloat16
  assert lse.shape == (b, h, n) and lse.dtype == torch.float32
  assert len(_ffpa_attn_forward_sm100.compile_cache.cache) == before


@requires_sm100a
@pytest.mark.parametrize("causal", [False, True])
def test_torch_compile_matches_eager_through_the_sm100_seam(causal):
  """``torch.compile`` must route through the same custom op and agree."""
  q, k, v = _qkv_bhnd(1, 2, 256, seed=4)
  scale = 1.0 / math.sqrt(SM100_D512)
  eager, _ = _ffpa_attn_forward_cute(q, k, v, scale, causal, return_lse=True)
  compiled = torch.compile(_ffpa_attn_forward_cute, fullgraph=False)
  got, _ = compiled(q, k, v, scale, causal, return_lse=True)
  torch.testing.assert_close(got, eager, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Materialisation — builds every descriptor and compiles the kernel for the
# exact frozen architecture.
# ---------------------------------------------------------------------------

# The kernel keeps the donor's full forward signature and rejects every
# feature this package does not support.  Those arguments are spelled out
# positionally, exactly as the wrapper spells them, so a donor signature
# change breaks the tests here rather than silently binding a tensor to the
# wrong parameter.
_FORWARD_UNSUPPORTED_ARGS = (
  None,  # mSeqUsedQ
  None,  # mSeqUsedK
  None,  # mPageTable
  None,  # window_size_left
  None,  # window_size_right
  None,  # learnable_sink
  None,  # descale_tensors
  None,  # blocksparse_tensors
  AuxData(),
)


def _materialise(
  kernel, *, packed=False, cu_q=False, cu_k=False, kv_dtype=None
):
  """Compile ``kernel`` against tensors built by hand, not by the wrapper.

  The kernel's *own* contract checks are the subject, and the wrapper would
  reject the bad combinations before they ever got here.
  """
  b, s, h, d = 2, 256, 2, SM100_D512
  dtype = torch.bfloat16
  shape = (b * s, h, d) if packed else (b, s, h, d)
  mk = lambda t: torch.randn(shape, device="cuda", dtype=t)
  tensors = (
    mk(dtype), mk(kv_dtype or dtype), mk(kv_dtype or dtype),
    torch.empty(shape, device="cuda", dtype=dtype)
  )
  cu = torch.tensor([0, s, b * s], device="cuda", dtype=torch.int32)
  as_cu = lambda: to_cute_tensor(cu, assumed_align=4, leading_dim=0)
  stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
  return cute.compile(
    kernel, *[to_cute_tensor(t) for t in tensors], None, 1.0 / math.sqrt(d),
    as_cu() if cu_q else None,
    as_cu() if cu_k else None, *_FORWARD_UNSUPPORTED_ARGS, stream
  )


@requires_sm100a
@pytest.mark.parametrize("causal", [False, True])
def test_kernel_compiles_for_sm100a(causal):
  """Layouts, TMA descriptors, pipelines and the kernel body all build."""
  assert _materialise(_make_kernel(is_causal=causal)) is not None


@requires_sm100a
def test_materialisation_rejects_dtype_mismatch():
  with pytest.raises(TypeError, match="Type mismatch"):
    _materialise(_make_kernel(), kv_dtype=torch.float16)


@requires_sm100a
@pytest.mark.parametrize(
  "kwargs, call, message",
  [
    # Packed rank-3 Q/K/V/O with both prefixes: the supported varlen mode.
    pytest.param(
      dict(is_varlen_q=True, is_causal=True),
      dict(packed=True, cu_q=True, cu_k=True),
      None,
      id="packed-varlen"
    ),
    # One prefix alone: Q's rebases the query origin and the LSE row, K's
    # rebases the key origin and the bottom-right offset. Half of that is
    # not a supported mode, it is a hang.
    pytest.param(
      dict(is_varlen_q=True),
      dict(packed=True, cu_q=True),
      "requires both",
      id="cu_q_only"
    ),
    pytest.param(
      dict(is_varlen_q=True),
      dict(packed=True, cu_k=True),
      "requires both",
      id="cu_k_only"
    ),
    # The prefixes pick the scheduler and the compile key at construction
    # time, so a call that disagrees would launch the wrong specialisation.
    pytest.param(
      dict(is_varlen_q=False),
      dict(packed=True, cu_q=True, cu_k=True),
      "must match the is_varlen_q",
      id="dense-kernel-varlen-call"
    ),
    pytest.param(
      dict(is_varlen_q=True),
      dict(packed=True),
      "must match",
      id="varlen-kernel-dense-call"
    ),
    # Packed data is rank 3; a dense rank-4 tensor with prefixes is a caller
    # error, not a layout to guess at.
    pytest.param(
      dict(is_varlen_q=True),
      dict(cu_q=True, cu_k=True),
      "expects q rank 3",
      id="rank-4-with-prefixes"
    ),
  ],
)
def test_materialisation_varlen_contract_is_enforced(kwargs, call, message):
  """The supported varlen pairing lowers; every mis-pairing fails loudly."""
  raises = pytest.raises((AssertionError, RuntimeError), match=message)
  with nullcontext() if message is None else raises:
    _materialise(_make_kernel(**kwargs), **call)


# ---------------------------------------------------------------------------
# fp32 oracles — one per plane, sharing the one thing the two planes agree on
# ---------------------------------------------------------------------------

#: Rows per score block in the forward oracle. At the widest swept shape
#: (s = 4096, h = 8) a chunk is 128 MiB of fp32 scores; the full block would be
#: four times that for no gain.
_REF_ROW_CHUNK = 1024


def _bottom_right_mask(s_q, s_k, device):
  """True where a query row may attend. SDPA's ``is_causal`` is top-left.

  Bottom-right alignment is a different attention pattern rather than a
  tolerance question when ``s_q != s_k``, so it is spelled out everywhere it
  is needed instead of being asked of a flag.
  """
  i = torch.arange(s_q, device=device)[:, None]
  j = torch.arange(s_k, device=device)[None, :]
  return j <= i + (s_k - s_q)


def _ref_fwd(q, k, v, causal, scale):
  """fp32 O ``[b, s, h, d]`` and LSE ``[b, h, s]`` for ``[b, s, h, d]`` input.

  Rows with no legal key are ``O = 0`` / ``LSE = -inf`` by construction rather
  than by asking SDPA: softmax over an all ``-inf`` row is backend-defined,
  and the declared contract is not.
  """
  b, s_q, h_q, _ = q.shape
  s_k, h_kv = k.shape[1], k.shape[2]
  qf, kf, vf = [t.float().transpose(1, 2) for t in (q, k, v)]
  mask = _bottom_right_mask(s_q, s_k, q.device) if causal else None
  out = F.scaled_dot_product_attention(
    qf, kf, vf, attn_mask=mask, scale=scale, enable_gqa=(h_q != h_kv)
  )
  # LSE against the same reduction, taken directly rather than through SDPA.
  if h_q != h_kv:
    kf = kf.repeat_interleave(h_q // h_kv, dim=1)
  lse = torch.empty(b, h_q, s_q, device=q.device, dtype=torch.float32)
  for lo in range(0, s_q, _REF_ROW_CHUNK):
    hi = min(lo + _REF_ROW_CHUNK, s_q)
    scores = (qf[:, :, lo:hi] @ kf.transpose(-1, -2)) * scale
    if mask is not None:
      scores = scores.masked_fill(~mask[lo:hi], float("-inf"))
    lse[:, :, lo:hi] = torch.logsumexp(scores, dim=-1)
  empty = lse == -float("inf")
  return out.masked_fill(empty.unsqueeze(-1), 0.0).transpose(1, 2), lse


def _ref_bwd(q, k, v, do, causal, scale):
  """fp32 autograd dQ/dK/dV, all at ``[b, s, h, d]``."""
  h_q, h_kv = q.shape[2], k.shape[2]
  qf, kf, vf = [
    t.float().transpose(1, 2).detach().requires_grad_() for t in (q, k, v)
  ]
  mask = (
    _bottom_right_mask(q.shape[1], k.shape[1], q.device) if causal else None
  )
  F.scaled_dot_product_attention(
    qf, kf, vf, attn_mask=mask, scale=scale, enable_gqa=(h_q != h_kv)
  ).backward(do.float().transpose(1, 2))
  return tuple(t.grad.transpose(1, 2) for t in (qf, kf, vf))


def _assert_grads_match(lanes, *, tag=(), rel_tol=None):
  """Every named lane finite, non-degenerate, and aligned with the oracle.

  ``max|got| > 0`` is not implied by the cosine and is asserted separately:
  the dK grid partitions the head axis and the wrapper zero-fills, so a
  dropped head or sequence comes back exactly zero -- an unambiguous
  signature that an aggregate cosine only blurs.  ``rel_tol`` adds the
  magnitude bound cosine cannot give, since a uniformly rescaled lane still
  scores 1.0.
  """
  for name, got, want in lanes:
    where = (*tag, name)
    assert torch.isfinite(got).all(), where
    assert got.abs().max() > 0, (*where, "no gradient at all")
    assert _cosine(got, want) > 0.999, where
    if rel_tol is not None:
      rel = (got.float() -
             want.float()).abs().max() / (want.float().abs().max() + 1e-30)
      assert rel < rel_tol, (*where, float(rel))


# ---------------------------------------------------------------------------
# Dense shape sweep
#
# The paired donor-vs-port latency sweep runs a grid rather than a single
# shape. These are its correctness counterpart: the same shape *classes*,
# checked against an fp32 oracle here instead of against the donor, so they
# keep their meaning once the donor repository is gone.
#
# What makes that grid a grid, rather than one shape stretched, is the dK
# launcher. ``choose_lpt_grouping`` runs at *construction* time -- by the time
# the kernel sees them every extent is a dynamic ``Int32`` -- so a causal dK is
# a different compiled kernel at different ``(seq_k, heads_kv, batch)``. The
# grouping partitions the head axis, which is the part that matters here: a
# wrong one does not degrade an answer, it leaves whole KV heads undispatched.
# ---------------------------------------------------------------------------

#: ``(seq_k, heads_kv, batch) -> (head_group, num_groups)`` at ``block_k = 64``.
#: Pinned rather than recomputed. This table *is* the sweep's coverage claim,
#: and a retune of the chooser that collapsed four dispatch configurations into
#: one would otherwise silently shrink what the long shapes below exercise,
#: without failing anything.
_LPT_GROUPINGS = [
  ((1024, 8, 1), (8, 1)),
  ((2048, 8, 1), (4, 2)),
  ((4096, 8, 1), (2, 4)),
  ((2048, 1, 1), (0, 0)),  # heads_kv < 2: nothing to partition
  ((8192, 8, 2), (0, 0)),  # past the wave gate: stock grid
]

#: ``(b, s_q, s_k, h_q, h_kv, causal)``. The short rows walk the mask and
#: head-mapping arms; the long rows land the causal dK launcher on each
#: grouping above, which is why they are lengths rather than round numbers.
_SHAPES = [
  (1, 256, 256, 2, 2, False),
  (1, 256, 256, 2, 2, True),
  (2, 128, 128, 4, 4, True),
  (1, 384, 384, 4, 2, True),  # GQA
  (1, 200, 200, 2, 2, True),  # residue tile
  (1, 128, 320, 2, 2, True),  # s_q < s_k, bottom-right
  (1, 320, 128, 2, 2, True),  # s_q > s_k: leading rows have no legal key
  (1, 256, 256, 8, 1, True),  # MQA
  (1, 1024, 1024, 8, 8, True),  # LPT (8, 1)
  (1, 2048, 2048, 8, 8, True),  # LPT (4, 2)
  (1, 4096, 4096, 8, 8, True),  # LPT (2, 4), the widest head partition
  (1, 2048, 2048, 8, 1, True),  # heads_kv < 2: stock grid
  (1, 2048, 2048, 8, 8, False),
  (2, 1024, 1024, 4, 2, True),
]

_SHAPE_IDS = [
  f"b{b}_{s_q}x{s_k}_h{h_q}x{h_kv}_{'causal' if c else 'full'}"
  for b, s_q, s_k, h_q, h_kv, c in _SHAPES
]

#: The PV reduction's error floor, as a multiple of ``max|V|``. bf16 keeps 8
#: significand bits, so the operands of ``O = sum_j p_j v_j`` carry a quantum of
#: ``max|V| * 2^-9``; the weights sum to 1, so the reduction does not amplify
#: it, and two quanta is the bound.
#:
#: The scale must be ``max|V|`` and not ``max|O|``. An *absolute* tolerance is
#: wrong at any value, but so is a fraction of the peak output: on a long
#: non-causal row O is a near-uniform average of ``s_k`` random value vectors,
#: so ``max|O|`` collapses as ~1/sqrt(s_k) -- ~0.3 at s=2048, against ~3.5 on
#: the causal rows, whose leading rows see one or two keys and therefore have
#: ``max|O| ~ max|V|``. The error floor does not collapse with it, so a bound
#: written as ``max|O| * 2^-8`` straddles 1.0 on exactly those rows: measured
#: at s=2048 non-causal it lands anywhere in 0.72 .. 1.22 with nothing varying
#: but the seed. That is why it is not the instrument here.
#:
#: Measured over every shape below at four seeds: 0.036 to 0.495 of this bound.
#: It is loosest where averaging cancels operand error (the non-causal rows,
#: ~0.05) and tightest where the reduction is short (~0.50); the cosine floor,
#: which is scale-free, is what stays sharp on the loose rows.
_BF16_OPERAND_ULP = 2**-8

#: Worst measured LSE deviation over the same shapes is 6.0e-5 (relative to a
#: peak |LSE| of ~9). LSE is an fp32 output, so it is held far tighter than O.
_LSE_ABS_TOL = 2e-4


def test_dk_lpt_grouping_spans_four_dispatch_configurations():
  """Pure: the sweep's coverage claim, checked rather than asserted in prose."""
  seen = set()
  for (seq_k, heads_kv, batch), expected in _LPT_GROUPINGS:
    got = FFPAAttnBwdDKSm100D512.choose_lpt_grouping(
      seq_k, heads_kv, batch, SM100_BWD_TILE_N_DKDV, SM100_D512
    )
    assert got == expected, (seq_k, heads_kv, batch, got, expected)
    seen.add(got)
  assert len(seen) >= 4, seen

  # A grouping that does not divide heads_kv would leave the last
  # ``heads_kv % head_group`` KV heads undispatched. The chooser is the only
  # supplier of a valid grouping, so the property is asserted at the source.
  for (_seq_k, heads_kv, _b), (group, ngroups) in _LPT_GROUPINGS:
    if group:
      assert heads_kv % group == 0
      assert group * ngroups == heads_kv


@requires_sm100a
@pytest.mark.parametrize("b,s_q,s_k,h_q,h_kv,causal", _SHAPES, ids=_SHAPE_IDS)
def test_fwd_matches_fp32_oracle(b, s_q, s_k, h_q, h_kv, causal):
  """Forward O and LSE against an fp32 oracle.

  The forward has no construction-time shape decision of its own, so this is
  the control for the backward test below: a shape that fails in both
  directions is a bad shape, one that fails only in dK is a bad grouping.
  """
  q, k, v, _do = _qkv(b, s_q, s_k, h_q, h_kv, seed=s_q * 31 + s_k)
  scale = 1.0 / math.sqrt(SM100_D512)
  out, lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=causal, return_lse=True
  )
  ref_o, ref_lse = _ref_fwd(q, k, v, causal, scale)
  assert lse.shape == (b, h_q, s_q)

  # Rows with no legal key: the contract names the values, and a large
  # negative LSE or a merely small O would still be a defect.
  finite = torch.isfinite(ref_lse)
  assert torch.all(lse[~finite] == -float("inf"))
  assert torch.all(
    out[~finite.transpose(1, 2).unsqueeze(-1).expand_as(out)] == 0
  )
  assert (lse[finite] - ref_lse[finite]).abs().max() < _LSE_ABS_TOL

  bound = v.float().abs().max() * _BF16_OPERAND_ULP
  err = (out.float() - ref_o).abs().max()
  assert err <= bound, (
    f"max|dO| = {err:.3e} exceeds two bf16 quanta of max|V| ({bound:.3e})"
  )
  assert _cosine(out, ref_o) > 0.99999


@requires_sm100a
@pytest.mark.parametrize("b,s_q,s_k,h_q,h_kv,causal", _SHAPES, ids=_SHAPE_IDS)
def test_bwd_matches_fp32_oracle(b, s_q, s_k, h_q, h_kv, causal):
  """dQ/dK/dV against an fp32 oracle, in aggregate and then per head.

  Per head is not redundant. The blocked-LPT grid partitions the head axis,
  so the failure mode it can produce is a *missing* head rather than a
  slightly wrong one, and the wrapper allocates dK with ``zeros_like``, so a
  dropped head comes back exactly zero. An aggregate cosine does fail on that
  -- four blank heads of eight score 0.71 -- but it fails the same way it
  would for any numerical drift, whereas ``max|dK| == 0`` for a named head
  points at the grouping directly.

  Measured, not assumed: forcing ``choose_lpt_grouping`` to report ``(2, 2)``
  where it should report ``(2, 4)`` at ``s = 4096, h_kv = 8`` leaves heads 4-7
  with ``max|dK| = 0.000`` and no error raised anywhere.
  """
  q, k, v, do = _qkv(b, s_q, s_k, h_q, h_kv, seed=s_q * 31 + s_k)
  scale = 1.0 / math.sqrt(SM100_D512)
  out, lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=causal, return_lse=True
  )
  dq, dk, dv = _ffpa_attn_backward_sm100(
    q, k, v, out, do, lse, softmax_scale=scale, causal=causal
  )
  rq, rk, rv = _ref_bwd(q, k, v, do, causal, scale)

  # Rows with at most one legal key have an analytically zero dS row, so the
  # fp32 reference there is reduction residue and a *correct* kernel scores
  # cosine 0 against it. Assert the kernel's zero directly and exclude those
  # rows, rather than widening a tolerance until both pass. The ``LSE = -inf``
  # the forward wrote for them is what the preprocess hands to all three
  # gradient kernels, and a path that does not special-case it produces
  # ``exp2(+inf)``, so this is also the witness that the -inf survived.
  keep = torch.ones(s_q, dtype=torch.bool, device="cuda")
  if causal:
    keep = (torch.arange(s_q, device="cuda") + (s_k - s_q) + 1) > 1
    if (~keep).any():
      assert dq[:, ~keep].abs().max() == 0

  lanes = (("dQ", dq[:, keep], rq[:, keep]), ("dK", dk, rk), ("dV", dv, rv))
  _assert_grads_match(lanes, rel_tol=3e-2)
  for name, got, want in lanes:
    for head in range(got.shape[2]):
      _assert_grads_match(((name, got[:, :, head], want[:, :, head]), ),
                          tag=(f"head{head}", ))


@requires_sm100a
def test_bwd_dk_single_visible_key_is_analytically_zero():
  """``s_k == 1``: every dS row is analytically zero, and dK/dQ say so exactly.

  ``dP == dpsum`` row-wise when only one key is visible, so ``dS == 0`` in
  exact arithmetic. The kernels write the analytic zero rather than the ~1-ulp
  reduction residue; this is the output-level witness of that frozen-contract
  clause. dV stays dense and nonzero -- ``P == 1`` for the single key, so
  ``dV == sum_s dO`` -- which pins that the zeros come from the analytic rule
  and not from a skipped launch.
  """
  q, k, v, do = _qkv(1, 128, 1, 2, 2, seed=17)
  scale = 1.0 / math.sqrt(SM100_D512)
  out, lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=False, return_lse=True
  )
  dq, dk, dv = _ffpa_attn_backward_sm100(
    q, k, v, out, do, lse, softmax_scale=scale, causal=False
  )
  assert dk.abs().max() == 0
  assert dq.abs().max() == 0
  assert _cosine(dv, do.float().sum(dim=1, keepdim=True)) > 0.999


# ---------------------------------------------------------------------------
# Packed varlen — device matrix
#
# Every case here goes through the *public* varlen entry, and each asserts the
# dispatch decision first: a silent delegation to SM80 Split-D would otherwise
# produce correct numbers and read as evidence for the dedicated path.
# ---------------------------------------------------------------------------


def _varlen_oracle(q, k, v, cu_q, cu_k, causal, num_head):
  """Per-sequence fp32 reference, one ``_ref_fwd`` call per sequence.

  Sequences with no query or no key keep the initialised ``O = 0`` /
  ``LSE = -inf``; within a sequence, ``_ref_fwd`` applies the same rule to
  rows the bottom-right mask leaves empty.
  """
  d = q.shape[-1]
  ref_o = torch.zeros(q.shape[0], num_head, d, device="cuda")
  ref_lse = torch.full((num_head, q.shape[0]),
                       -float("inf"),
                       device="cuda",
                       dtype=torch.float32)
  for i in range(cu_q.numel() - 1):
    qs, qe = int(cu_q[i]), int(cu_q[i + 1])
    ks, ke = int(cu_k[i]), int(cu_k[i + 1])
    if qe == qs or ke == ks:
      continue
    o, lse = _ref_fwd(
      q[qs:qe].unsqueeze(0),
      k[ks:ke].unsqueeze(0),
      v[ks:ke].unsqueeze(0),
      causal,
      1.0 / math.sqrt(d),
    )
    ref_o[qs:qe], ref_lse[:, qs:qe] = o[0], lse[0]
  return ref_o, ref_lse


@requires_sm100a
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
  "lens_q, lens_k, num_head, num_head_kv",
  [
    # Uneven lengths, both Sq < Sk and Sq > Sk, since bottom-right alignment
    # is re-derived per sequence and the two directions are different code.
    # The Sq > Sk row also covers rows with no legal key at all.
    pytest.param([130, 512, 7, 1024], None, 4, 4, id="uneven_equal"),
    pytest.param([64, 200, 1], [512, 333, 1024], 4, 4, id="cross_sq_lt_sk"),
    pytest.param([512, 700, 300], [64, 129, 1], 4, 4, id="cross_sq_gt_sk"),
    pytest.param([128, 256], [0, 256], 2, 2, id="zero_len_k"),
    pytest.param([0, 256, 130], [64, 256, 130], 2, 2, id="zero_len_q"),
    # M residue around the 128-row supertile, in both directions.
    pytest.param([1, 127, 128, 129, 255], None, 2, 2, id="residue_m"),
    pytest.param([777], None, 2, 2, id="single_seq"),
    # GQA/MQA reach the KV head by zero-stride broadcast, never a pack.
    pytest.param([255, 1024], None, 8, 2, id="gqa_4to1"),
    pytest.param([300, 129], None, 8, 1, id="mqa_8to1"),
    # The prefix mapper groups batches 31 at a time; 70 crosses that boundary
    # twice and carries zero-length sequences across it, which subsumes the
    # single-crossing and at-the-boundary batch counts.
    pytest.param([(0 if i % 7 == 0 else 40 + 9 * i) for i in range(70)],
                 None,
                 2,
                 2,
                 id="batch70_with_zeros"),
  ],
)
def test_varlen_matches_per_sequence_oracle(
  lens_q, lens_k, num_head, num_head_kv, causal, dtype
):
  lens_k = lens_q if lens_k is None else lens_k
  torch.manual_seed(0)
  cu_q, cu_k = _cu(lens_q), _cu(lens_k)
  total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
  d = SM100_D512
  q = torch.randn(total_q, num_head, d, device="cuda", dtype=dtype) / 4
  k, v = [
    torch.randn(total_k, num_head_kv, d, device="cuda", dtype=dtype) / 4
    for _ in range(2)
  ]

  # Routing first: correct numbers from the fallback are not evidence here.
  assert _ffpa_fwd_sm100._sm100_d512_fallback_reason(
    q,
    k,
    v,
    head_dim=d,
    head_dim_v=d,
    requires_grad=False,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
  ) is None

  out, lse = ffpa_attn_varlen_func(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max(lens_q),
    max(lens_k),
    causal=causal,
    enable_gqa=num_head_kv != num_head,
    return_lse=True,
  )
  assert out.shape == (total_q, num_head, d)
  assert lse.shape == (num_head, total_q)

  ref_o, ref_lse = _varlen_oracle(q, k, v, cu_q, cu_k, causal, num_head)
  torch.testing.assert_close(out.float(), ref_o, **_tol(dtype))
  finite = torch.isfinite(ref_lse)
  torch.testing.assert_close(lse[finite], ref_lse[finite], atol=5e-2, rtol=5e-2)
  # Empty rows must be exactly -inf, not merely small: the contract names
  # the value, and a large negative number would still be a defect.
  assert torch.all(lse[~finite] == -float("inf"))


@requires_sm100a
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_varlen_zero_legal_key_reads_no_kv(dtype):
  """Every row is empty, so no K/V byte may reach an output.

  NaN-poisoning the whole K/V buffer turns any leaked read into a NaN rather
  than a small error, which a tolerance check would absorb.  This is the
  device witness that ``O = 0`` / ``LSE = -inf`` comes from the mask and not
  from arithmetic on data that should never have been loaded.
  """
  torch.manual_seed(0)
  lens_q, d = [130, 256, 1], SM100_D512
  cu_q, cu_k = _cu(lens_q), _cu([0, 0, 0])
  q = torch.randn(int(cu_q[-1]), 2, d, device="cuda", dtype=dtype)
  k, v = [
    torch.full((0, 2, d), float("nan"), device="cuda", dtype=dtype)
    for _ in range(2)
  ]

  out, lse = ffpa_attn_varlen_func(
    q, k, v, cu_q, cu_k, max(lens_q), 1, causal=True, return_lse=True
  )
  assert torch.all(out == 0)
  assert torch.all(lse == -float("inf"))


@requires_sm100a
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
  "lens",
  [
    [1024, 320, 2048],  # one long, one non-tile-multiple, one long
    [512, 64, 1536, 200, 1024],  # five sequences, two of them short
  ],
  ids=["v3", "v5"],
)
def test_bwd_varlen_matches_per_sequence_oracle(lens, causal):
  """Packed varlen gradients, per sequence, against an fp32 oracle.

  Long enough to span many Q tiles, and deliberately not all tile multiples,
  because the cumulative ``padded_offset_q`` is where a wrong statistics
  extent shows up and it only shows up past the first sequence.
  """
  cu, q, k, v, do = _packed_qkv(lens, 2, seed=sum(lens), grad=True)
  out = ffpa_attn_varlen_func(
    q, k, v, cu, cu, max(lens), max(lens), causal=causal
  )
  out.backward(do)
  scale = 1.0 / math.sqrt(SM100_D512)

  for i, ln in enumerate(lens):
    lo, hi = int(cu[i]), int(cu[i + 1])
    refs = _ref_bwd(
      q[lo:hi].detach().unsqueeze(0),
      k[lo:hi].detach().unsqueeze(0),
      v[lo:hi].detach().unsqueeze(0),
      do[lo:hi].unsqueeze(0),
      causal,
      scale,
    )
    _assert_grads_match(
      [(name, t.grad[lo:hi], ref[0])
       for name, t, ref in zip(("dQ", "dK", "dV"), (q, k, v), refs)],
      tag=(f"seq{i}", ln),
    )


# ---------------------------------------------------------------------------
# Backward — contract seams the public wrapper cannot reach
# ---------------------------------------------------------------------------


@requires_sm100a
@pytest.mark.parametrize(
  "kwargs, message",
  [
    pytest.param(lambda: {"softcap": 1.0}, "softcap", id="softcap"),
    pytest.param(lambda: {"window_size_left": 8}, "local/window", id="window"),
    # One prefix alone would have each lane deriving lengths differently.
    pytest.param(
      lambda: dict(cu_seqlens_q=_cu([64]), max_seqlen_q=64),
      "requires both",
      id="varlen-one-sided"
    ),
  ],
)
def test_bwd_rejects_out_of_contract_features(kwargs, message):
  q, k, v, do = _qkv(1, 64, 64, 2, 2, seed=0)
  out = torch.zeros_like(q)
  lse = torch.zeros(1, 2, 64, device="cuda", dtype=torch.float32)
  with pytest.raises(NotImplementedError, match=message):
    _ffpa_attn_backward_sm100(q, k, v, out, do, lse, causal=False, **kwargs())


def test_bwd_lanes_agree_on_the_q_tile():
  """The padded statistics layout serves all three lanes, so it is asserted.

  ``SeqlenInfoQK.padded_offset_q`` is formed with ``tile_m``; if the three
  lanes disagreed on it, one of them would read another's padding.  The
  witness reads each kernel's declared CTA tile, which is what the wrapper
  constructs them with.
  """
  assert (
    FFPAAttnBwdDVSm100D512.TARGET_CTA_TILER[0] ==
    FFPAAttnBwdDKSm100D512.TARGET_CTA_TILER[0] ==
    FFPAAttnBwdDQSm100D512.TARGET_CTA_TILER[0] == SM100_BWD_TILE_M == 64
  )


@requires_sm100a
def test_bwd_dv_materialisation_rejects_non_k_major_do():
  """dO must be K-major, asked rather than assumed.

  q, k and dv were always rejected when not K-major; dO used to be assumed
  and never checked (a documented gap, closed by the style-alignment work).
  The public wrapper cannot reach this seam -- ``maybe_contiguous`` hands the
  kernel a contiguous ``dout`` -- so it is exercised at the kernel boundary.

  The rejection is layered: the stride-divisibility contract
  (``cute.assume(divby=64)``), the DSL's ``LayoutEnum`` classification, and
  the kernel's explicit K-major check (symmetric with q/k/dv) each guard a
  slice of the bad-layout space -- batch-major storage dies at the first,
  head-middle at the second.  What is pinned here is the *seam behaviour*:
  every non-K-major dO dies at construction, never as silent wrong numbers.
  """
  b, s, h, d = 2, 128, 8, SM100_D512
  kernel = FFPAAttnBwdDVSm100D512(
    cutlass.Float32, (SM100_BWD_TILE_M, SM100_BWD_TILE_N_DKDV, SM100_D512),
    False, None, None, False, True
  )
  torch.manual_seed(0)
  q, k = [
    torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    for _ in range(2)
  ]
  dv = torch.zeros(b, s, h, d, device="cuda", dtype=torch.bfloat16)
  lse = torch.zeros(b, h, s, device="cuda", dtype=torch.float32)
  q_t, k_t, dv_t = [to_cute_tensor(t) for t in (q, k, dv)]
  lse_t = to_cute_tensor(lse, assumed_align=4)
  stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  mk = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
  for do, leading_dim, message in (
    (mk(s, h, d, b).permute(3, 0, 1, 2), 0, "divisible by 64"),
    (mk(b, s, d, h).transpose(2, 3), 2, "leading dimension"),
  ):
    with pytest.raises(ValueError, match=message):
      cute.compile(
        kernel, q_t, k_t, to_cute_tensor(do, leading_dim=leading_dim), lse_t,
        dv_t, 1.0 / math.sqrt(d), None, None, stream
      )


@requires_sm100a
@pytest.mark.parametrize("varlen", [False, True])
def test_bwd_stats_padding_is_fully_written_not_inherited(monkeypatch, varlen):
  """Nothing the three kernels read may come from uninitialised memory.

  The wrapper allocates ``lse_log2`` and ``dpsum`` with ``torch.empty`` over
  the *padded* extent and relies on the preprocess to cover all of it: real
  rows get their value, rows past a sequence get ``+inf`` LSE and zero dpsum,
  and rows the forward marked empty (``LSE = -inf``) get ``0.0``. Poison the
  buffers first, so any gap shows up as a NaN gradient instead of as whatever
  the allocator happened to return.
  """
  import ffpa_attn.cute._ffpa_bwd_sm100 as bwd_mod

  real_empty = torch.empty

  def poisoned_empty(*args, **kwargs):
    t = real_empty(*args, **kwargs)
    # Narrow on purpose: only the rank-2/3 fp32 buffers this path allocates
    # (the two statistics buffers, and LSE on the forward side). Poisoning
    # every fp32 allocation in the process would put NaN into unrelated
    # internals and turn this into a test of the allocator.
    if t.dtype == torch.float32 and t.dim() in (2, 3):
      t.fill_(float("nan"))
    return t

  monkeypatch.setattr(bwd_mod.torch, "empty", poisoned_empty)

  # Non-tile-multiple lengths on purpose: 200 is not a multiple of 64, and in
  # the varlen arm the padding between two sequences' statistics is exactly
  # what a wrong extent would expose.
  lens = [100, 200, 33] if varlen else [200]
  if varlen:
    cu, q, k, v, do = _packed_qkv(lens, 2, seed=23)
    extra = dict(
      cu_seqlens_q=cu,
      cu_seqlens_k=cu,
      max_seqlen_q=max(lens),
      max_seqlen_k=max(lens),
    )
  else:
    q, k, v, do = _qkv(1, lens[0], lens[0], 2, 2, seed=23)
    extra = {}

  scale = 1.0 / math.sqrt(SM100_D512)
  out, lse = _ffpa_attn_forward_sm100(
    q, k, v, softmax_scale=scale, causal=True, return_lse=True, **extra
  )
  grads = _ffpa_attn_backward_sm100(
    q, k, v, out, do, lse, softmax_scale=scale, causal=True, **extra
  )
  for lane, g in zip(("dQ", "dK", "dV"), grads):
    assert torch.isfinite(g).all(), lane
