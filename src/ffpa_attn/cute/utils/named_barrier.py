# This file is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/named_barrier.py
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# The SM100 maps below come from flash-attention-512-dev's named_barrier.py.

import enum


class NamedBarrierFwd(enum.IntEnum):
  Epilogue = enum.auto(
  )  # starts from 1 as barrier 0 is reserved for sync_threads()
  WarpSchedulerWG1 = enum.auto()
  WarpSchedulerWG2 = enum.auto()
  WarpSchedulerWG3 = enum.auto()
  PFull = enum.auto()
  PEmpty = enum.auto()
  VZero = enum.auto()
  QueryEmpty = enum.auto()
  ScaleReady = enum.auto()


class NamedBarrierBwd(enum.IntEnum):
  # SM90 PTX `bar.sync`/`bar.arrive` accept barrier IDs 0..15 ONLY. IDs >15 are undefined behavior
  Epilogue = enum.auto()  # 1
  WarpSchedulerWG1 = enum.auto()  # 2
  WarpSchedulerWG2 = enum.auto()  # 3
  WarpSchedulerWG3 = enum.auto()  # 4
  PdS = enum.auto()  # 5  (V1 dQ/dKdV)
  # cooperative ① cross-WG handshake on sdS for shared Phase E.
  # NOTE: relocated into IDs 6/7 (formerly dQFullWG0/1) to stay within the
  # 0..15 range noted above.
  dSFull = enum.auto()  # 6  WG2 → WG1+WG2: sdS[0] published (256-thread)
  dSEmpty = enum.auto()  # 7  WG1+WG2 → WG2: sdS[0] consumed
  dQFullWG2 = enum.auto()  # 8  reserved/unused
  dQEmptyWG0 = enum.auto()  # 9  reserved/unused
  dQEmptyWG1 = enum.auto()  # 10 reserved/unused
  dQEmptyWG2 = enum.auto()  # 11 reserved/unused
  VTailZero = enum.auto()  # 12
  # cross-WG handshake on single-buffered sP.
  PFull = enum.auto()  # 13 WG1 → WG2: sP[0] published (256-thread barrier)
  PEmpty = enum.auto(
  )  # 14 WG2 → WG1: sP[0] consumed; init credit + polite-close
  dSLocal = enum.auto(
  )  # 15 WG2-internal STSM(sdS) → WGMMA(read sdS) fence (128-thread)


# SM100 (Blackwell) D512 2-CTA maps, disjoint from the SM90 maps above.  Ids
# are explicit rather than enum.auto() because each map is a contract with a
# specific participant count: the four maps agree on some ids and diverge on
# others, and the gaps are reserved.  Do not renumber.
# Id 0 is CtaSync-only: bar.sync 0 with the full CTA count is __syncthreads.


class NamedBarrierFwdSm100Hd512(enum.IntEnum):
  TmemPtr = 1
  # Ids 2 and 3 belonged to a retired cooperative epilogue and stay unassigned.
  # M128 pairs two softmax threads on each query row; their row-max
  # reduction meets here.
  SoftmaxPair = 4


class NamedBarrierBwdDKSm100Hd512(enum.IntEnum):
  CtaSync = 0
  TmemPtr = 1
  Compute = 2
  Epilogue = 3
  # Both compute warpgroups must finish their STS into the sK-hosted arena
  # before the leader warp's TMA bulk store reads it back out; ids 4 and 5
  # are free.
  EpilogueArena = 6


class NamedBarrierBwdDQSm100Hd512(enum.IntEnum):
  TmemPtr = 1
  Compute = 2
  # dQ has no whole-CTA sync and no separate epilogue barrier, so its id 3 is
  # the arena rendezvous the dK and dV siblings put at 6 and 5; id 7 was the
  # retired persistent path's dO-alias fence.
  EpilogueArena = 3


class NamedBarrierBwdDVSm100Hd512(enum.IntEnum):
  CtaSync = 0
  TmemPtr = 1
  Compute = 2
  Epilogue = 3
  # Both warpgroups must finish writing s_epi_dV before the leader warp reads
  # it back out through TMA; id 4 is free.
  EpilogueArena = 5
