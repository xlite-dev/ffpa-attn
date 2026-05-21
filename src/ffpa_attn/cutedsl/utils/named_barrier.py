# This file is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/named_barrier.py
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

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
