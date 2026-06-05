"""Ray-based distributed autotune runner for ffpa-attn.

This module provides GPU-parallel autotune scheduling via Ray actors.
It is designed to be backend-agnostic: the worker delegates to the
appropriate per-backend tune function (currently Triton, with CuTeDSL
planned as a future addition).

Importing this module does **not** import ``ray``.  The Ray runtime is
only loaded on the first call to :func:`run_ray_autotune`.
"""

from __future__ import annotations


def run_ray_autotune(tasks: list, args) -> list[dict]:
  """Run *tasks* across multiple GPUs via Ray actors.

    This function is the only public entry point.  It lazily imports the
    engine so that ``import ffpa_attn.ray`` stays cheap.

    :param tasks: List of :class:`~ffpa_attn.autotune.TuneTask` objects.
    :param args: Parsed CLI namespace from :func:`~ffpa_attn.autotune._parse_args`.
    :returns: Merged list of entry dicts ready for ``_record_entry``.
    """
  from ._autotune_engine import run_ray_autotune as _impl

  return _impl(tasks, args)
