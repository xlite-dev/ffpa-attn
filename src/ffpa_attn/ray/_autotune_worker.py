"""Ray actor that runs a single FFPA Triton autotune task on one GPU.

The actor owns exactly one CUDA device (``num_gpus=1``) and calls the
existing ``_tune_forward`` / ``_tune_backward`` functions from
:mod:`ffpa_attn.autotune`.  Imports are deferred to ``run_task`` to
avoid circular import risks when the worker module is loaded by Ray.

Each worker is assigned an isolated Triton cache directory via
``TRITON_CACHE_DIR`` to prevent concurrent cache-file races when
multiple workers compile the same kernel family on the same machine.

Results are returned as entry dicts; the caller is responsible for
merging and deduplication.

Future backends (e.g. CuTeDSL) should add their own worker class and
follow the same pattern: a ``@ray.remote(num_gpus=1)`` actor with a
``run_task`` method that expects a :class:`~ffpa_attn.autotune.TuneTask`
and returns entry dicts.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import ray
import torch

if TYPE_CHECKING:
  from ..autotune import TuneTask


@ray.remote(num_gpus=1)
class TritonAutotuneWorker:
  """Ray actor that benchmarks Triton autotune configs on its assigned GPU.

    Ray isolates the actor to a single GPU via ``num_gpus=1``, so the
    actor only ever sees device index 0.

    Each actor creates a private Triton cache directory so that
    concurrent JIT compilations across workers do not race on
    ``~/.triton/cache``.
    """

  def __init__(self) -> None:
    torch.cuda.set_device(0)
    self._triton_cache = tempfile.mkdtemp(prefix="ffpa_triton_cache_")
    os.environ["TRITON_CACHE_DIR"] = self._triton_cache

  def run_task(
    self,
    task: TuneTask,
    batch: int,
    mode: str,
    enable_fwd_tma: bool = False,
    enable_fwd_ws: bool = False,
    enable_bwd_tma: bool = False,
    enable_bwd_ws: bool = False,
    enable_bwd_split_launch: bool = False,
  ) -> list[dict]:
    """Run one Triton autotune task and return the resulting entry dicts.

        Imports are deferred to this method to break potential circular
        dependencies between :mod:`ffpa_attn.ray` and
        :mod:`ffpa_attn.autotune`.

        :param task: Shape / dtype / direction descriptor.
        :param batch: Batch size for tuning.
        :param mode: Triton autotune search-space mode (``"fast"`` or ``"max"``).
        :param enable_fwd_tma: Enable SM90 TMA forward path.
        :param enable_fwd_ws: Force warp-specialized forward configs.
        :param enable_bwd_tma: Enable SM90 TMA backward path.
        :param enable_bwd_ws: Force warp-specialized backward configs.
        :param enable_bwd_split_launch: Also tune split-launch dK/dV + dQ.
        :returns: List of entry dicts, or an empty list on OOM.
        """
    from ..autotune import _tune_backward, _tune_forward
    from ..triton._autotune_utils import exact_autotune_seqlen_keys

    with exact_autotune_seqlen_keys():
      try:
        if task.direction == "forward":
          tuned = _tune_forward(
            task,
            batch,
            mode,
            {},
            enable_tma=enable_fwd_tma,
            enable_ws=enable_fwd_ws,
          )
        else:
          tuned = _tune_backward(
            task,
            batch,
            mode,
            {},
            enable_tma=enable_bwd_tma,
            enable_ws=enable_bwd_ws,
            enable_split_launch=enable_bwd_split_launch,
          )
        torch.cuda.synchronize()
        return [entry for entry, _ in tuned]
      except Exception as exc:
        if isinstance(exc,
                      RuntimeError) and "out of memory" in str(exc).lower():
          torch.cuda.empty_cache()
          return []
        import logging
        logging.getLogger(__name__).warning(
          "Task failed: %s D=%d Q=%dxK=%d: %s",
          task.direction,
          task.headdim,
          task.seqlen_q,
          task.seqlen_k,
          exc,
        )
        return []
