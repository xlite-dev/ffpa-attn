"""Ray-based scheduling engine for parallel FFPA autotune.

Distributes a list of :class:`~ffpa_attn.autotune.TuneTask` objects
across a pool of :class:`TritonAutotuneWorker` actors, one per GPU, and
collects the merged results.
"""

from __future__ import annotations

import os
import time
from typing import Any

import ray
from ray.exceptions import RayActorError, RayTaskError

from ..logger import init_logger

logger = init_logger(__name__)


def _task_desc(task: Any) -> str:
  """Return a compact human-readable label for a tuning task.

    :param task: :class:`~ffpa_attn.autotune.TuneTask` instance.
    :returns: Short description string for progress logs.
    """
  return (
    f"{task.direction[:3]}:{task.case_name},"
    f"D{task.headdim},Q{task.seqlen_q}xK{task.seqlen_k}"
  )


def _submit_task(worker: Any, task: Any, args: Any) -> ray.ObjectRef:
  """Submit one task to a worker and return the future.

    :param worker: :class:`TritonAutotuneWorker` actor handle.
    :param task: :class:`TuneTask` to run.
    :param args: Parsed CLI namespace.
    :returns: Ray ObjectRef for the pending task.
    """
  return worker.run_task.remote(
    task,
    args.B,
    args.mode,
    enable_fwd_tma=args.enable_fwd_tma,
    enable_fwd_ws=args.enable_fwd_ws,
    enable_bwd_tma=args.enable_bwd_tma,
    enable_bwd_ws=args.enable_bwd_ws,
    enable_bwd_split_launch=args.enable_bwd_split_launch,
  )


def run_ray_autotune(tasks: list[Any], args: Any) -> list[dict[str, Any]]:
  """Run all *tasks* across *args.num_gpus* GPUs via Ray.

    :param tasks: List of :class:`~ffpa_attn.autotune.TuneTask` objects.
    :param args: Parsed CLI namespace.
    :returns: Merged entry dicts ready for ``_record_entry``.
    :raises SystemExit: If fewer GPUs are available than requested.
    """
  if not tasks:
    return []

  os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
  ray.init(
    address=args.ray_address, ignore_reinit_error=True, include_dashboard=False
  )
  try:
    gpu_count = int(ray.cluster_resources().get("GPU", 0))
    if gpu_count < args.num_gpus:
      raise SystemExit(
        f"Requested {args.num_gpus} GPUs but only {gpu_count} available. "
        f"Check CUDA_VISIBLE_DEVICES or --ray-address."
      )

    from ._autotune_worker import TritonAutotuneWorker

    workers = [
      TritonAutotuneWorker.options(num_gpus=1).remote()
      for _ in range(args.num_gpus)
    ]

    all_entries: list[dict[str, Any]] = []
    total = len(tasks)
    pending: dict[ray.ObjectRef, tuple[int, Any, float]] = {}
    task_index = 0

    # Phase A: fill every worker initially
    for i, worker in enumerate(workers):
      if task_index >= total:
        break
      task = tasks[task_index]
      future = _submit_task(worker, task, args)
      pending[future] = (i, task, time.perf_counter())
      task_index += 1

    # Phase B: as tasks complete, dispatch new ones
    while pending:
      ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=None)
      for future in ready:
        worker_idx, task, submit_time = pending.pop(future)
        try:
          entries = ray.get(future)
          all_entries.extend(entries)
        except (RayActorError, RayTaskError) as exc:
          logger.warning(
            "Worker %d failed on task %s: %s",
            worker_idx,
            _task_desc(task),
            exc,
          )
        elapsed = time.perf_counter() - submit_time
        done = total - len(pending) - (total - task_index)
        logger.info(
          "[AUTOTUNED][%d/%d] %s, t=%.1fs",
          done,
          total,
          _task_desc(task),
          elapsed,
        )
        if task_index < total:
          next_task = tasks[task_index]
          future = _submit_task(workers[worker_idx], next_task, args)
          pending[future] = (
            worker_idx,
            next_task,
            time.perf_counter(),
          )
          task_index += 1

    return all_entries
  finally:
    ray.shutdown()
