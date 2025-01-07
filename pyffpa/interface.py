from enum import Enum
from functools import partial
from typing import Optional

import torch

# pyffpa_cuda.cpython.*.so
from pyffpa_cuda import ffpa_mma_acc_f16_L1, ffpa_mma_acc_f32_L1


class LevelType(Enum):
    L1 = 0
    L2 = 1
    L3 = 2


class MMAAccType(Enum):
    FP32 = 0
    FP16 = 1


def faster_prefill_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: Optional[torch.Tensor] = None,
    num_stages: int = 2,
    level: LevelType = LevelType.L1,
    acc: MMAAccType = MMAAccType.FP32,
):
    # Q, K, V, O: [B, H, N, D] layout
    if not isinstance(o, torch.Tensor) or o is None:
        o = torch.zeros_like(q)
    assert level == LevelType.L1, "only support FFPA L1 level now."
    if acc == MMAAccType.FP32:
        ffpa_mma_acc_f32_L1(q, k, v, o, num_stages)
    else:
        ffpa_mma_acc_f16_L1(q, k, v, o, num_stages)
    return o


ffpa: callable = faster_prefill_attn_func
ffpa_acc_f32_L1 = partial(
    faster_prefill_attn_func, level=LevelType.L1, acc=MMAAccType.FP32
)
ffpa_acc_f16_L1 = partial(
    faster_prefill_attn_func, level=LevelType.L1, acc=MMAAccType.FP16
)
