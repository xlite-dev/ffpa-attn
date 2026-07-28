// SM120a TMA + MMA warp-specialised forward dispatch.
// Compiled by nvcc (unlike ffpa_attn_api.cc which uses the host compiler),
// so it can include launch_templates.cuh -> tma.cuh -> <cuda/barrier>.
#include "launch_templates.cuh"

#include <torch/extension.h>

// Returns true if the SM120a TMA+MMA WS path handled the call (device is
// sm_120a, tma requested, dtype/acc supported); false otherwise (caller
// falls back to the architecture-agnostic generated dispatch).
bool ffpa_attn_fwd_sm120_dispatch(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V, torch::Tensor O,
                                  torch::Tensor attn_bias,
                                  torch::Tensor softmax_lse, int64_t acc,
                                  int causal, double softmax_scale,
                                  double dropout_p, int64_t philox_seed,
                                  int64_t philox_offset) {
  auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major != 12 || prop->minor != 0) {
    return false;
  }
  const auto dtype = Q.scalar_type();
  constexpr int kSm120StageQK = 4;
  constexpr int kSm120StagePV = 4;
  if (dtype == torch::kHalf) {
    if (acc == 1) {
      launch_ffpa_attn_fwd_template_sm120<__half, 512, 1, 1, kSm120StageQK,
                                          kSm120StagePV>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset);
    } else {
      launch_ffpa_attn_fwd_template_sm120<__half, 512, 0, 0, kSm120StageQK,
                                          kSm120StagePV>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset);
    }
    return true;
  }
  if (dtype == torch::kBFloat16) {
    launch_ffpa_attn_fwd_template_sm120<__nv_bfloat16, 512, 1, 1, kSm120StageQK,
                                        kSm120StagePV>(
        Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
        philox_seed, philox_offset);
    return true;
  }
  return false;
}
