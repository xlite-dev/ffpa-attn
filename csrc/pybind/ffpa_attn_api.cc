#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void ffpa_mma_acc_f16_L1(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         torch::Tensor O, int stages);

void ffpa_mma_acc_f32_L1(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         torch::Tensor O, int stages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(ffpa_mma_acc_f16_L1)
  TORCH_BINDING_COMMON_EXTENSION(ffpa_mma_acc_f32_L1)
}
