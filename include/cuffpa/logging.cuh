#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                  \
  if (((T2).size(0) != (T1).size(0)) || ((T2).size(1) != (T1).size(1)) || \
      ((T2).size(2) != (T1).size(2)) || ((T2).size(3) != (T1).size(3))) { \
    throw std::runtime_error("Tensor size mismatch!");                    \
  }

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
