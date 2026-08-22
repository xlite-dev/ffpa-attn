// Numerical gate for the sm_120 MXFP8 block-scaled MMA the Phase-4 MXFP8
// PV path needs:
//   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale
//       .scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
// One warp computes C(16x8) = A(16x32) @ B(32x8) with one E8M0 scale per A
// row (32-K group) and per B col. Checks:
//   T1 uniform scales 2^0 -> C matches the fp32 reference exactly
//   T2 SFA=2^1, SFB=2^2   -> C == 8x the T1 result
// Scale selectors {byte-id, thread-id}: SFA {0,0} -> lanes %4 in {0,1}
// provide (byte 0 of their u32); SFB {0,0} -> lane %4 == 0 provides.
// Build: nvcc -O2 -arch=sm_120f tools/mxfp8_probe/mxfp8_probe.cu -o mxfp8_probe
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CK(x)                                           \
  do {                                                  \
    cudaError_t e_ = (x);                               \
    if (e_ != cudaSuccess) {                            \
      printf("CUDA err %s:%d %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(e_));                   \
      return 1;                                         \
    }                                                   \
  } while (0)

static uint8_t f32_to_e4m3_host(float v) {
  __nv_fp8_e4m3 r(v);
  return *reinterpret_cast<uint8_t*>(&r);
}
static float e4m3_to_f32_host(uint8_t b) {
  __nv_fp8_e4m3 r;
  *reinterpret_cast<uint8_t*>(&r) = b;
  return float(r);
}

__global__ void mxfp8_mma_kernel(const uint8_t* __restrict__ A,
                                 const uint8_t* __restrict__ B, int sfa_exp,
                                 int sfb_exp, float* __restrict__ Cout) {
  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int tid4 = lane & 3;

  // A fragment (m16n8k32, 8-bit types): reg r holds a[4r..4r+3];
  // rows {gid, gid+8}, cols tid4*4+{0..3} (regs 0,1) / +16 (regs 2,3).
  uint32_t a[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = gid + ((r & 1) ? 8 : 0);
    const int col_base = tid4 * 4 + ((r & 2) ? 16 : 0);
    uint32_t w = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b)
      w |= uint32_t(A[row * 32 + col_base + b]) << (8 * b);
    a[r] = w;
  }
  // B fragment: reg r holds b[4r..4r+3]; rows tid4*4+{0..3} (+16 reg 1),
  // col = gid. B stored [k][n] row-major (32 x 8).
  uint32_t b[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t w = 0;
#pragma unroll
    for (int bb = 0; bb < 4; ++bb)
      w |= uint32_t(B[(tid4 * 4 + bb + (r ? 16 : 0)) * 8 + gid]) << (8 * bb);
    b[r] = w;
  }

  // E8M0 encoding: value = 2^(code-127). Uniform scales for the gate tests,
  // so the provider->row mapping does not matter yet. Non-provider lanes are
  // ignored by the hardware (thread-id selectors pick the providers).
  const uint32_t sfa = uint32_t(uint8_t(sfa_exp + 127));
  const uint32_t sfb = uint32_t(uint8_t(sfb_exp + 127));

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sfa), "r"(sfb));

  // C layout m16n8: rows {gid, gid+8}, cols tid4*2+{0,1}.
  Cout[(gid) * 8 + tid4 * 2 + 0] = c0;
  Cout[(gid) * 8 + tid4 * 2 + 1] = c1;
  Cout[(gid + 8) * 8 + tid4 * 2 + 0] = c2;
  Cout[(gid + 8) * 8 + tid4 * 2 + 1] = c3;
}

// T3: per-row SFA / per-col SFB provider mapping, derived from the CUTLASS
// MMA_Traits SF layouts (cute/atom/mma_traits_sm120.hpp):
//   SFALayout Shape((2,2,8),32) Stride((8,0,1),16): row m = 8*a+c is
//     provided (2:0 broadcast) by lanes 16a+c and 16a+c+8, byte 0.
//   SFBLayout Shape((4,8),32) Stride((0,1),8): col n is provided (4:0
//     broadcast) by every lane with lane%8 == n, byte 0.
__global__ void mxfp8_mma_per_rc_kernel(const uint8_t* __restrict__ A,
                                        const uint8_t* __restrict__ B,
                                        const uint8_t* __restrict__ SFArow,
                                        const uint8_t* __restrict__ SFBcol,
                                        float* __restrict__ Cout) {
  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int tid4 = lane & 3;

  uint32_t a[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = gid + ((r & 1) ? 8 : 0);
    const int col_base = tid4 * 4 + ((r & 2) ? 16 : 0);
    uint32_t w = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b)
      w |= uint32_t(A[row * 32 + col_base + b]) << (8 * b);
    a[r] = w;
  }
  uint32_t b[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t w = 0;
#pragma unroll
    for (int bb = 0; bb < 4; ++bb)
      w |= uint32_t(B[(tid4 * 4 + bb + (r ? 16 : 0)) * 8 + gid]) << (8 * bb);
    b[r] = w;
  }

  const uint32_t sfa = uint32_t(SFArow[16 * (lane / 16) + lane % 8]);
  const uint32_t sfb = uint32_t(SFBcol[lane % 8]);

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sfa), "r"(sfb));

  Cout[(gid) * 8 + tid4 * 2 + 0] = c0;
  Cout[(gid) * 8 + tid4 * 2 + 1] = c1;
  Cout[(gid + 8) * 8 + tid4 * 2 + 0] = c2;
  Cout[(gid + 8) * 8 + tid4 * 2 + 1] = c3;
}

// T4: brute-force SF provider discovery. mode 0 probes SFA: A[i][k] =
// 2*(k==i), B all ones, SFB all 2^0; lane L's SFA byte0 = 2^(L%5 + 1).
// C[i][j] = 2 * SF_hw(i) reveals which lane byte each row used.
// mode 1 probes SFB: A all ones, B[k][j] = 2*(k==j), SFA all 2^0; lane
// L's SFB byte0 = 2^(L%5 + 1). C[i][j] = 2 * SF_hw(j).
__global__ void mxfp8_sf_discover_kernel(const uint8_t* __restrict__ A,
                                         const uint8_t* __restrict__ B,
                                         int mode, float* __restrict__ Cout) {
  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int tid4 = lane & 3;

  uint32_t a[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = gid + ((r & 1) ? 8 : 0);
    const int col_base = tid4 * 4 + ((r & 2) ? 16 : 0);
    uint32_t w = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b)
      w |= uint32_t(A[row * 32 + col_base + b]) << (8 * b);
    a[r] = w;
  }
  uint32_t b[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t w = 0;
#pragma unroll
    for (int bb = 0; bb < 4; ++bb)
      w |= uint32_t(B[(tid4 * 4 + bb + (r ? 16 : 0)) * 8 + gid]) << (8 * bb);
    b[r] = w;
  }

  const uint32_t probe = uint32_t(uint8_t(127 + lane - 16));
  const uint32_t sfa = mode == 0 ? probe : 127u;
  const uint32_t sfb = mode == 0 ? 127u : probe;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sfa), "r"(sfb));

  Cout[(gid) * 8 + tid4 * 2 + 0] = c0;
  Cout[(gid) * 8 + tid4 * 2 + 1] = c1;
  Cout[(gid + 8) * 8 + tid4 * 2 + 0] = c2;
  Cout[(gid + 8) * 8 + tid4 * 2 + 1] = c3;
}

int main() {
  unsigned char A[16 * 32], B[32 * 8];
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 32; ++k)
      A[i * 32 + k] = f32_to_e4m3_host(0.25f * ((i + k % 7) & 7));
  for (int k = 0; k < 32; ++k)
    for (int n = 0; n < 8; ++n)
      B[k * 8 + n] = f32_to_e4m3_host(0.5f * ((n + k % 5) & 3));

  unsigned char *dA, *dB;
  float* dC;
  CK(cudaMalloc(&dA, sizeof(A)));
  CK(cudaMalloc(&dB, sizeof(B)));
  CK(cudaMalloc(&dC, 16 * 8 * sizeof(float)));
  CK(cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice));

  float ref0[16 * 8];
  for (int i = 0; i < 16; ++i)
    for (int n = 0; n < 8; ++n) {
      float acc = 0.f;
      for (int k = 0; k < 32; ++k)
        acc += e4m3_to_f32_host(A[i * 32 + k]) * e4m3_to_f32_host(B[k * 8 + n]);
      ref0[i * 8 + n] = acc;
    }

  int fails = 0;
  float C[16 * 8];
  for (int test = 0; test < 2; ++test) {
    const int sfa_exp = test == 0 ? 0 : 1;
    const int sfb_exp = test == 0 ? 0 : 2;
    const float expect_scale = test == 0 ? 1.f : 8.f;
    mxfp8_mma_kernel<<<1, 32>>>(dA, dB, sfa_exp, sfb_exp, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost));
    float max_err = 0.f;
    for (int i = 0; i < 16 * 8; ++i)
      max_err = fmaxf(max_err, fabsf(C[i] - ref0[i] * expect_scale));
    const bool ok = max_err < 1e-4f;
    fails += ok ? 0 : 1;
    printf("T%d (SFA=2^%d SFB=2^%d) max_err=%.6f %s\n", test + 1, sfa_exp,
           sfb_exp, max_err, ok ? "PASS" : "FAIL");
  }

  // T3: per-row/col scales. SF_A[i] = 2^((i%4)-2), SF_B[j] = 2^((j%3)-1).
  unsigned char SFArow[16], SFBcol[8], *dSFA, *dSFB;
  float ref3[16 * 8];
  for (int i = 0; i < 16; ++i) {
    const int e = (i % 4) - 2;
    SFArow[i] = uint8_t(127 + e);
  }
  for (int j = 0; j < 8; ++j) {
    const int e = (j % 3) - 1;
    SFBcol[j] = uint8_t(127 + e);
  }
  for (int i = 0; i < 16; ++i)
    for (int n = 0; n < 8; ++n) {
      float acc = 0.f;
      for (int k = 0; k < 32; ++k)
        acc += e4m3_to_f32_host(A[i * 32 + k]) * e4m3_to_f32_host(B[k * 8 + n]);
      ref3[i * 8 + n] = ldexpf(ldexpf(acc, (i % 4) - 2), (n % 3) - 1);
    }
  CK(cudaMalloc(&dSFA, 16));
  CK(cudaMalloc(&dSFB, 8));
  CK(cudaMemcpy(dSFA, SFArow, 16, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFB, SFBcol, 8, cudaMemcpyHostToDevice));
  mxfp8_mma_per_rc_kernel<<<1, 32>>>(dA, dB, dSFA, dSFB, dC);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost));
  float max_err3 = 0.f;
  for (int i = 0; i < 16 * 8; ++i)
    max_err3 = fmaxf(max_err3, fabsf(C[i] - ref3[i]));
  const bool ok3 = max_err3 < 1e-4f;
  fails += ok3 ? 0 : 1;
  printf("T3 (per-row SFA / per-col SFB) max_err=%.6f %s\n", max_err3,
         ok3 ? "PASS" : "FAIL");

  // T4: discover the hardware SF provider mapping.
  {
    unsigned char Aid[16 * 32], Bid[32 * 8];
    for (int i = 0; i < 16; ++i)
      for (int k = 0; k < 32; ++k)
        Aid[i * 32 + k] = f32_to_e4m3_host(k == i ? 2.f : 0.f);
    for (int k = 0; k < 32; ++k)
      for (int j = 0; j < 8; ++j)
        Bid[k * 8 + j] = f32_to_e4m3_host(1.f);
    CK(cudaMemcpy(dA, Aid, sizeof(Aid), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, Bid, sizeof(Bid), cudaMemcpyHostToDevice));
    mxfp8_sf_discover_kernel<<<1, 32>>>(dA, dB, 0, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost));
    printf("T4 SFA map: row -> provider lane\n");
    for (int i = 0; i < 16; ++i) {
      const int e = (int)lroundf(log2f(C[i * 8] / 2.f));
      printf("  row %2d: lane %d\n", i, e + 16);
    }
    for (int i = 0; i < 16; ++i)
      for (int k = 0; k < 32; ++k)
        Aid[i * 32 + k] = f32_to_e4m3_host(1.f);
    for (int k = 0; k < 32; ++k)
      for (int j = 0; j < 8; ++j)
        Bid[k * 8 + j] = f32_to_e4m3_host(k == j ? 2.f : 0.f);
    CK(cudaMemcpy(dA, Aid, sizeof(Aid), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, Bid, sizeof(Bid), cudaMemcpyHostToDevice));
    mxfp8_sf_discover_kernel<<<1, 32>>>(dA, dB, 1, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost));
    printf("T4 SFB map: col -> provider lane\n");
    for (int j = 0; j < 8; ++j) {
      const int e = (int)lroundf(log2f(C[j] / 2.f));
      printf("  col %d: lane %d\n", j, e + 16);
    }
  }
  printf(fails == 0 ? "ALL PASS\n" : "FAILURES=%d\n", fails);
  return fails == 0 ? 0 : 1;
}
