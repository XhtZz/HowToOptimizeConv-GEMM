#pragma once

#define GEMM_OPTIMIZE_8x8(FUNC_NAME) \
    void FUNC_NAME(float* A, float* B, float* C, int m, int n, int k);

GEMM_OPTIMIZE_8x8(Optimize10_8x8_mul_add)
GEMM_OPTIMIZE_8x8(Optimize10_8x8_fma)
GEMM_OPTIMIZE_8x8(Optimize11_8x8)
GEMM_OPTIMIZE_8x8(Optimize12_13_8x8)
GEMM_OPTIMIZE_8x8(Optimize14_15_8x8)