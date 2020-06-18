#pragma once

#define GEMM_OPTIMIZE_4X4(FUNC_NAME) \
    void FUNC_NAME(float* A, float* B, float* C, int m, int n, int k);

GEMM_OPTIMIZE_4X4(Optimize3_4x4)
GEMM_OPTIMIZE_4X4(Optimize4_4x4)
GEMM_OPTIMIZE_4X4(Optimize5_4x4)
GEMM_OPTIMIZE_4X4(Optimize6_4x4)
GEMM_OPTIMIZE_4X4(Optimize7_4x4)
GEMM_OPTIMIZE_4X4(Optimize9_4x4)