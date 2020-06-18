#pragma once

#define GEMM_OPTIMIZE_1X4(FUNC_NAME) \
    void FUNC_NAME(float* A, float* B, float* C, int m, int n, int k);

GEMM_OPTIMIZE_1X4(Optimize3_1x4)
GEMM_OPTIMIZE_1X4(Optimize4_1x4)
GEMM_OPTIMIZE_1X4(Optimize5_1x4)
GEMM_OPTIMIZE_1X4(Optimize6_1x4)
GEMM_OPTIMIZE_1X4(Optimize7_1x4)
GEMM_OPTIMIZE_1X4(Optimize8_1x4)
GEMM_OPTIMIZE_1X4(Optimize9_1x4)
