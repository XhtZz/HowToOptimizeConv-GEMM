#pragma once

#define A(i,j) A[(j)*m+(i)]
#define B(i,j) B[(j)*k+(i)]
#define C(i,j) C[(j)*m+(i)]

#define A_row(p) A_row[m*(p)]
#define min(x,y) (x)<(y)?(x):(y)

inline static void AddDot(float* A_row, float* B_col, float* c, int m, int n, int k){
    for(int p=0;p<k;++p){
        *c += A_row(p) * B_col[p];
    }
}

//Optimize_1x4
void AddDot1x4(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_inline(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_fused(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_register(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_reduce_B_indexing(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_unroll_k(float* A, float* B, float* C, int m, int n, int k);
void AddDot1x4_reduce_ptrB_update(float* A, float* B, float* C, int m, int n, int k);

//Optimize_4x4
void AddDot4x4(float* A, float* B, float* C, int m, int n, int k);
void AddDot4x4_inline(float* A, float* B, float* C, int m, int n, int k);
void AddDot4x4_fused(float* A, float* B, float* C, int m, int n, int k);
void AddDot4x4_register(float* A, float* B, float* C, int m, int n, int k);
void AddDot4x4_reduce_B_indexing(float* A, float* B, float* C, int m, int n, int k);
void AddDot4x4_rows_rearrange(float* A, float* B, float* C, int m, int n, int k);

//Optimize_8x8
void AddDot8x8_vector_operator_mul_add(float* A, float* B, float* C, int m, int n, int k);
void AddDot8x8_vector_operator_fma(float* A, float* B, float* C, int m, int n, int k);
void AddDot8x8_block_mc_n(float* A, float* B, float* C, int pb, int m, int n, int k);
void AddDot8x8_packed_A(float* a, float* B, float* C, int pb, int m, int n, int k);
void AddDot8x8_packed_AB(float* a, float* B, float* C, int pb, int m, int n, int k);
