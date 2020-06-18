#include "helpler.h"
#include "Optimize1x4.h"

// C=A*B+C
// A(m*n)
// B(nk)
// C(m*k)
void Optimize3_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize4_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_inline(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize5_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_fused(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize6_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_register(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize7_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_reduce_B_indexing(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize8_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_unroll_k(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize9_1x4( float* A,
                    float* B,
                    float* C,
                    int m,
                    int n,
                    int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot1x4_reduce_ptrB_update(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}