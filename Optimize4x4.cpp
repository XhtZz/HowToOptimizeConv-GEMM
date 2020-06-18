#include "helpler.h"
#include "Optimize4x4.h"

// C=A*B+C
// A(m*n)
// B(nk)
// C(m*k)
void Optimize3_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}


void Optimize4_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4_inline(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}


void Optimize5_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4_fused(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize6_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4_register(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize7_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4_reduce_B_indexing(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize9_4x4( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    for(int i=0;i<m;i+=4){
        for(int j=0;j<n;j+=4){
            AddDot4x4_rows_rearrange(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

