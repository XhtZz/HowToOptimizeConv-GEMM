#pragma once

#include "helpler.h"

// C = A * B + C
// A(m*n)
// B(n*k)
// C(m*k)
inline static void Matrix_base(float* A,
                                float* B,
                                float* C,
                                int m,
                                int n,
                                int k){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            for(int p=0;p<k;++p){
                C(i,j) = C(i,j) + A(i,p)*B(p,j);
            }
        }
    }
    return;
}

inline static void Optimize1( float* A,
                            float* B,
                            float* C,
                            int m,
                            int n,
                            int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            AddDot(&A(i,0),&B(0,j),&C(i,j),m,n,k);
        }
    }
    return;
}

inline static void Optimize2( float* A,
                            float* B,
                            float* C,
                            int m,
                            int n,
                            int k ){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;j+=4){
            AddDot(&A(i,0),&B(0,j),&C(i,j),m,n,k);
            AddDot(&A(i,0),&B(0,j+1),&C(i,j+1),m,n,k);
            AddDot(&A(i,0),&B(0,j+2),&C(i,j+2),m,n,k);
            AddDot(&A(i,0),&B(0,j+3),&C(i,j+3),m,n,k);
        }
    }
    return;
}

