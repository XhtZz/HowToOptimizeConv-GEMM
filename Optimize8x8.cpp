#include "helpler.h"
#include "Optimize8x8.h"

void Optimize10_8x8_mul_add( float* A, 
                            float* B, 
                            float* C, 
                            int m, 
                            int n, 
                            int k ){
    for(int i=0;i<m;i+=8){
        for(int j=0;j<n;j+=8){
            AddDot8x8_vector_operator_mul_add(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

void Optimize10_8x8_fma( float* A, 
                        float* B, 
                        float* C, 
                        int m, 
                        int n, 
                        int k ){
    for(int i=0;i<m;i+=8){
        for(int j=0;j<n;j+=8){
            AddDot8x8_vector_operator_fma(&A(i,0), &B(0,j), &C(i,j), m, n, k);
        }
    }
    return;
}

/* Block sizes */
#define mc 256
#define kc 128
#define nb 2000

static void Kernel_block( float* A, 
                    float* B, 
                    float* C, 
                    int ib,
                    int pb,
                    int m, 
                    int n, 
                    int k ){
    for(int j=0;j<n;j+=8){
        for(int i=0;i<ib;i+=8){
            AddDot8x8_block_mc_n(&A(i,0), &B(0,j), &C(i,j), pb, m, n, k);
        }
    }
    return;
}

void Optimize11_8x8( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    int i, j, p, pb, ib;

    for ( p=0; p<k; p+=kc ){
        pb = min( k-p, kc );
        for ( i=0; i<m; i+=mc ){
            ib = min( m-i, mc );
            Kernel_block(&A( i,p ), &B(p, 0 ), &C( i,0 ), ib, pb, m, n, k);
        }
    }
}

static void packMatrixA(float* A, float* packed_A, int pb, int m){
    int i = 0, j = 0;
    for(j=0;j<pb;++j){
        float* Aij= &A(0,j);
        *packed_A++ = *Aij;
        *packed_A++ = *(Aij+1);
        *packed_A++ = *(Aij+2);
        *packed_A++ = *(Aij+3);
        *packed_A++ = *(Aij+4);
        *packed_A++ = *(Aij+5);
        *packed_A++ = *(Aij+6);
        *packed_A++ = *(Aij+7);
    }
}

static void Kernel_packed_A( float* A, 
                            float* B, 
                            float* C, 
                            int ib,
                            int pb,
                            int m, 
                            int n, 
                            int k ){
    float packed_A[ib*pb];
    for(int j=0;j<n;j+=8){ 
        for(int i=0;i<ib;i+=8){
            if(j==0) packMatrixA(&A(i,0),&packed_A[i*pb],pb,m);
            AddDot8x8_packed_A(&packed_A[i*pb], &B(0,j), &C(i,j), pb, m, n, k);
        }
    }
    return;
}

void Optimize12_13_8x8( float* A, 
                        float* B, 
                        float* C, 
                        int m, 
                        int n, 
                        int k ){
    int i, j, p, pb, ib;

    for ( p=0; p<k; p+=kc ){
        pb = min( k-p, kc );
        for ( i=0; i<m; i+=mc ){
            ib = min( m-i, mc );
            Kernel_packed_A(&A( i,p ), &B(p, 0 ), &C( i,0 ), ib, pb, m, n, k);
        }
    }
}

static void packMatrixB(float* B, float* packed_B, int pb, int k){
    int i = 0, j = 0;
    float* col0 = &B(0,j); 
    float* col1 = &B(0,j+1);
    float* col2 = &B(0,j+2); 
    float* col3 = &B(0,j+3); 
    float* col4 = &B(0,j+4); 
    float* col5 = &B(0,j+5); 
    float* col6 = &B(0,j+6); 
    float* col7 = &B(0,j+7); 
    
    for(j=0;j<pb;++j){
        *packed_B++ = *col0++;
        *packed_B++ = *col1++;
        *packed_B++ = *col2++;
        *packed_B++ = *col3++;
        *packed_B++ = *col4++;
        *packed_B++ = *col5++;
        *packed_B++ = *col6++;
        *packed_B++ = *col7++;
    }
}

static void Kernel_packed_AB( float* A, 
                            float* B, 
                            float* C, 
                            int ib,
                            int pb,
                            int m, 
                            int n, 
                            int k,
                            bool is_1st ){
    float packed_A[ib*pb];
    static float packed_B[kc*nb];
    for(int j=0;j<n;j+=8){ 
        if(is_1st) packMatrixB(&B(0,j),&packed_B[j*pb],pb,k);
        for(int i=0;i<ib;i+=8){
            if(j==0) packMatrixA(&A(i,0),&packed_A[i*pb],pb,m);
            AddDot8x8_packed_AB(&packed_A[i*pb], &packed_B[j*pb], &C(i,j), pb, m, n, k);
        }
    }
    return;
}

void Optimize14_15_8x8( float* A, 
                    float* B, 
                    float* C, 
                    int m, 
                    int n, 
                    int k ){
    int i, j, p, pb, ib;

    for ( p=0; p<k; p+=kc ){
        pb = min( k-p, kc );
        for ( i=0; i<m; i+=mc ){
            ib = min( m-i, mc );
            Kernel_packed_AB(&A( i,p ), &B(p, 0 ), &C( i,0 ), ib, pb, m, n, k, i==0);
        }
    }
}