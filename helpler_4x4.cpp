#include <immintrin.h>
#include "helpler.h"

void AddDot4x4(float* A, float* B, float* C, int m, int n, int k){
    // row0 1x4
    AddDot(&A(0,0),&B(0,0),&C(0,0),m,n,k);
    AddDot(&A(0,0),&B(0,1),&C(0,1),m,n,k);
    AddDot(&A(0,0),&B(0,2),&C(0,2),m,n,k);
    AddDot(&A(0,0),&B(0,3),&C(0,3),m,n,k);

    // row1 1x4
    AddDot(&A(1,0),&B(0,0),&C(1,0),m,n,k);
    AddDot(&A(1,0),&B(0,1),&C(1,1),m,n,k);
    AddDot(&A(1,0),&B(0,2),&C(1,2),m,n,k);
    AddDot(&A(1,0),&B(0,3),&C(1,3),m,n,k);

    // row2 1x4
    AddDot(&A(2,0),&B(0,0),&C(2,0),m,n,k);
    AddDot(&A(2,0),&B(0,1),&C(2,1),m,n,k);
    AddDot(&A(2,0),&B(0,2),&C(2,2),m,n,k);
    AddDot(&A(2,0),&B(0,3),&C(2,3),m,n,k);

    // row3 1x4
    AddDot(&A(3,0),&B(0,0),&C(3,0),m,n,k);
    AddDot(&A(3,0),&B(0,1),&C(3,1),m,n,k);
    AddDot(&A(3,0),&B(0,2),&C(3,2),m,n,k);
    AddDot(&A(3,0),&B(0,3),&C(3,3),m,n,k);
} 

void AddDot4x4_inline(float* A, float* B, float* C, int m, int n, int k){
    // row0 1x4
    for(int p=0;p<k;++p){
        C(0,0) += A(0,p) * B(p,0);
    }
    for(int p=0;p<k;++p){
        C(0,1) += A(0,p) * B(p,1);
    }
    for(int p=0;p<k;++p){
        C(0,2) += A(0,p) * B(p,2);
    }
    for(int p=0;p<k;++p){
        C(0,3) += A(0,p) * B(p,3);
    }

    // row1 1x4
    for(int p=0;p<k;++p){
        C(1,0) += A(1,p) * B(p,0);
    }
    for(int p=0;p<k;++p){
        C(1,1) += A(1,p) * B(p,1);
    }
    for(int p=0;p<k;++p){
        C(1,2) += A(1,p) * B(p,2);
    }
    for(int p=0;p<k;++p){
        C(1,3) += A(1,p) * B(p,3);
    }

    // row2 1x4
    for(int p=0;p<k;++p){
        C(2,0)  += A(2,p) * B(p,0);
    }
    for(int p=0;p<k;++p){
        C(2,1) += A(2,p) * B(p,1);
    }
    for(int p=0;p<k;++p){
        C(2,2) += A(2,p) * B(p,2);
    }
    for(int p=0;p<k;++p){
        C(2,3) += A(2,p) * B(p,3);
    }

    // row3 1x4
    for(int p=0;p<k;++p){
        C(3,0) += A(3,p) * B(p,0);
    }
    for(int p=0;p<k;++p){
        C(3,1) += A(3,p) * B(p,1);
    }
    for(int p=0;p<k;++p){
        C(3,2) += A(3,p) * B(p,2);
    }
    for(int p=0;p<k;++p){
        C(3,3) += A(3,p) * B(p,3);
    }
}

void AddDot4x4_fused(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    for(p=0;p<k;++p){
        // row0 1x4
        C(0,0)  += A(0,p) * B(p,0);
        C(0,1)  += A(0,p) * B(p,1);
        C(0,2)  += A(0,p) * B(p,2);
        C(0,3)  += A(0,p) * B(p,3);

        // row1 1x4
        C(1,0)  += A(1,p) * B(p,0);
        C(1,1)  += A(1,p) * B(p,1);
        C(1,2)  += A(1,p) * B(p,2);
        C(1,3)  += A(1,p) * B(p,3);
    
        // row2 1x4
        C(2,0)  += A(2,p) * B(p,0);
        C(2,1)  += A(2,p) * B(p,1);
        C(2,2)  += A(2,p) * B(p,2);
        C(2,3)  += A(2,p) * B(p,3);

        // row3 1x4
        C(3,0)  += A(3,p) * B(p,0);
        C(3,1)  += A(3,p) * B(p,1);
        C(3,2)  += A(3,p) * B(p,2);
        C(3,3)  += A(3,p) * B(p,3);
    }
}

void AddDot4x4_register(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    register float reg_c00,reg_c01,reg_c02,reg_c03, \
                    reg_c10,reg_c11,reg_c12,reg_c13, \
                    reg_c20,reg_c21,reg_c22,reg_c23, \
                    reg_c30,reg_c31,reg_c32,reg_c33;

    reg_c00 = 0, reg_c01 = 0, reg_c02 = 0, reg_c03 = 0;
    reg_c10 = 0, reg_c11 = 0, reg_c12 = 0, reg_c13 = 0;
    reg_c20 = 0, reg_c21 = 0, reg_c22 = 0, reg_c23 = 0;
    reg_c30 = 0, reg_c31 = 0, reg_c32 = 0, reg_c33 = 0;

    register float reg_a0p,reg_a1p,reg_a2p,reg_a3p;

    reg_a0p = 0, reg_a1p = 0, reg_a2p = 0, reg_a3p = 0;

    for(p=0;p<k;++p){
        reg_a0p = A(0,p);
        reg_a1p = A(1,p);
        reg_a2p = A(2,p);
        reg_a3p = A(3,p);

        // row0 1x4
        reg_c00  += reg_a0p * B(p,0);
        reg_c01  += reg_a0p * B(p,1);
        reg_c02  += reg_a0p * B(p,2);
        reg_c03  += reg_a0p * B(p,3);

        // row1 1x4
        reg_c10  += reg_a1p * B(p,0);
        reg_c11  += reg_a1p * B(p,1);
        reg_c12  += reg_a1p * B(p,2);
        reg_c13  += reg_a1p * B(p,3);
    
        // row2 1x4
        reg_c20  += reg_a2p * B(p,0);
        reg_c21  += reg_a2p * B(p,1);
        reg_c22  += reg_a2p * B(p,2);
        reg_c23  += reg_a2p * B(p,3);

        // row3 1x4
        reg_c30  += reg_a3p * B(p,0);
        reg_c31  += reg_a3p * B(p,1);
        reg_c32  += reg_a3p * B(p,2);
        reg_c33  += reg_a3p * B(p,3);
    }

    C(0,0) += reg_c00, C(0,1) += reg_c01, C(0,2) += reg_c02, C(0,3) += reg_c03;
    C(1,0) += reg_c10, C(1,1) += reg_c11, C(1,2) += reg_c12, C(1,3) += reg_c13;
    C(2,0) += reg_c20, C(2,1) += reg_c21, C(2,2) += reg_c22, C(2,3) += reg_c23;
    C(3,0) += reg_c30, C(3,1) += reg_c31, C(3,2) += reg_c32, C(3,3) += reg_c33;
}

void AddDot4x4_reduce_B_indexing(float* A, float* B, float* C, int m, int n, int k){
    register float reg_c00,reg_c01,reg_c02,reg_c03, \
                    reg_c10,reg_c11,reg_c12,reg_c13, \
                    reg_c20,reg_c21,reg_c22,reg_c23, \
                    reg_c30,reg_c31,reg_c32,reg_c33;

    reg_c00 = 0, reg_c01 = 0, reg_c02 = 0, reg_c03 = 0;
    reg_c10 = 0, reg_c11 = 0, reg_c12 = 0, reg_c13 = 0;
    reg_c20 = 0, reg_c21 = 0, reg_c22 = 0, reg_c23 = 0;
    reg_c30 = 0, reg_c31 = 0, reg_c32 = 0, reg_c33 = 0;

    register float reg_a0p,reg_a1p,reg_a2p,reg_a3p;

    reg_a0p = 0, reg_a1p = 0, reg_a2p = 0, reg_a3p = 0;

    float *ptr_bp0, *ptr_bp1, *ptr_bp2, *ptr_bp3;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);

    int p = 0;
    for(p=0;p<k;++p){
        reg_a0p = A(0,p);
        reg_a1p = A(1,p);
        reg_a2p = A(2,p);
        reg_a3p = A(3,p);

        // row0 1x4
        reg_c00  += reg_a0p * *ptr_bp0;
        reg_c01  += reg_a0p * *ptr_bp1;
        reg_c02  += reg_a0p * *ptr_bp2;
        reg_c03  += reg_a0p * *ptr_bp3;

        // row1 1x4
        reg_c10  += reg_a1p * *ptr_bp0;
        reg_c11  += reg_a1p * *ptr_bp1;
        reg_c12  += reg_a1p * *ptr_bp2;
        reg_c13  += reg_a1p * *ptr_bp3;
    
        // row2 1x4
        reg_c20  += reg_a2p * *ptr_bp0;
        reg_c21  += reg_a2p * *ptr_bp1;
        reg_c22  += reg_a2p * *ptr_bp2;
        reg_c23  += reg_a2p * *ptr_bp3;

        // row3 1x4
        reg_c30  += reg_a3p * *ptr_bp0++;
        reg_c31  += reg_a3p * *ptr_bp1++;
        reg_c32  += reg_a3p * *ptr_bp2++;
        reg_c33  += reg_a3p * *ptr_bp3++;
    }

    C(0,0) += reg_c00, C(0,1) += reg_c01, C(0,2) += reg_c02, C(0,3) += reg_c03;
    C(1,0) += reg_c10, C(1,1) += reg_c11, C(1,2) += reg_c12, C(1,3) += reg_c13;
    C(2,0) += reg_c20, C(2,1) += reg_c21, C(2,2) += reg_c22, C(2,3) += reg_c23;
    C(3,0) += reg_c30, C(3,1) += reg_c31, C(3,2) += reg_c32, C(3,3) += reg_c33;
}

void AddDot4x4_rows_rearrange(float* A, float* B, float* C, int m, int n, int k){
    register float reg_c00,reg_c01,reg_c02,reg_c03,  \
                    reg_c10,reg_c11,reg_c12,reg_c13, \
                    reg_c20,reg_c21,reg_c22,reg_c23, \
                    reg_c30,reg_c31,reg_c32,reg_c33;

    reg_c00 = 0, reg_c01 = 0, reg_c02 = 0, reg_c03 = 0;
    reg_c10 = 0, reg_c11 = 0, reg_c12 = 0, reg_c13 = 0;
    reg_c20 = 0, reg_c21 = 0, reg_c22 = 0, reg_c23 = 0;
    reg_c30 = 0, reg_c31 = 0, reg_c32 = 0, reg_c33 = 0;

    register float reg_a0p,reg_a1p,reg_a2p,reg_a3p;

    reg_a0p = 0, reg_a1p = 0, reg_a2p = 0, reg_a3p = 0;

    float *ptr_bp0, *ptr_bp1, *ptr_bp2, *ptr_bp3;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);

    int p = 0;
    for(p=0;p<k;++p){
        reg_a0p = A(0,p);
        reg_a1p = A(1,p);
        reg_a2p = A(2,p);
        reg_a3p = A(3,p);

        // row0_0/row1_0/row3_0/row4_0
        reg_c00  += reg_a0p * *ptr_bp0;
        reg_c10  += reg_a1p * *ptr_bp0;
        reg_c20  += reg_a2p * *ptr_bp0;
        reg_c30  += reg_a3p * *ptr_bp0++;

        // row0_1/row1_1/row3_1/row4_1
        reg_c01  += reg_a0p * *ptr_bp1;
        reg_c11  += reg_a1p * *ptr_bp1;
        reg_c21  += reg_a2p * *ptr_bp1;
        reg_c31  += reg_a3p * *ptr_bp1++;

        // row0_2/row1_2/row3_2/row4_2
        reg_c02  += reg_a0p * *ptr_bp2;
        reg_c12  += reg_a1p * *ptr_bp2;
        reg_c22  += reg_a2p * *ptr_bp2;
        reg_c32  += reg_a3p * *ptr_bp2++;

        // row0_3/row1_3/row3_3/row4_3
        reg_c03  += reg_a0p * *ptr_bp3;
        reg_c13  += reg_a1p * *ptr_bp3;
        reg_c23  += reg_a2p * *ptr_bp3;
        reg_c33  += reg_a3p * *ptr_bp3++;
    }

    C(0,0) = reg_c00, C(0,1) = reg_c01, C(0,2) = reg_c02, C(0,3) = reg_c03;
    C(1,0) = reg_c10, C(1,1) = reg_c11, C(1,2) = reg_c12, C(1,3) = reg_c13;
    C(2,0) = reg_c20, C(2,1) = reg_c21, C(2,2) = reg_c22, C(2,3) = reg_c23;
    C(3,0) = reg_c30, C(3,1) = reg_c31, C(3,2) = reg_c32, C(3,3) = reg_c33;
}