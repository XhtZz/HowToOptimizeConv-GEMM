#include <vector>
#include <stdio.h>
#include "helpler.h"

void AddDot1x4(float* A, float* B, float* C, int m, int n, int k){
    AddDot(&A(0,0),&B(0,0),&C(0,0),m,n,k);
    AddDot(&A(0,0),&B(0,1),&C(0,1),m,n,k);
    AddDot(&A(0,0),&B(0,2),&C(0,2),m,n,k);
    AddDot(&A(0,0),&B(0,3),&C(0,3),m,n,k);
}

void AddDot1x4_inline(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    for(p=0;p<k;++p){
        C(0,0) += A(0,p) * B(p,0);
    }
    for(p=0;p<k;++p){
        C(0,1) += A(0,p) * B(p,1);
    }
    for(p=0;p<k;++p){
        C(0,2) += A(0,p) * B(p,2);
    }
    for(p=0;p<k;++p){
        C(0,3) += A(0,p) * B(p,3);
    }
}

void AddDot1x4_fused(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    for(p=0;p<k;++p){
        C(0,0) += A(0,p) * B(p,0);
        C(0,1) += A(0,p) * B(p,1);
        C(0,2) += A(0,p) * B(p,2);
        C(0,3) += A(0,p) * B(p,3);
    }
}

void AddDot1x4_register(float* A, float* B, float* C, int m, int n, int k){
    /*精度有损*/
    
    // int p = 0;
    // register float reg_a = 0;
    // float reg_c0 = 0.0;
    // for(p=0;p<k;++p){
    //     reg_a = A(0,p);
    //     reg_c0 += reg_a * B(p,0);
    //     C(0,1) += reg_a * B(p,1);
    //     C(0,2) += reg_a * B(p,2);
    //     C(0,3) += reg_a * B(p,3);
    // }
    // C(0,0) += reg_c0;

    int p = 0;
    register float reg_c0,reg_c1,reg_c2,reg_c3,reg_a;

    reg_c0 = 0.0;
    reg_c1 = 0.0;
    reg_c2 = 0.0;
    reg_c3 = 0.0;
    reg_a  = 0.0;
    
    for(p=0;p<k;++p){
        reg_a = A(0,p);
        reg_c0 += reg_a * B(p,0);
        reg_c1 += reg_a * B(p,1);
        reg_c2 += reg_a * B(p,2);
        reg_c3 += reg_a * B(p,3);
    }

    C(0,0) += reg_c0;
    C(0,1) += reg_c1;
    C(0,2) += reg_c2;
    C(0,3) += reg_c3;
}

void AddDot1x4_reduce_B_indexing(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    register float reg_c0,reg_c1,reg_c2,reg_c3,reg_a;
    reg_c0 = 0.0;
    reg_c1 = 0.0;
    reg_c2 = 0.0;
    reg_c3 = 0.0;

    float* ptr_b0 = &B(0,0);
    float* ptr_b1 = &B(0,1);
    float* ptr_b2 = &B(0,2);
    float* ptr_b3 = &B(0,3);

    for(p=0;p<k;++p){
        reg_a = A(0,p);
        reg_c0 += reg_a * *ptr_b0++;
        reg_c1 += reg_a * *ptr_b1++;
        reg_c2 += reg_a * *ptr_b2++;
        reg_c3 += reg_a * *ptr_b3++;
    }

    C(0,0) += reg_c0;
    C(0,1) += reg_c1;
    C(0,2) += reg_c2;
    C(0,3) += reg_c3;
}


void AddDot1x4_unroll_k(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    register float reg_c0,reg_c1,reg_c2,reg_c3,reg_a;
    reg_c0 = 0.0;
    reg_c1 = 0.0;
    reg_c2 = 0.0;
    reg_c3 = 0.0;

    float* ptr_b0 = &B(0,0);
    float* ptr_b1 = &B(0,1);
    float* ptr_b2 = &B(0,2);
    float* ptr_b3 = &B(0,3);

    for(p=0;p<k;p+=4){
        reg_a = A(0,p);
        reg_c0 += reg_a * *ptr_b0++;
        reg_c1 += reg_a * *ptr_b1++;
        reg_c2 += reg_a * *ptr_b2++;
        reg_c3 += reg_a * *ptr_b3++;

        reg_a = A(0,p+1);
        reg_c0 += reg_a * *ptr_b0++;
        reg_c1 += reg_a * *ptr_b1++;
        reg_c2 += reg_a * *ptr_b2++;
        reg_c3 += reg_a * *ptr_b3++;

        reg_a = A(0,p+2);
        reg_c0 += reg_a * *ptr_b0++;
        reg_c1 += reg_a * *ptr_b1++;
        reg_c2 += reg_a * *ptr_b2++;
        reg_c3 += reg_a * *ptr_b3++;

        reg_a = A(0,p+3);
        reg_c0 += reg_a * *ptr_b0++;
        reg_c1 += reg_a * *ptr_b1++;
        reg_c2 += reg_a * *ptr_b2++;
        reg_c3 += reg_a * *ptr_b3++;
    }

    C(0,0) += reg_c0;
    C(0,1) += reg_c1;
    C(0,2) += reg_c2;
    C(0,3) += reg_c3;
}

void AddDot1x4_reduce_ptrB_update(float* A, float* B, float* C, int m, int n, int k){
    int p = 0;
    register float reg_c0,reg_c1,reg_c2,reg_c3,reg_a;
    reg_c0 = 0.0;
    reg_c1 = 0.0;
    reg_c2 = 0.0;
    reg_c3 = 0.0;

    float* ptr_b0 = &B(0,0);
    float* ptr_b1 = &B(0,1);
    float* ptr_b2 = &B(0,2);
    float* ptr_b3 = &B(0,3);

    for(p=0;p<k;p+=4){
        reg_a = A(0,p);
        reg_c0 += reg_a * *ptr_b0;
        reg_c1 += reg_a * *ptr_b1;
        reg_c2 += reg_a * *ptr_b2;
        reg_c3 += reg_a * *ptr_b3;

        reg_a = A(0,p+1);
        reg_c0 += reg_a * *(ptr_b0+1);
        reg_c1 += reg_a * *(ptr_b1+1);
        reg_c2 += reg_a * *(ptr_b2+1);
        reg_c3 += reg_a * *(ptr_b3+1);

        reg_a = A(0,p+2);
        reg_c0 += reg_a * *(ptr_b0+2);
        reg_c1 += reg_a * *(ptr_b1+2);
        reg_c2 += reg_a * *(ptr_b2+2);
        reg_c3 += reg_a * *(ptr_b3+2);

        reg_a = A(0,p+3);
        reg_c0 += reg_a * *(ptr_b0+3);
        reg_c1 += reg_a * *(ptr_b1+3);
        reg_c2 += reg_a * *(ptr_b2+3);
        reg_c3 += reg_a * *(ptr_b3+3);

        ptr_b0 = ptr_b0+4;
        ptr_b1 = ptr_b1+4;
        ptr_b2 = ptr_b2+4;
        ptr_b3 = ptr_b3+4;
    }

    C(0,0) += reg_c0;
    C(0,1) += reg_c1;
    C(0,2) += reg_c2;
    C(0,3) += reg_c3;
}