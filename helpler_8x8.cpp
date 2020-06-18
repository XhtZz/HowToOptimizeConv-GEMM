
#include <immintrin.h>
#include <avx2intrin.h>
#include "helpler.h"

typedef union
{
    __m256 v;
    float f[8];
} v4f_t;

void AddDot8x8_vector_operator_mul_add(float* A, float* B, float* C, int m, int n, int k){
    v4f_t   vecC_01234567_0, 
            vecC_01234567_1, 
            vecC_01234567_2, 
            vecC_01234567_3,
            vecC_01234567_4, 
            vecC_01234567_5, 
            vecC_01234567_6, 
            vecC_01234567_7;
    
    float   *ptr_bp0, 
            *ptr_bp1, 
            *ptr_bp2, 
            *ptr_bp3,
            *ptr_bp4, 
            *ptr_bp5, 
            *ptr_bp6, 
            *ptr_bp7;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);
    ptr_bp4 = &B(0,4);
    ptr_bp5 = &B(0,5);
    ptr_bp6 = &B(0,6);
    ptr_bp7 = &B(0,7);

    // vecC_01234567_0.v = _mm256_setzero_ps();
    // vecC_01234567_1.v = _mm256_setzero_ps();
    // vecC_01234567_2.v = _mm256_setzero_ps();
    // vecC_01234567_3.v = _mm256_setzero_ps();
    // vecC_01234567_4.v = _mm256_setzero_ps();
    // vecC_01234567_5.v = _mm256_setzero_ps();
    // vecC_01234567_6.v = _mm256_setzero_ps();
    // vecC_01234567_7.v = _mm256_setzero_ps();

    vecC_01234567_0.v = _mm256_loadu_ps((float*)&C(0,0));
    vecC_01234567_1.v = _mm256_loadu_ps((float*)&C(0,1));
    vecC_01234567_2.v = _mm256_loadu_ps((float*)&C(0,2));
    vecC_01234567_3.v = _mm256_loadu_ps((float*)&C(0,3));
    vecC_01234567_4.v = _mm256_loadu_ps((float*)&C(0,4));
    vecC_01234567_5.v = _mm256_loadu_ps((float*)&C(0,5));
    vecC_01234567_6.v = _mm256_loadu_ps((float*)&C(0,6));
    vecC_01234567_7.v = _mm256_loadu_ps((float*)&C(0,7));

    int p = 0;
    for(p=0;p<k;++p){
        __m256 vecA_01234567_0 = _mm256_loadu_ps((float*)&A(0,p));

        __m256 vecB_p_00000000 = _mm256_broadcast_ss((float*)ptr_bp0++);
        __m256 vecB_p_11111111 = _mm256_broadcast_ss((float*)ptr_bp1++);
        __m256 vecB_p_22222222 = _mm256_broadcast_ss((float*)ptr_bp2++);
        __m256 vecB_p_33333333 = _mm256_broadcast_ss((float*)ptr_bp3++);
        __m256 vecB_p_44444444 = _mm256_broadcast_ss((float*)ptr_bp4++);
        __m256 vecB_p_55555555 = _mm256_broadcast_ss((float*)ptr_bp5++);
        __m256 vecB_p_66666666 = _mm256_broadcast_ss((float*)ptr_bp6++);
        __m256 vecB_p_77777777 = _mm256_broadcast_ss((float*)ptr_bp7++);

        // col0/col1/col3/col4/col5/col6/col7
        vecC_01234567_0.v = _mm256_add_ps(vecC_01234567_0.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_00000000));
        vecC_01234567_1.v = _mm256_add_ps(vecC_01234567_1.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_11111111));
        vecC_01234567_2.v = _mm256_add_ps(vecC_01234567_2.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_22222222));
        vecC_01234567_3.v = _mm256_add_ps(vecC_01234567_3.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_33333333));
        vecC_01234567_4.v = _mm256_add_ps(vecC_01234567_4.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_44444444));
        vecC_01234567_5.v = _mm256_add_ps(vecC_01234567_5.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_55555555));
        vecC_01234567_6.v = _mm256_add_ps(vecC_01234567_6.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_66666666));
        vecC_01234567_7.v = _mm256_add_ps(vecC_01234567_7.v, _mm256_mul_ps(vecA_01234567_0, vecB_p_77777777));
    }

    //8x8 block
    // int i=0;
    // for(i=0;i<8;++i){
    //     C(i,0) += vecC_01234567_0.f[i];
    //     C(i,1) += vecC_01234567_1.f[i];
    //     C(i,2) += vecC_01234567_2.f[i];
    //     C(i,3) += vecC_01234567_3.f[i];
    //     C(i,4) += vecC_01234567_4.f[i];
    //     C(i,5) += vecC_01234567_5.f[i];
    //     C(i,6) += vecC_01234567_6.f[i]; 
    //     C(i,7) += vecC_01234567_7.f[i];
    // }

    //8x8 block
    _mm256_storeu_ps(&C(0,0), vecC_01234567_0.v);
    _mm256_storeu_ps(&C(0,1), vecC_01234567_1.v);
    _mm256_storeu_ps(&C(0,2), vecC_01234567_2.v);
    _mm256_storeu_ps(&C(0,3), vecC_01234567_3.v);
    _mm256_storeu_ps(&C(0,4), vecC_01234567_4.v);
    _mm256_storeu_ps(&C(0,5), vecC_01234567_5.v);
    _mm256_storeu_ps(&C(0,6), vecC_01234567_6.v);
    _mm256_storeu_ps(&C(0,7), vecC_01234567_7.v);
    
    return;
}


void AddDot8x8_vector_operator_fma(float* A, float* B, float* C, int m, int n, int k){
    v4f_t   vecC_01234567_0, 
            vecC_01234567_1, 
            vecC_01234567_2, 
            vecC_01234567_3,
            vecC_01234567_4, 
            vecC_01234567_5, 
            vecC_01234567_6, 
            vecC_01234567_7;
    
    float   *ptr_bp0, 
            *ptr_bp1, 
            *ptr_bp2, 
            *ptr_bp3,
            *ptr_bp4, 
            *ptr_bp5, 
            *ptr_bp6, 
            *ptr_bp7;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);
    ptr_bp4 = &B(0,4);
    ptr_bp5 = &B(0,5);
    ptr_bp6 = &B(0,6);
    ptr_bp7 = &B(0,7);

    vecC_01234567_0.v = _mm256_loadu_ps((float*)&C(0,0));
    vecC_01234567_1.v = _mm256_loadu_ps((float*)&C(0,1));
    vecC_01234567_2.v = _mm256_loadu_ps((float*)&C(0,2));
    vecC_01234567_3.v = _mm256_loadu_ps((float*)&C(0,3));
    vecC_01234567_4.v = _mm256_loadu_ps((float*)&C(0,4));
    vecC_01234567_5.v = _mm256_loadu_ps((float*)&C(0,5));
    vecC_01234567_6.v = _mm256_loadu_ps((float*)&C(0,6));
    vecC_01234567_7.v = _mm256_loadu_ps((float*)&C(0,7));

    int p = 0;
    for(p=0;p<k;++p){
        __m256 vec_A01234567_0 = _mm256_loadu_ps((float*)&A(0,p));

        __m256 vec_Bp_00000000 = _mm256_broadcast_ss((float*)ptr_bp0++);
        __m256 vec_Bp_11111111 = _mm256_broadcast_ss((float*)ptr_bp1++);
        __m256 vec_Bp_22222222 = _mm256_broadcast_ss((float*)ptr_bp2++);
        __m256 vec_Bp_33333333 = _mm256_broadcast_ss((float*)ptr_bp3++);
        __m256 vec_Bp_44444444 = _mm256_broadcast_ss((float*)ptr_bp4++);
        __m256 vec_Bp_55555555 = _mm256_broadcast_ss((float*)ptr_bp5++);
        __m256 vec_Bp_66666666 = _mm256_broadcast_ss((float*)ptr_bp6++);
        __m256 vec_Bp_77777777 = _mm256_broadcast_ss((float*)ptr_bp7++);

        // col0/col1/col3/col4/col5/col6/col7
        vecC_01234567_0.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_00000000, vecC_01234567_0.v);
        vecC_01234567_1.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_11111111, vecC_01234567_1.v);
        vecC_01234567_2.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_22222222, vecC_01234567_2.v);
        vecC_01234567_3.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_33333333, vecC_01234567_3.v);
        vecC_01234567_4.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_44444444, vecC_01234567_4.v);
        vecC_01234567_5.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_55555555, vecC_01234567_5.v);
        vecC_01234567_6.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_66666666, vecC_01234567_6.v);
        vecC_01234567_7.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_77777777, vecC_01234567_7.v);
    }

    //8x8 block
    _mm256_storeu_ps(&C(0,0), vecC_01234567_0.v);
    _mm256_storeu_ps(&C(0,1), vecC_01234567_1.v);
    _mm256_storeu_ps(&C(0,2), vecC_01234567_2.v);
    _mm256_storeu_ps(&C(0,3), vecC_01234567_3.v);
    _mm256_storeu_ps(&C(0,4), vecC_01234567_4.v);
    _mm256_storeu_ps(&C(0,5), vecC_01234567_5.v);
    _mm256_storeu_ps(&C(0,6), vecC_01234567_6.v);
    _mm256_storeu_ps(&C(0,7), vecC_01234567_7.v);
    
    return;
}

void AddDot8x8_block_mc_n(float* A, float* B, float* C, int pb, int m, int n, int k){
    v4f_t   vecC_01234567_0, 
            vecC_01234567_1, 
            vecC_01234567_2, 
            vecC_01234567_3,
            vecC_01234567_4, 
            vecC_01234567_5, 
            vecC_01234567_6, 
            vecC_01234567_7;
    
    float   *ptr_bp0, 
            *ptr_bp1, 
            *ptr_bp2, 
            *ptr_bp3,
            *ptr_bp4, 
            *ptr_bp5, 
            *ptr_bp6, 
            *ptr_bp7;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);
    ptr_bp4 = &B(0,4);
    ptr_bp5 = &B(0,5);
    ptr_bp6 = &B(0,6);
    ptr_bp7 = &B(0,7);

    vecC_01234567_0.v = _mm256_loadu_ps((float*)&C(0,0));
    vecC_01234567_1.v = _mm256_loadu_ps((float*)&C(0,1));
    vecC_01234567_2.v = _mm256_loadu_ps((float*)&C(0,2));
    vecC_01234567_3.v = _mm256_loadu_ps((float*)&C(0,3));
    vecC_01234567_4.v = _mm256_loadu_ps((float*)&C(0,4));
    vecC_01234567_5.v = _mm256_loadu_ps((float*)&C(0,5));
    vecC_01234567_6.v = _mm256_loadu_ps((float*)&C(0,6));
    vecC_01234567_7.v = _mm256_loadu_ps((float*)&C(0,7));

    int p = 0;
    for(p=0;p<pb;++p){
        __m256 vec_A01234567_0 = _mm256_loadu_ps((float*)&A(0,p));

        __m256 vec_Bp_00000000 = _mm256_broadcast_ss((float*)ptr_bp0++);
        __m256 vec_Bp_11111111 = _mm256_broadcast_ss((float*)ptr_bp1++);
        __m256 vec_Bp_22222222 = _mm256_broadcast_ss((float*)ptr_bp2++);
        __m256 vec_Bp_33333333 = _mm256_broadcast_ss((float*)ptr_bp3++);
        __m256 vec_Bp_44444444 = _mm256_broadcast_ss((float*)ptr_bp4++);
        __m256 vec_Bp_55555555 = _mm256_broadcast_ss((float*)ptr_bp5++);
        __m256 vec_Bp_66666666 = _mm256_broadcast_ss((float*)ptr_bp6++);
        __m256 vec_Bp_77777777 = _mm256_broadcast_ss((float*)ptr_bp7++);

        // col0/col1/col3/col4/col5/col6/col7
        vecC_01234567_0.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_00000000, vecC_01234567_0.v);
        vecC_01234567_1.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_11111111, vecC_01234567_1.v);
        vecC_01234567_2.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_22222222, vecC_01234567_2.v);
        vecC_01234567_3.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_33333333, vecC_01234567_3.v);
        vecC_01234567_4.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_44444444, vecC_01234567_4.v);
        vecC_01234567_5.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_55555555, vecC_01234567_5.v);
        vecC_01234567_6.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_66666666, vecC_01234567_6.v);
        vecC_01234567_7.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_77777777, vecC_01234567_7.v);
    }

    //8x8 block
    _mm256_storeu_ps(&C(0,0), vecC_01234567_0.v);
    _mm256_storeu_ps(&C(0,1), vecC_01234567_1.v);
    _mm256_storeu_ps(&C(0,2), vecC_01234567_2.v);
    _mm256_storeu_ps(&C(0,3), vecC_01234567_3.v);
    _mm256_storeu_ps(&C(0,4), vecC_01234567_4.v);
    _mm256_storeu_ps(&C(0,5), vecC_01234567_5.v);
    _mm256_storeu_ps(&C(0,6), vecC_01234567_6.v);
    _mm256_storeu_ps(&C(0,7), vecC_01234567_7.v);
    
    return;
}


void AddDot8x8_packed_A(float* a, float* B, float* C, int pb, int m, int n, int k){
    v4f_t   vecC_01234567_0, 
            vecC_01234567_1, 
            vecC_01234567_2, 
            vecC_01234567_3,
            vecC_01234567_4, 
            vecC_01234567_5, 
            vecC_01234567_6, 
            vecC_01234567_7;
    
    float   *ptr_bp0, 
            *ptr_bp1, 
            *ptr_bp2, 
            *ptr_bp3,
            *ptr_bp4, 
            *ptr_bp5, 
            *ptr_bp6, 
            *ptr_bp7;

    ptr_bp0 = &B(0,0);
    ptr_bp1 = &B(0,1);
    ptr_bp2 = &B(0,2);
    ptr_bp3 = &B(0,3);
    ptr_bp4 = &B(0,4);
    ptr_bp5 = &B(0,5);
    ptr_bp6 = &B(0,6);
    ptr_bp7 = &B(0,7);

    vecC_01234567_0.v = _mm256_loadu_ps((float*)&C(0,0));
    vecC_01234567_1.v = _mm256_loadu_ps((float*)&C(0,1));
    vecC_01234567_2.v = _mm256_loadu_ps((float*)&C(0,2));
    vecC_01234567_3.v = _mm256_loadu_ps((float*)&C(0,3));
    vecC_01234567_4.v = _mm256_loadu_ps((float*)&C(0,4));
    vecC_01234567_5.v = _mm256_loadu_ps((float*)&C(0,5));
    vecC_01234567_6.v = _mm256_loadu_ps((float*)&C(0,6));
    vecC_01234567_7.v = _mm256_loadu_ps((float*)&C(0,7));

    int p = 0;
    for(p=0;p<pb;++p){
        __m256 vec_A01234567_0 = _mm256_loadu_ps((float*)a);
        a+=8;

        __m256 vec_Bp_00000000 = _mm256_broadcast_ss((float*)ptr_bp0++);
        __m256 vec_Bp_11111111 = _mm256_broadcast_ss((float*)ptr_bp1++);
        __m256 vec_Bp_22222222 = _mm256_broadcast_ss((float*)ptr_bp2++);
        __m256 vec_Bp_33333333 = _mm256_broadcast_ss((float*)ptr_bp3++);
        __m256 vec_Bp_44444444 = _mm256_broadcast_ss((float*)ptr_bp4++);
        __m256 vec_Bp_55555555 = _mm256_broadcast_ss((float*)ptr_bp5++);
        __m256 vec_Bp_66666666 = _mm256_broadcast_ss((float*)ptr_bp6++);
        __m256 vec_Bp_77777777 = _mm256_broadcast_ss((float*)ptr_bp7++);

        // col0/col1/col3/col4/col5/col6/col7
        vecC_01234567_0.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_00000000, vecC_01234567_0.v);
        vecC_01234567_1.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_11111111, vecC_01234567_1.v);
        vecC_01234567_2.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_22222222, vecC_01234567_2.v);
        vecC_01234567_3.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_33333333, vecC_01234567_3.v);
        vecC_01234567_4.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_44444444, vecC_01234567_4.v);
        vecC_01234567_5.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_55555555, vecC_01234567_5.v);
        vecC_01234567_6.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_66666666, vecC_01234567_6.v);
        vecC_01234567_7.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_77777777, vecC_01234567_7.v);
    }

    //8x8 block
    _mm256_storeu_ps(&C(0,0), vecC_01234567_0.v);
    _mm256_storeu_ps(&C(0,1), vecC_01234567_1.v);
    _mm256_storeu_ps(&C(0,2), vecC_01234567_2.v);
    _mm256_storeu_ps(&C(0,3), vecC_01234567_3.v);
    _mm256_storeu_ps(&C(0,4), vecC_01234567_4.v);
    _mm256_storeu_ps(&C(0,5), vecC_01234567_5.v);
    _mm256_storeu_ps(&C(0,6), vecC_01234567_6.v);
    _mm256_storeu_ps(&C(0,7), vecC_01234567_7.v);
    
    return;
}


void AddDot8x8_packed_AB(float* a, float* b, float* C, int pb, int m, int n, int k){
    v4f_t   vecC_01234567_0, 
            vecC_01234567_1, 
            vecC_01234567_2, 
            vecC_01234567_3,
            vecC_01234567_4, 
            vecC_01234567_5, 
            vecC_01234567_6, 
            vecC_01234567_7;
    
    float   *ptr_bp0, 
            *ptr_bp1, 
            *ptr_bp2, 
            *ptr_bp3,
            *ptr_bp4, 
            *ptr_bp5, 
            *ptr_bp6, 
            *ptr_bp7;

    vecC_01234567_0.v = _mm256_loadu_ps((float*)&C(0,0));
    vecC_01234567_1.v = _mm256_loadu_ps((float*)&C(0,1));
    vecC_01234567_2.v = _mm256_loadu_ps((float*)&C(0,2));
    vecC_01234567_3.v = _mm256_loadu_ps((float*)&C(0,3));
    vecC_01234567_4.v = _mm256_loadu_ps((float*)&C(0,4));
    vecC_01234567_5.v = _mm256_loadu_ps((float*)&C(0,5));
    vecC_01234567_6.v = _mm256_loadu_ps((float*)&C(0,6));
    vecC_01234567_7.v = _mm256_loadu_ps((float*)&C(0,7));

    int p = 0;
    for(p=0;p<pb;++p){
        __m256 vec_A01234567_0 = _mm256_loadu_ps((float*)a);
        a+=8;

        __m256 vec_Bp_00000000 = _mm256_broadcast_ss((float*)b);
        __m256 vec_Bp_11111111 = _mm256_broadcast_ss((float*)(b+1));
        __m256 vec_Bp_22222222 = _mm256_broadcast_ss((float*)(b+2));
        __m256 vec_Bp_33333333 = _mm256_broadcast_ss((float*)(b+3));
        __m256 vec_Bp_44444444 = _mm256_broadcast_ss((float*)(b+4));
        __m256 vec_Bp_55555555 = _mm256_broadcast_ss((float*)(b+5));
        __m256 vec_Bp_66666666 = _mm256_broadcast_ss((float*)(b+6));
        __m256 vec_Bp_77777777 = _mm256_broadcast_ss((float*)(b+7));
        b+=8;

        // col0/col1/col3/col4/col5/col6/col7
        vecC_01234567_0.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_00000000, vecC_01234567_0.v);
        vecC_01234567_1.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_11111111, vecC_01234567_1.v);
        vecC_01234567_2.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_22222222, vecC_01234567_2.v);
        vecC_01234567_3.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_33333333, vecC_01234567_3.v);
        vecC_01234567_4.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_44444444, vecC_01234567_4.v);
        vecC_01234567_5.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_55555555, vecC_01234567_5.v);
        vecC_01234567_6.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_66666666, vecC_01234567_6.v);
        vecC_01234567_7.v = _mm256_fmadd_ps(vec_A01234567_0, vec_Bp_77777777, vecC_01234567_7.v);
    }

    //8x8 block
    _mm256_storeu_ps(&C(0,0), vecC_01234567_0.v);
    _mm256_storeu_ps(&C(0,1), vecC_01234567_1.v);
    _mm256_storeu_ps(&C(0,2), vecC_01234567_2.v);
    _mm256_storeu_ps(&C(0,3), vecC_01234567_3.v);
    _mm256_storeu_ps(&C(0,4), vecC_01234567_4.v);
    _mm256_storeu_ps(&C(0,5), vecC_01234567_5.v);
    _mm256_storeu_ps(&C(0,6), vecC_01234567_6.v);
    _mm256_storeu_ps(&C(0,7), vecC_01234567_7.v);
    
    return;
}



