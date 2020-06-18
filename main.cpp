#include <stdio.h>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>

#include "Matrix_base.h"
#include "Optimize1x4.h"
#include "Optimize4x4.h"
#include "Optimize8x8.h"

float compare_matrices(std::vector<float>& C_,std::vector<float>& C_REF)
{
    float max_diff = 0.0, diff;
    int num = C_.size();
    for(int i=0;i<num;++i){
        diff = fabs(C_[i]-C_REF[i]);
        max_diff = ( diff > max_diff ? diff : max_diff );
    }
    return max_diff;
}

int main(){
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    int max_m = 2000, max_n = 2000, max_k = 2000;
    std::vector<float> input(max_m*max_n);
    std::vector<float> kernel(max_n*max_k);
    std::vector<float> output(max_m*max_k);
    std::generate(input.begin(), input.end(), std::ref(f32rng));
    std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
    std::generate(output.begin(), output.end(), std::ref(f32rng));

    for(int m=40;m<=2000;m+=40){
        int n=m,k=m;
        std::vector<float> A_(input.begin(),input.begin()+m*n);
        std::vector<float> B_(kernel.begin(),kernel.begin()+n*k);

        std::vector<float> C_REF(output.begin(),output.begin()+m*k);
        Matrix_base(A_.data(),B_.data(),C_REF.data(),m,n,k);

        std::vector<float> C_RESULT;
        double tictoc = 0.0,tictoc_best = 0.0;
        int iter_num = 2;
        for(int iter=0;iter<iter_num;++iter){
            std::vector<float> C_(output.begin(),output.begin()+m*k);

            /* Time your implementation */
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            /************************* base matrix multi *************************/
            // Optimize1(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize2(A_.data(),B_.data(),C_.data(),m,n,k);

            /************************* 1x4 matrix multi *************************/
            // Optimize3_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize4_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize5_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize6_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize7_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize8_1x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize9_1x4(A_.data(),B_.data(),C_.data(),m,n,k);

            /************************* 4x4 matrix multi *************************/
            // Optimize3_4x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize4_4x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize5_4x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize6_4x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize7_4x4(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize9_4x4(A_.data(),B_.data(),C_.data(),m,n,k);

            /************* 8x8 x86_avx vector operator matrix multi *************/
            // Optimize10_8x8_mul_add(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize10_8x8_fma(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize11_8x8(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize12_13_8x8(A_.data(),B_.data(),C_.data(),m,n,k);
            // Optimize14_15_8x8(A_.data(),B_.data(),C_.data(),m,n,k);

            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span_nchw = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            tictoc = time_span_nchw.count();

            if (iter==0)
                tictoc_best = tictoc;
            else
                tictoc_best = ( tictoc < tictoc_best ? tictoc : tictoc_best );

            if(iter==iter_num-1)
                C_RESULT = C_;
        }

        float max_diff = compare_matrices(C_RESULT,C_REF);

        float GFLOPS = 2.0*n*m*k*1.0e-09;
        printf("m:%d n:%d k:%d GFLOPS:%f maxdiff:%f\n",m,n,k,GFLOPS/tictoc_best,max_diff);
    }
    return 0;
}