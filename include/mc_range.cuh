#ifndef MC_RANGE_H_
#define MC_RANGE_H_

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define SAMPLE_NUM 	16

__forceinline__ __device__ void get_mc_samples(REAL lo, REAL hi, 
        REAL *sample_array, int sample_num) {

    curandState s;
    curand_init(sample_num, 0, 0, &s);

    double range = hi - lo;
    for(int i=0; i<sample_num-2; i++)
    {
        // curand uniform distribution from [0,1]
        sample_array[i] = curand_uniform(&s)*range + lo;
    }
    
    sample_array[sample_num - 2] = lo;
    sample_array[sample_num - 1] = hi;
}

 
__forceinline__  __device__ void sqrt_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<0 || x1<0)
        return;

    REAL sqrt_sample_array[SAMPLE_NUM];

    get_mc_samples(x0, x1, sqrt_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++)
        sqrt_sample_array[i] = sqrt(sqrt_sample_array[i]);
    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM; i++) {
        if(sqrt_sample_array[i] < sqrt_sample_array[min_pos])
            min_pos=i;

        if(sqrt_sample_array[i] > sqrt_sample_array[max_pos])
            max_pos=i;

    }

    *ret0 = sqrt_sample_array[min_pos];
    *ret1 = sqrt_sample_array[max_pos];
}

__forceinline__  __device__ void log_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<0 || x1<0)
        return;
		
    REAL log_sample_array[SAMPLE_NUM];

    get_mc_samples(x0, x1, log_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++)
        log_sample_array[i] = log(log_sample_array[i]);

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM; i++) {
        if(log_sample_array[i] < log_sample_array[min_pos])
            min_pos=i;

        if(log_sample_array[i] > log_sample_array[max_pos])
            max_pos=i;

    }

    *ret0 = log_sample_array[min_pos];
    *ret1 = log_sample_array[max_pos];
}


__forceinline__ __device__ void exp_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    REAL exp_sample_array[SAMPLE_NUM];

    get_mc_samples(x0, x1, exp_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++)
        exp_sample_array[i] = exp(exp_sample_array[i]);

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM; i++) {
        if(exp_sample_array[i] < exp_sample_array[min_pos])
            min_pos=i;

        if(exp_sample_array[i] > exp_sample_array[max_pos])
            max_pos=i;
    }

    *ret0 = exp_sample_array[min_pos];
    *ret1 = exp_sample_array[max_pos];
}

__forceinline__  __device__ void div_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {

    if(y0*y1<=0) // value of divisor could not include 0 
        return;
    
    REAL dividend_sample_array[SAMPLE_NUM];
    REAL divisor_sample_array[SAMPLE_NUM];
    REAL quotient_sample_array[SAMPLE_NUM * SAMPLE_NUM];

    get_mc_samples(x0, x1, dividend_sample_array, SAMPLE_NUM);
    get_mc_samples(y0, y1, divisor_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++) {
	for(int j = 0; j < SAMPLE_NUM; j++) {
            quotient_sample_array[i * SAMPLE_NUM + j] = dividend_sample_array[i] / divisor_sample_array[j];
        }
    }

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM*SAMPLE_NUM; i++) {
        if(quotient_sample_array[i] < quotient_sample_array[min_pos])
            min_pos=i;

        if(quotient_sample_array[i] > quotient_sample_array[max_pos])
            max_pos=i;
    }

    *ret0 = quotient_sample_array[min_pos];
    *ret1 = quotient_sample_array[max_pos];
}

__forceinline__  __device__ void mult_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL mult_sample_array1[SAMPLE_NUM];
    REAL mult_sample_array2[SAMPLE_NUM];
    REAL product_sample_array[SAMPLE_NUM * SAMPLE_NUM];

    get_mc_samples(x0, x1, mult_sample_array1, SAMPLE_NUM);
    get_mc_samples(y0, y1, mult_sample_array2, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++) {
	for(int j = 0; j < SAMPLE_NUM; j++) {
            product_sample_array[i * SAMPLE_NUM + j] = mult_sample_array1[i] * mult_sample_array2[j];
        }
    }

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM*SAMPLE_NUM; i++) {
        if(product_sample_array[i] < product_sample_array[min_pos])
            min_pos=i;

        if(product_sample_array[i] > product_sample_array[max_pos])
            max_pos=i;
    }

    *ret0 = product_sample_array[min_pos];
    *ret1 = product_sample_array[max_pos];
}

__forceinline__  __device__ void add_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL add_sample_array1[SAMPLE_NUM];
    REAL add_sample_array2[SAMPLE_NUM];
    REAL sum_sample_array[SAMPLE_NUM * SAMPLE_NUM];

    get_mc_samples(x0, x1, add_sample_array1, SAMPLE_NUM);
    get_mc_samples(y0, y1, add_sample_array2, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++) {
	for(int j = 0; j < SAMPLE_NUM; j++) {
            sum_sample_array[i * SAMPLE_NUM + j] = add_sample_array1[i] + add_sample_array2[j];
        }
    }

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM*SAMPLE_NUM; i++) {
        if(sum_sample_array[i] < sum_sample_array[min_pos])
            min_pos=i;

        if(sum_sample_array[i] > sum_sample_array[max_pos])
            max_pos=i;
    }

    *ret0 = sum_sample_array[min_pos];
    *ret1 = sum_sample_array[max_pos];
}

__forceinline__  __device__ void sub_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL min_sample_array[SAMPLE_NUM];
    REAL sub_sample_array[SAMPLE_NUM];
    REAL diff_sample_array[SAMPLE_NUM * SAMPLE_NUM];

    get_mc_samples(x0, x1, min_sample_array, SAMPLE_NUM);
    get_mc_samples(y0, y1, sub_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++) {
	for(int j = 0; j < SAMPLE_NUM; j++) {
            diff_sample_array[i * SAMPLE_NUM + j] = min_sample_array[i] - sub_sample_array[j];
        }
    }

    int max_pos=0;
    int min_pos=0;
    for(int i=0; i<SAMPLE_NUM*SAMPLE_NUM; i++) {
        if(diff_sample_array[i] < diff_sample_array[min_pos])
            min_pos=i;

        if(diff_sample_array[i] > diff_sample_array[max_pos])
            max_pos=i;
    }

    *ret0 = diff_sample_array[min_pos];
    *ret1 = diff_sample_array[max_pos];
}

__forceinline__  __device__ int integer_bit_calc(REAL low_bound, REAL high_bound)
{
    int lo = abs(low_bound);
    int hi = abs(high_bound);
    
    REAL bit_sample_array[SAMPLE_NUM];

    get_mc_samples(lo, hi, bit_sample_array, SAMPLE_NUM);
    
    for(int i = 0; i < SAMPLE_NUM; i++)
        bit_sample_array[i] = ceil(log2(bit_sample_array[i]+1));

    int max_pos=0;
    for(int i=0; i<SAMPLE_NUM; i++) {
        if(bit_sample_array[i] > bit_sample_array[max_pos])
            max_pos=i;

    }

    int integer_bit = (int)(bit_sample_array[max_pos]);

    return integer_bit;
} 
#endif
