#ifndef RANGE_H_
#define RANGE_H_

__forceinline__  __device__ void sqrt_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<0 || x1<0)
        return;
    *ret1 = sqrt(x1);
    *ret0 = sqrt(x0);
}
__forceinline__  __device__ void log_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<=0 || x1<=0)
        return;
    *ret1 = log(x1);
    *ret0 = log(x0);
}

__forceinline__ __device__ void exp_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    *ret1 = exp(x1);
    *ret0 = exp(x0);
}

__forceinline__  __device__ void div_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    if(y0*y1<=0) // value of divisor could not include 0 
        return;
    REAL t1 = x1/y1;
    REAL t2 = x1/y0;
    REAL t3 = x0/y1;
    REAL t4 = x0/y0;
    *ret1 = max(t1, max(t2, max(t3, t4)));
    *ret0 = min(t1, min(t2, min(t3, t4)));
}

__forceinline__  __device__ void mult_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL t1 = x1*y1;
    REAL t2 = x1*y0;
    REAL t3 = x0*y1;
    REAL t4 = x0*y0;

    *ret1 = max(t1, max(t2, max(t3, t4)));
    *ret0 = min(t1, min(t2, min(t3, t4)));
}

__forceinline__  __device__ void add_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL t1 = x1+y1;
    REAL t2 = x0+y0;

    *ret1 = max(t1, t2);
    *ret0 = min(t1, t2);
}

__forceinline__  __device__ void sub_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    REAL t1 = x1-y0;
    REAL t2 = x0-y1;

    *ret1 = max(t1, t2);
    *ret0 = min(t1, t2);
}

__forceinline__  __device__ int integer_bit_calc(REAL low_bound, REAL high_bound)
{
    int lo = (int)low_bound;
    int hi = (int)high_bound;

    int max_abs = (abs(lo) > abs(hi)) ? abs(lo) : abs(hi);

    int integer_bit = ceil(log2(float(max_abs + 1)));

    return integer_bit;
} 
#endif
