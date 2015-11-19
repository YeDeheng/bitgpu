#ifndef peace_CUH_
#define peace_CUH_

#include<curand.h>
#include<curand_kernel.h>
#include<thrust/device_vector.h>

typedef struct 
{
    int index;
    REAL out_err;
    float out_area;
} output_stuff;

__global__ void range(REAL *in_lo, REAL *in_hi, REAL *out_lo, REAL *out_hi, 
int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
int N_threads, int N_intervals,
REAL* pow_intervals);

__global__ void bitslice_error(REAL *temp_lo, REAL *temp_hi, 
int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
int* Na, int* Nb, unsigned long N, int* pow_bitslice, REAL* pow_error, 
float* out_area, float* out_err, REAL ERROR_THRESH);

__global__ void peace_area(REAL* out_area, int *opcode, int* dest, int INSTRUCTIONS, int Na, int Nb, unsigned long N, int* pow_bitslice);

__global__ void peace_mc_range(float *res, int *time, REAL* in_lo, REAL* in_hi, int INPUTS);

__global__ void setup_kernel (curandState* state, unsigned long idx);

__global__ void monte_carlo_error(int mc_loops, 
REAL *temp_lo, REAL *temp_hi, 
int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
int* Na, int* Nb, unsigned long N, REAL* pow_error, 
float* out_area, float* out_err, REAL ERROR_THRESH, 
curandState* globalState);

__global__ void monte_carlo_recover_sequence(
int* opcode, int* src0, int* dest, int INSTRUCTIONS, int* Na, int* Nb, 
curandState* globalState, int* bits, int thread);

#endif 

