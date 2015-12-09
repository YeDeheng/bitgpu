#ifndef bitgpu_CUH_
#define bitgpu_CUH_

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

__global__ void bitgpu_error(REAL *temp_lo, REAL *temp_hi, 
int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
int* Na, int* Nb, unsigned long N, int* pow_bitslice, REAL* pow_error, 
float* out_area, float* out_err, REAL ERROR_THRESH);

__global__ void bitgpu_area(REAL* out_area, int *opcode, int* dest, int INSTRUCTIONS, int Na, int Nb, unsigned long N, int* pow_bitslice);

__global__ void setup_kernel (curandState* state, unsigned long idx);

#endif 

