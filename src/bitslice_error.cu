/* standard headers */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string>

/* self-defined headers */
#include "error.cuh"
#include "area.cuh"
#ifdef USE_MC
#include "mc_range.cuh"
#else
#include "range.cuh"
#endif

// crazy non-sense to allow host and device to run same code...
#define REALV REAL 
#define INTV int 
#define ARRAY_ACCESS(x,j) x[j]

#include "bitslice_core.cuh"

/* cuda headers */
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

/*
 * Assigning one bitwidth combination for each thread, 
 * every thread will call function bitslice_core to process the error equations 
 * and resource model expressions to determine the cumulative error of the output variable
 * and resource costs, and save in out_err array and out_area array respectively. 
 * Treat bitwidth combinations that result in error 
 * larger than the user-supplied error constraint as invalid.
 */ 
 __global__ void bitslice_error(REAL *temp_lo, REAL *temp_hi, 
	int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
	int* Na, int* Nb, unsigned long N, int* pow_bitslice, REAL* pow_error, 
	float* out_area, float* out_err, REAL ERROR_THRESH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bits[REGISTERS] = {0};
    if(idx<N)
    {
	// create the bit-permutation that will be evaluated.. 
        for(int j=0; j<INSTRUCTIONS; j++)
        {
            int CommonCode = (int)idx/pow_bitslice[j];
            int t0 = Na[j] + CommonCode%(Nb[j]-Na[j]+1);
	    if(opcode[j] != ST) {
           	bits[dest[j]] = t0;
	    } else {
           	bits[src0[j]] = t0;
	    }
	}
	bitslice_core(temp_lo, temp_hi,
			opcode, src0, src1, dest, INSTRUCTIONS,
			pow_error, 
			&(out_area[idx]), &(out_err[idx]), bits, ERROR_THRESH);
    } 
} 

