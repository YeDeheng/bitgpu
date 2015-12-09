/* standard headers */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string>

/* self-defined headers */
#include "error.cuh"
#include "area.cuh"
#include "range.cuh"

// crazy nonsense to allow host and device to run same code...
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
 * Split the search space across the various GPU threads.
 * Each thread calculates error (1/quality) and area (cost).
 * 
 * Inputs: 1. custom IR the captures the dataflow expression in opcode, src0, src1, dest
 *         2. other metadata to guide the processing of instructions
 *         3. range information for each variable temp_lo, temp_hi
 * Outputs: out_err array, out_area arrays
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
	    // each thread idx evaluates a given combination 
	    // of bits N[i] for each instruction i 
            int CommonCode = (int)idx/pow_bitslice[j];
            int t0 = Na[j] + CommonCode%(Nb[j]-Na[j]+1);

	    // LD/ST instructions are I/Os to dataflow expression
	    if(opcode[j] != ST) {
           	bits[dest[j]] = t0;
	    } else {
           	bits[src0[j]] = t0;
	    }
	}
	// call the core on the assigned bitwidth combination for 
	// thread idx
	bitslice_core(temp_lo, temp_hi,
			opcode, src0, src1, dest, INSTRUCTIONS,
			pow_error, 
			&(out_area[idx]), &(out_err[idx]), bits, ERROR_THRESH);
    } 
} 

