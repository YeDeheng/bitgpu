/* standard headers */
#include <stdio.h>
#include <math.h>
#include <time.h>

/* cuda headers */
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

/* self-defined headers */
#include "error.cuh"
#include "area.cuh"
#include "mc_range.cuh"

// crazy non-sense to allow host and device to run same code...
#define REALV REAL 
#define INTV int 
#define ARRAY_ACCESS(x,j) x[j]

#include "bitslice_core.cuh"

__global__ void setup_kernel (curandState* state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// setup the random number in each thread very fast..
	curand_init((seed<<20)+idx, 0, 0, &(state[idx]));
	//curand_init(seed, idx, 0, &state[idx]);
}

__global__ void monte_carlo_error(int mc_loops, 
	REAL *temp_lo, REAL *temp_hi, 
	int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
	int* Na, int* Nb, unsigned long N, 
	REAL* pow_error, 
	float* out_area, float* out_err, REAL ERROR_THRESH, 
	curandState* globalState)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// get the state of random number
	curandState localState = globalState[idx];
	//curandState replayState;

	int bits[REGISTERS] = {0};
	float min_area = INT_MAX;

	if(idx<N)
	{
		// create the bitwidth combinations
		for(int j=0; j<INSTRUCTIONS; j++)
		{
			int range = Nb[j] - Na[j] + 1; // +1 is to avoid truncation error 
			// curand uniform distribution from [0,1] 
			int t0 = (int)(curand_uniform(&localState)*range) + Na[j];
			if(opcode[j] != ST) {
				bits[dest[j]] = t0;
			} else {
				bits[src0[j]] = t0;
			}
		}

		// evaluate area/error/range
		bitslice_core(temp_lo, temp_hi,
				opcode, src0, src1, dest, INSTRUCTIONS,
				pow_error, 
				&(out_area[idx]), &(out_err[idx]), bits, ERROR_THRESH);

		/* TODO: something for remembering good old mc trial
		if(out_area[idx] < min_area) {
			min_area = out_area[idx];
			// overwrite globalState...
			globalState[idx] = replayState;
		}*/
	} 
} 

__global__ void monte_carlo_recover_sequence(
		int* opcode, int* src0, int* dest, int INSTRUCTIONS, int* Na, int* Nb, 
		curandState* globalState, int* bits, int thread)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = globalState[idx];
	if(idx == thread)
	{
		for(int j=0; j<INSTRUCTIONS; j++)
		{
			int range = Nb[j] - Na[j] + 1; 
			// curand uniform distribution from [0,1] 
			int t0 = (int)( curand_uniform(&localState)*range) + Na[j];
			if(opcode[j] != ST) {
				bits[dest[j]] = t0;
			} else {
				bits[src0[j]] = t0;
			}
		}
	} 
} 

