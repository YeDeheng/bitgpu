#include <stdio.h>
#include <math.h>

#include "opcode.h"
#include "bitgpu.cuh"
#include "range.cuh"

/*
 * Propagates intervals in dataflow order 
 * Inputs: 1. custom IR encoded list of instructions in opcode, src0, src1, dest
 *         2. the intervals in_lo, in_hi
 *         3. miscellaneous metadata
 * Outputs:  intervals out_lo and out_hi
*/
__global__ void range(REAL *in_lo, REAL *in_hi, REAL *out_lo, REAL *out_hi, 
        int *opcode, int *src0, int *src1, int* dest, int INSTRUCTIONS, 
	int N_threads, int N_intervals,
	REAL* pow_intervals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ REAL temp_lo[REGISTERS];//={0}; 
    __shared__ REAL temp_hi[REGISTERS];//={0};

    if(idx<N_threads)
    {
        int inputs = 0;
        for(int j=0; j<INSTRUCTIONS; j++) {
            if(opcode[j] == LOG) {
                log_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == EXP) {
                exp_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == DIV) {
                div_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        temp_lo[src1[j]], temp_hi[src1[j]],
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == MUL) {
                mult_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        temp_lo[src1[j]], temp_hi[src1[j]],
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == ADD) {
                add_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        temp_lo[src1[j]], temp_hi[src1[j]], 
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == SUB) {
                sub_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], 
                        temp_lo[src1[j]], temp_hi[src1[j]],
                        &temp_lo[dest[j]], &temp_hi[dest[j]]);
            } else if(opcode[j] == LD) { 
                REAL in_lo_temp = in_lo[dest[j]]; 
                REAL in_hi_temp = in_hi[dest[j]]; 
                int powf_slice = (int)pow_intervals[inputs];
                int local_idx = (idx/powf_slice)%N_intervals;
                temp_lo[dest[j]] = in_lo_temp + 
                	(local_idx)*(in_hi_temp-in_lo_temp)/N_intervals;
                temp_hi[dest[j]] = in_lo_temp + 
                	(local_idx+1)*(in_hi_temp-in_lo_temp)/N_intervals;
                inputs++;
            } else if(opcode[j] == ST) {
                    temp_lo[src0[j]] = temp_lo[src0[j]];
                    temp_hi[src0[j]] = temp_hi[src0[j]];
            }
        }

        // TODO: optimize memcopy to global memory
        for(int j=0;j<INSTRUCTIONS;j++) {
            out_hi[idx*INSTRUCTIONS+j] = temp_hi[j];
            out_lo[idx*INSTRUCTIONS+j] = temp_lo[j];
        }
    } 
}
