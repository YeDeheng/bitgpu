#include<iostream>
#include<stdio.h>
#include<math.h>

#include "opcode.h"
#include<thrust/device_vector.h>
#include "range_host.h"

#ifndef ERROR_H2_
#define ERROR_H2_

// No need to create a copy of the original error equations!
#define __device__ 
#include "error.cuh"

/*
REAL fx_error(thrust::host_vector<REAL> temp_lo, thrust::host_vector<REAL> temp_hi, 
        thrust::host_vector<int> opcode, thrust::host_vector<int> src0, thrust::host_vector<int> src1, thrust::host_vector<int> dest, int& INSTRUCTIONS, thrust::host_vector<int> x)
{
    REAL out_err = 0; 
    REAL error[REGISTERS] = {0};
    int t0;

    for(int j=0; j < INSTRUCTIONS; j++) 
    {
        t0 = x[j];
	REAL e3 = pow(2, -(t0+1));
        
        if(opcode[j] == LOG) 
        {
            log_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            log_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]], e3, &error[dest[j]]);
        }
        else if(opcode[j] == EXP) 
        {
            exp_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            exp_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]], e3, &error[dest[j]]);
        } 
        else if(opcode[j] == DIV) 
        {
            div_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            div_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], error[src1[j]], e3, &error[dest[j]]);
        }
        else if(opcode[j] == MUL) 
        {
            mult_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            mult_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]],temp_lo[src1[j]], temp_hi[src1[j]], error[src1[j]], e3, &error[dest[j]]);
        }
        else if(opcode[j] == ADD) 
        {
            add_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            add_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], error[src1[j]], e3, &error[dest[j]]);
        } 
        else if(opcode[j] == SUB) 
        {
            sub_rangerule(temp_lo[src0[j]], temp_hi[src0[j]], temp_lo[src1[j]], temp_hi[src1[j]], &temp_lo[dest[j]], &temp_hi[dest[j]]);
            sub_errrule(temp_lo[src0[j]], temp_hi[src0[j]], error[src0[j]],temp_lo[src1[j]], temp_hi[src1[j]], error[src1[j]], e3, &error[dest[j]]);
        } 
        else if(opcode[j] == LD) 
        {
            REAL in_hi_temp = temp_hi[dest[j]];
            REAL in_lo_temp = temp_lo[dest[j]];
            if(fabs(in_hi_temp-in_lo_temp) < 1e-30 ) 
            {
                REAL shift = in_lo_temp*pow(2.0, t0);
                double intpart;
                REAL fractpart = (REAL)modf((double)shift, &intpart);
                if( fractpart > 1e-30 )
                    error[dest[j]] = pow(2.0, -(t0+1));
                else 
                    error[dest[j]] = 0;
            } 
            else 
                error[dest[j]] = pow(2.0, -(t0+1));
        }
        else if(opcode[j] == ST)
        {
            out_err = error[src0[j]];
        }
    }
    return out_err;
} 
*/

#endif
