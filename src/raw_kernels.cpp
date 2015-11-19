#include <stdio.h>
#include <math.h>

#include "raw_kernels.h"
#include "error_host.h"
#include "range_host.h"
#include "area_host.h"

// wrapper for the range kernel
void wrapper_range_kernel(range_kernel_t type, REAL* in0, REAL* in1, REAL* in2, REAL* in3, REAL* out0, REAL* out1, int tid) {
	REAL d_out0;
	REAL d_out1;
	for(int i=0;i<LIMIT;i++) {
		if(type==ADD_RANGE) {
			add_rangerule(in0[tid],in1[tid],in2[tid],in3[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		}
		else if(type==SUB_RANGE) {
			sub_rangerule(in0[tid],in1[tid],in2[tid],in3[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		}
		else if(type==MUL_RANGE) {
			mult_rangerule(in0[tid],in1[tid],in2[tid],in3[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		}
		else if(type==DIV_RANGE) {
			div_rangerule(in0[tid],in1[tid],in2[tid],in3[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		}
		else if(type==EXP_RANGE) {
			exp_rangerule(in0[tid],in1[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		}
		else if(type==LOG_RANGE) {
			log_rangerule(in0[tid],in1[tid],&d_out0,&d_out1);
			out0[tid]=d_out0;
			out1[tid]=d_out1;
		} else {
			printf("Fucking stupid\n");
		}
	}
}

// wrapper for the error kernel
void wrapper_error_kernel(error_kernel_t type, REAL* x0, REAL* x1, REAL* e1, REAL* y0, REAL* y1, REAL* e2, REAL* out0, REAL* out1, REAL* out_error, REAL* pow_error, int t0, int tid) {
	REAL d_out0 = 2;
	REAL d_out1 = 1;
	REAL d_error = 0.1;
	for(int i=0;i<LIMIT;i++) {
		if(type==ADD_ERROR) {
			add_errrule(x0[tid], x1[tid], e1[tid], y0[tid], y1[tid], e2[tid], pow_error[t0], &d_error);
			out_error[tid]=d_error;
		}
		else if(type==SUB_ERROR) {
			sub_errrule(x0[tid], x1[tid], e1[tid], y0[tid], y1[tid], e2[tid], pow_error[t0],&d_error);
			out_error[tid]=d_error;
		}
		else if(type==MUL_ERROR) {
			mult_errrule(x0[tid], x1[tid], e1[tid], y0[tid], y1[tid], e2[tid], pow_error[t0], &d_error);
			out_error[tid]=d_error;
		}
		else if(type==DIV_ERROR) {
			div_errrule(x0[tid], x1[tid], e1[tid], y0[tid], y1[tid], e2[tid], pow_error[t0], &d_error);
			out_error[tid]=d_error;
		}
		else if(type==EXP_ERROR) {
			exp_errrule(x0[tid], x1[tid], e1[tid], pow_error[t0], &d_error);
			out_error[tid]=d_error;
		}
		else if(type==LOG_ERROR) {
			log_errrule(x0[tid], x1[tid], e1[tid], pow_error[t0], &d_error);
			out_error[tid]=d_error;
		} else {
			printf("Fucking stupid\n");
		} 
	} 
}

// wrapper for the area kernel
void wrapper_area_kernel(area_kernel_t type, REAL* x0, REAL* x1, REAL* y0, REAL* y1, REAL* out0, REAL* out1, REAL* out_area, int t0, int tid) {
	REAL d_out0 = 2;
	REAL d_out1 = 1;

	REAL area_total=0.1;

	int integer_bit_src0;
	int integer_bit_src1;

	for(int i=0;i<LIMIT;i++) { 
		if(type==ADD_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			integer_bit_src1 = integer_bit_calc(d_out0, d_out1);
			area_total += add_area(integer_bit_src0 + t0, integer_bit_src1 + t0);
		}
		else if(type==SUB_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			integer_bit_src1 = integer_bit_calc(d_out0, d_out1);
			area_total += add_area(integer_bit_src0 + t0, integer_bit_src1 + t0);
		}
		else if(type==MUL_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			integer_bit_src1 = integer_bit_calc(d_out0, d_out1);
			area_total += mult_area(integer_bit_src0 + t0, integer_bit_src1 + t0);
		}
		else if(type==DIV_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			integer_bit_src1 = integer_bit_calc(d_out0, d_out1);
			area_total += div_area(max(integer_bit_src0 + t0, integer_bit_src1 + t0));
		}
		else if(type==EXP_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			area_total += exp_area(integer_bit_src0 + t0);
		}
		else if(type==LOG_AREA) {
			integer_bit_src0 = integer_bit_calc(d_out0, d_out1);
			area_total += log_area(integer_bit_src0 + t0);
		} 
	}
}

void wrapper_rand_kernel(unsigned long seed, int tid) {
//   curandState localState;
//   curand_init( (seed<<20)+tid, 0, 0, &localState );
//   int t0 = curand_uniform(&localState);
//   __syncthreads();
}
