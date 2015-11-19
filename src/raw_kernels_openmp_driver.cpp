/* Common Headers */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include <time.h>
#include <sys/time.h>

#include <omp.h>

#include "opcode.h"
#include "raw_kernels.h"
#include "asm.h"
#include "helper.h"

using namespace std;
using namespace thrust;

int main(int argc, char** argv)
{

	if(argc!=5)
	{
		printf("Usage: raw_kernels_openmp_driver <model> <operation> <loop-count> <threads>\n");
		exit(1);
	}

	int model = atoi(argv[1]);
	int operation = atoi(argv[2]);
	int loop_count= atoi(argv[3]);
	int threads = atoi(argv[4]);

	srand((unsigned)time(0)); 
	int t0 = (rand()%15)+1; 

	HOST(x0, REAL, loop_count);
	HOST(x1, REAL, loop_count);
	HOST(y0, REAL, loop_count);
	HOST(y1, REAL, loop_count);
	HOST(out0, REAL, loop_count);
	HOST(out1, REAL, loop_count);
	HOST(e1, REAL, loop_count);
	HOST(e2, REAL, loop_count);
	HOST(out_error, REAL, loop_count);
	HOST(out_area, REAL, loop_count);

	for(int i=0; i<loop_count; i++)
	{
		hv_x0[i] = 1;
		hv_x1[i] = 2;
		hv_y0[i] = 3;
		hv_y1[i] = 4;
		hv_e1[i] = pow(2, -8);
		hv_e2[i] = pow(2, -8);
	}

	// pre-define pow(2,-i) to avoid wasted GPU time in each ste
	HOST(pow_error, REAL, UPPER_BOUND+1);
	for(int i=0; i<=UPPER_BOUND; i++) {
		hv_pow_error[i] = pow(2.0, -(1 + i));
	}

	omp_set_num_threads(threads);

	TIMER t_start, t_end;
	record_time(&t_start);

	if((model_type_t)model == RANGE) {
#pragma omp parallel for //shared(threads) private(i)
		for(long int i=0; i<loop_count; i++) {
			wrapper_range_kernel((range_kernel_t)operation, 
					raw_pointer_cast(&(hv_x0[0])), raw_pointer_cast(&(hv_x1[0])), raw_pointer_cast(&(hv_y0[0])), raw_pointer_cast(&(hv_y1[0])), 
					raw_pointer_cast(&(hv_out0[0])), raw_pointer_cast(&(hv_out1[0])), i);
		}
	} else if((model_type_t)model == ERROR) {
#pragma omp parallel for //shared(threads) private(i)
		for(long int i=0; i < loop_count; i++) {
			wrapper_error_kernel((error_kernel_t)operation,
					raw_pointer_cast(&(hv_x0[0])), raw_pointer_cast(&(hv_x1[0])), raw_pointer_cast(&(hv_e1[0])), 
					raw_pointer_cast(&(hv_y0[0])), raw_pointer_cast(&(hv_y1[0])), raw_pointer_cast(&(hv_e2[0])), 
					raw_pointer_cast(&(hv_out0[0])), raw_pointer_cast(&(hv_out1[0])), raw_pointer_cast(&(hv_out_error[0])), 
					raw_pointer_cast(&(hv_pow_error[0])), t0, i);
		}
	} else if((model_type_t)model == AREA) {
#pragma omp parallel for //shared(threads) private(i)
		for(long int i=0; i < loop_count; i++) {
			wrapper_area_kernel((area_kernel_t)operation,
					raw_pointer_cast(&(hv_x0[0])), raw_pointer_cast(&(hv_x1[0])), raw_pointer_cast(&(hv_y0[0])), raw_pointer_cast(&(hv_y1[0])), 
					raw_pointer_cast(&(hv_out0[0])), raw_pointer_cast(&(hv_out1[0])), raw_pointer_cast(&(hv_out_area[0])), t0, i);
		}
	} else if((model_type_t)model == RANDOM) {
#pragma omp parallel for //shared(threads) private(i)
		for(long int i=0; i < loop_count; i++) {
			wrapper_rand_kernel(i, i); // seed=i, iteration=i
		}
	} else {
		cout << "We're idiots" << endl;
	}
	record_time(&t_end);
	float time_in_s = calculate_time(&t_start, &t_end);
	
	cout << "Raw kernel time is " << (1e3*time_in_s/(double)loop_count) << " ms" << endl; 

} 
