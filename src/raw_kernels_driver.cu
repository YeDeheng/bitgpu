/* Common Headers */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define UPPER_BOUND 25

#include "raw_kernels.cuh"
#include "asm.h"

// Goal of this driver is to run the various error, range, area kernel on the GPU
// and evaluate their speed
// 
// Generates data for Table 2 in FPGA 2016 paper
// "GPU-Accelerated High-Level Synthesis for Bitwidth Optimization of FPGA Datapaths"

using namespace std;

int main(int argc, char** argv)
{

    if(argc!=5)
    {
	printf("Usage: raw_kernels_driver <model> <operation> <threads> <block-size>\n");
	exit(1);
    }

    int model = atoi(argv[1]);
    int operation = atoi(argv[2]);
    int threads=atoi(argv[3]);
    int block_size = atoi(argv[4]);

    srand((unsigned)time(0)); 
    int t0 = (rand()%15)+1; 
    
    HOST(x0, REAL, threads);
    HOST(x1, REAL, threads);
    HOST(y0, REAL, threads);
    HOST(y1, REAL, threads);
    HOST(out0, REAL, threads);
    HOST(out1, REAL, threads);
    HOST(e1, REAL, threads);
    HOST(e2, REAL, threads);
    HOST(out_error, REAL, threads);
    HOST(out_area, REAL, threads);
    
    for(int i=0; i< threads; i++)
    {
        hv_x0[i] = 1;
        hv_x1[i] = 2;
        hv_y0[i] = 3;
        hv_y1[i] = 4;
        hv_e1[i] = pow(2, -8);
        hv_e2[i] = pow(2, -8);
    }
    DEVICE_COPY(x0, REAL);
    DEVICE_COPY(x1, REAL);
    DEVICE_COPY(y0, REAL);
    DEVICE_COPY(y1, REAL);
    DEVICE_COPY(out0, REAL);
    DEVICE_COPY(out1, REAL);
    DEVICE_COPY(e1, REAL);
    DEVICE_COPY(e2, REAL);
    DEVICE_COPY(out_error, REAL);
    DEVICE_COPY(out_area, REAL);

    // pre-define pow(2,-i) to avoid wasted GPU time in each ste
    HOST(pow_error, REAL, UPPER_BOUND+1);
    for(int i=0; i<=UPPER_BOUND; i++) {
	    hv_pow_error[i] = pow(2.0, -(1 + i));
    }
    DEVICE_COPY(pow_error, REAL);


    cudaEvent_t start, stop;
    float kernel_time;

    int n_blocks = (threads/block_size) + ((threads%block_size) ? 1 :0);
    std::cout << block_size << " " << n_blocks << std::endl;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    if( (model_type_t)model == RANGE) {

	    wrapper_range_kernel<<<n_blocks, block_size>>>((range_kernel_t)operation, 
	   	 d_x0, d_x1, d_y0, d_y1, d_out0, d_out1);

    } else if( (model_type_t)model == ERROR) {

	    wrapper_error_kernel<<<n_blocks, block_size>>>((error_kernel_t)operation, 
	    	d_x0, d_x1, d_e1, d_y0, d_y1, d_e2, t0, d_out0, d_out1, d_pow_error, d_out_error);

    } else if( (model_type_t)model == AREA) {

	    wrapper_area_kernel<<<n_blocks, block_size>>>((area_kernel_t)operation, 
	    	d_x0, d_x1, d_y0, d_y1, t0, d_out0, d_out1, d_out_area);

    } else if( (model_type_t)model == RANDOM) {

	    wrapper_curand_kernel<<<n_blocks, block_size>>>((unsigned long)time(NULL));

    } else {
	    cout << "We're idiots" << endl;
    }


    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Raw kernel time is %g ms\n", kernel_time/(double)(LIMIT*threads));

} 
