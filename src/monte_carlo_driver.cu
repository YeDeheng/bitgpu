/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <time.h>
/* Self-defined Headers */
#include "opcode.h"
#include "peace.cuh"
#include "asm.h"

using namespace std;

int main(int argc, char** argv)
{
    if(argc!=8) 
    {
        printf("Usage: monte_carlo_driver <asm file> <err-thresh> <upper-bound> <samples> <loop_per_thread> <block-size> <seed>\n");
        exit(1);
    }

    // read in #lines, #intervals 
    const int lines = line_counter(argv[1]);
    const float ERROR_THRESH = atof(argv[2]);
    const int upper_bound_arg = atoi(argv[3]);
    const int threads = atoi(argv[4]);
    const int loops_per_thread = atoi(argv[5]);
    const int block_size = atoi(argv[6]);
    const int seed = atoi(argv[7]);

    // allocate space on CPU 
    HOST(opcode, int, lines);
    HOST(src0, int, lines);
    HOST(src1, int, lines);
    HOST(dest, int, lines);
    HOST(in_lo, REAL, lines);
    HOST(in_hi, REAL, lines);

    // parse ASM code 
    asm_stuff stuff = parse_asm(argv[1], &hv_opcode, &hv_src0, &hv_src1, &hv_in_lo, &hv_in_hi, &hv_dest);

    DEVICE_COPY(opcode, int);
    DEVICE_COPY(src0, int);
    DEVICE_COPY(src1, int);
    DEVICE_COPY(dest, int);
    DEVICE_COPY(in_lo, REAL);
    DEVICE_COPY(in_hi, REAL);

    // parse range arrays
    HOST(lomin, REAL, stuff.INSTRUCTIONS);
    HOST(himax, REAL, stuff.INSTRUCTIONS);

    parse_intervals(argv[1], stuff.INSTRUCTIONS, &hv_lomin, &hv_himax);

    DEVICE_COPY(lomin, REAL);
    DEVICE_COPY(himax, REAL);

    // parse pruned search space arrays
    HOST(Na, int, stuff.INSTRUCTIONS);
    HOST(Nb, int, stuff.INSTRUCTIONS);

    parse_bitrange(argv[1], stuff.INSTRUCTIONS, &hv_Na, &hv_Nb);
//    for(int i=0;i<stuff.INSTRUCTIONS;i++) {
//        hv_Nb[i]=32;
//    }

    DEVICE_COPY(Na, int);
    DEVICE_COPY(Nb, int);

    // avoiding pow() calls on the GPU
    // +1 to consider worst-case pruning
    HOST(pow_error, REAL, UPPER_BOUND+1);
    create_pow_array(&hv_pow_error, 2, UPPER_BOUND+1);
    DEVICE_COPY(pow_error, REAL);

    // setting up threading/parallelization constants
    int n_blocks = threads/block_size + (threads%block_size == 0 ? 0 : 1);

    HOST(devStates, curandState, threads);
    DEVICE_COPY(devStates, curandState);

    DEVICE_RAW(out_area, float, threads);
    DEVICE_RAW(out_err, float, threads);
    DEVICE_RAW(bits, int, stuff.INSTRUCTIONS);

    cudaEvent_t start1, stop1;
    float kernel_time;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    setup_kernel<<<n_blocks, block_size>>>(d_devStates, seed);
    monte_carlo_error<<< n_blocks, block_size>>>(loops_per_thread, 
    	d_in_lo, d_in_hi, 
	d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, 
	d_Na, d_Nb, threads, 
	d_pow_error, 
	d_out_area, d_out_err, ERROR_THRESH, 
	d_devStates);
    
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&kernel_time, start1, stop1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    int position = thrust::min_element( dv_out_area.begin(), dv_out_area.end() ) - dv_out_area.begin();
    thrust::copy(dv_out_area.begin() + position, 
    		dv_out_area.begin() + position + 1, 
		ostream_iterator<float>(cout, ","));

    cout << kernel_time/(double)1000;
    monte_carlo_recover_sequence<<< n_blocks, block_size>>>(
    	d_opcode, d_src0, d_dest, stuff.INSTRUCTIONS, d_Na, d_Nb, 
	d_devStates, d_bits, position);

//    cout << ",[";
//    thrust::copy(dv_bits.begin(), dv_bits.end(), ostream_iterator<int>(cout, "-"));
//    cout << "]" << endl;

    cout << ",[";
    HOST(bits, int, dv_bits.size());
    thrust::copy(dv_bits.begin(), dv_bits.end(), hv_bits.begin());
    for(int j=0; j<hv_bits.size(); j++) {
    	if(hv_opcode[j] == ST) {
    		cout << hv_bits[hv_src0[j]] << "-";
    	} else {
    		cout << hv_bits[hv_dest[j]] << "-";
    	}
    }
    cout << "]" << endl;
}
