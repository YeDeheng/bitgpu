/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <iomanip> // to ensure bc can handle the numbers generated here! ugh!
#include <cuda.h>
#include <math.h>
#include <time.h>
/* Self-defined Headers */
#include "opcode.h"
#include "peace.cuh"
#include "thrust/extrema.h"
#include "asm.h"
#include "histogram.cuh"

using namespace std;
using namespace thrust;

int main(int argc, char** argv)
{
    if(argc!=6) 
    {
        printf("Usage: bitslice_driver <asm file> <error-thresh> <step-size> <loops-per-thread> <block-size>\n");
        exit(1);
    }

    // read in #lines, #intervals 
    const int lines = line_counter(argv[1]);
    REAL ERROR_THRESH = atof(argv[2]);
    const int step_size = atoi(argv[3]); 
    const int loops_per_thread = atoi(argv[4]);
    const int block_size = atoi(argv[5]);

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

    // copy range somehow.. -- only need to compute this once!
    int max_reg = get_max_reg(hv_dest, stuff);
    HOST(lomin, REAL, stuff.INSTRUCTIONS);
    HOST(himax, REAL, stuff.INSTRUCTIONS);

    parse_intervals(argv[1], stuff.INSTRUCTIONS, &hv_lomin, &hv_himax);
    
    DEVICE_COPY(lomin, REAL);
    DEVICE_COPY(himax, REAL);

    // also need to allocate the ranges for variables
    HOST(Na, int, stuff.INSTRUCTIONS);
    HOST(Nb, int, stuff.INSTRUCTIONS);

    parse_bitrange(argv[1], stuff.INSTRUCTIONS, &hv_Na, &hv_Nb);

    DEVICE_COPY(Na, int);
    DEVICE_COPY(Nb, int);
    
    // create the search space and sizes for warps/blocks
    // ST instruction should have same precision as the last computing instruction
    // Ok the gap between Na and Nb is magically the same size...
    unsigned long search_space = 1;
    for(int j=0; j<stuff.INSTRUCTIONS; j++) {
    	search_space *= (hv_Nb[j] - hv_Na[j] + 1);
    }
    unsigned long threads = search_space/loops_per_thread;
    int n_blocks = threads/block_size + (threads%block_size == 0 ? 0 : 1);

    // pre-define pow(2,-i) to avoid wasting GPU time in each ste
    // +1 to consider worst-case pruning
    HOST(pow_error, REAL, UPPER_BOUND + 1);
    create_pow_array(&hv_pow_error, 2, UPPER_BOUND + 1);
    DEVICE_COPY(pow_error, REAL);
    
    HOST(pow_bitslice, int, stuff.INSTRUCTIONS);
    create_pow_bitslice(&hv_pow_bitslice, stuff.INSTRUCTIONS, &hv_Na, &hv_Nb);
    DEVICE_COPY(pow_bitslice, int);

    DEVICE_RAW(out_area, float, threads);
    DEVICE_RAW(out_err, float, threads);

    cudaEvent_t start1, stop1;
    float kernel_time;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    bitslice_error<<< n_blocks, block_size >>>(d_in_lo, d_in_hi, 
    		d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, 
		d_Na, d_Nb, threads, 
		d_pow_bitslice, d_pow_error, 
		d_out_area, d_out_err, ERROR_THRESH);

    cudaDeviceSynchronize();
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&kernel_time, start1, stop1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    //the bitwidth combination which has lowest computation cost
    int position = thrust::min_element(dv_out_area.begin(), dv_out_area.end()) - dv_out_area.begin();
    
    HOST(out_area, float, threads);
    HOST(out_area_int, int, threads);
    thrust::copy(dv_out_area.begin(), dv_out_area.end(), hv_out_area.begin());

    int good_sols = 0;
    for(int j=0; j<hv_out_area.size(); j++)
    {
	    if(hv_out_area[j]!=0 && hv_out_area[j]!=INT_MAX) {
	    	good_sols++;
	    	hv_out_area_int[good_sols] = ceil(hv_out_area[j] - hv_out_area[position]);
	    }
    }
    hv_out_area_int.resize(good_sols);
	//print result
    cout << "good," << ERROR_THRESH << "," << good_sols << "," << search_space << "," << (100*good_sols/(double)search_space) << endl;


    //device_vector<int> histogram_val;
    //device_vector<int> histogram_cnt;
    //sparse_histogram(hv_out_area_int, histogram_val, histogram_cnt);

    //ostream_iterator<float>(cout, ","));
    cout << hv_out_area[position] << ",";
    if(hv_out_area[position] == 0) {
	    cout << 0;
    } else {
	    cout << " fixed " << kernel_time/(double)1000;
    }
    
    HOST(bits, int, stuff.INSTRUCTIONS);
    for(int j=0; j<stuff.INSTRUCTIONS; j++)
    {
	    int CommonCode = position/hv_pow_bitslice[j];
	    int t0 = hv_Na[j] + CommonCode%(hv_Nb[j]-hv_Na[j]+1);
	    if(hv_opcode[j] != ST) {
	    	hv_bits[hv_dest[j]] = t0;
	    } else {
	    	hv_bits[hv_src0[j]] = t0;
	    }
    }
    cout << ",[";
    for(int j=0; j<stuff.INSTRUCTIONS; j++) 
    {
	cout << hv_bits[j] << "-";
    }
    cout << "]" << endl;
} 
