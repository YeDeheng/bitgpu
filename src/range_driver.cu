/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <time.h>
/* Self-defined Headers */
#include "opcode.h"
#include "asm.h"
/* Host functions for pruning, device functions for range analysis */
#include "bitgpu.cuh"
#include "thrust/extrema.h"

using namespace std;

int main(int argc, char** argv)
{
	if(argc!=4) 
	{
		printf("Usage: range_driver <asm file> <interval splits> <block size>\n");
		exit(1);
	}

	// read in #lines, #intervals 
	const int N_intervals = atoi(argv[2]); 
	const int lines = line_counter(argv[1]);
	const int block_size = atoi(argv[3]);

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

	unsigned long threads = pow((float)N_intervals, stuff.INPUT_VARIABLES);
	int n_blocks = threads/block_size + (threads%block_size == 0 ? 0 : 1);

	DEVICE_RAW(out_lo, REAL, threads*stuff.INSTRUCTIONS);
	DEVICE_RAW(out_hi, REAL, threads*stuff.INSTRUCTIONS);

	HOST(pow_intervals, REAL, stuff.INSTRUCTIONS);
	create_pow_array_intervals(&hv_pow_intervals, N_intervals, stuff.INSTRUCTIONS);
	DEVICE_COPY(pow_intervals, REAL);

	cudaEvent_t start0, stop0;
	float range_time;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	cudaEventRecord(start0, 0);

	// analyze sub-intervals on the GPU while an individual sub-interval is sequentially evaluated. 
	range<<< n_blocks, block_size >>> (d_in_lo, d_in_hi, d_out_lo, d_out_hi, 
			d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, 
			threads, N_intervals,
			d_pow_intervals);

	cudaEventRecord(stop0, 0);
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&range_time, start0, stop0);
	cudaEventDestroy(start0);
	cudaEventDestroy(stop0);
	printf("range,%f,%d,%d\n", range_time, block_size, N_intervals);

	typedef thrust::device_vector<REAL>::iterator Iterator;
	HOST(lomin, REAL, stuff.INSTRUCTIONS);
	HOST(himax, REAL, stuff.INSTRUCTIONS);

	char csv[100]="";
	strcat(csv,argv[1]);
	strcat(csv,".interval.csv");
	FILE *ifs = fopen(csv,"w");
	for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		strided_range<Iterator> reg_lo(dv_out_lo.begin() + i, dv_out_lo.end(), stuff.INSTRUCTIONS);
		hv_lomin[i] = *(thrust::min_element(reg_lo.begin(), reg_lo.end()));
		strided_range<Iterator> reg_hi(dv_out_hi.begin() + i, dv_out_hi.end(), stuff.INSTRUCTIONS);
		hv_himax[i] = *(thrust::max_element(reg_hi.begin(), reg_hi.end()));
		fprintf(ifs,"%lf,%lf\n", hv_lomin[i], hv_himax[i]);
	}
	fclose(ifs);

}
