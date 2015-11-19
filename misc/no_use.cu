/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <time.h>
/* Thrust Headers */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/count.h>
/* Self-defined Headers */
#include "opcode.h"
#include "peace.cuh"
#include "error.h"

/* Thrust defines */
#define HOST(x,y) thrust::host_vector<y> hv_##x;
#define HOST0(x,y,size) thrust::host_vector<y> hv_##x(size);
#define DEVICE_RAW0(x,y,size) thrust::device_vector<y> dv_##x(size);
#define DEVICE_COPY0(x,y) thrust::device_vector<y> dv_##x = hv_##x;
#define CAST(x,y)  y* d_##x = thrust::raw_pointer_cast(&dv_##x[0]);
#define DEVICE_RAW(x,y,size)\
	DEVICE_RAW0(x,y,size)\
CAST(x,y)
#define DEVICE_COPY(x,y)\
	DEVICE_COPY0(x,y)\
CAST(x,y)
#define PRINT_THRUST_VECTOR(x,y)\
	std::cout << "dv_" #x ": { ";\
thrust::copy(dv_##x.begin(), dv_##x.end(), std::ostream_iterator<y>(std::cout, ", "));\
std::cout << "}" << std::endl;

/* Error threshold: targeting 8 frac-bits */
#define ERROR_THRESH pow(2,-9)

/* Thrust conditional operator */
struct is_smaller_than_thresh
{
	__host__ __device__
		bool operator()(const output_stuff a) const
		{
			return (a.out_err < ERROR_THRESH);
		}
};

template <typename Iterator>
class strided_range
{
	public:
		typedef typename thrust::iterator_difference<Iterator>::type difference_type;
		struct stride_functor : public thrust::unary_function<difference_type,difference_type>
	{
		difference_type stride;
		stride_functor(difference_type stride)
			: stride(stride) {}
		__host__ __device__
			difference_type operator()(const difference_type& i) const
			{ 
				return stride * i;
			}
	};
		typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
		typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
		typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;
		// type of the strided_range iterator
		typedef PermutationIterator iterator;
		// construct strided_range for the range [first,last)
		strided_range(Iterator first, Iterator last, difference_type stride)
			: first(first), last(last), stride(stride) {}
		iterator begin(void) const
		{
			return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
		}

		iterator end(void) const
		{
			return begin() + ((last - first) + (stride - 1)) / stride;
		}

	protected:
		Iterator first;
		Iterator last;
		difference_type stride;
};

typedef struct {
	int INSTRUCTIONS;
	int INPUTS;
	int INPUT_VARIABLES;
	int INPUT_CONSTANTS;
	int OUTPUTS;
	int ST_ADDR;
} asm_stuff;

asm_stuff parse_asm(FILE *fp, thrust::host_vector<int> *opcode, thrust::host_vector<int> *src0, thrust::host_vector<int> *src1, thrust::host_vector<double> *in_lo, thrust::host_vector<double> *in_hi, thrust::host_vector<int> *dest) 
{
	if(!fp) {
		exit(1);
	}
        fseek(fp,0,SEEK_SET);

	int i=0;
	int op=0, de=0;
	int loads=0, stores=0, variables=0, constants=0;
	asm_stuff stuff;
	while (!feof(fp)) {
		fscanf(fp,"%d,%d,",&((*opcode)[i]),&((*dest)[i]));
		//fscanf(fp,"%d,%d,",&op,&de);
		//opcode->push_back(op);
		//dest->push_back(de);
		if((*opcode)[i]==ST) 
			stores++;
		if((*opcode)[i]==LD) 
        {
			loads++;
			fscanf(fp,"%lf,%lf;\n",&((*in_lo)[(*dest)[i]]), &((*in_hi)[(*dest)[i]]));
			if((*in_lo)[(*dest)[i]] == (*in_hi)[(*dest)[i]]) 
				constants++;
			else 
				variables++;
			//printf("Read Range[%d]: %d,%d,%g,%g\n",i,(*opcode)[i],(*dest)[i],(*in_lo)[i],(*in_hi)[i]);
			(*src0)[i]=-1;
			(*src1)[i]=-1;
		} 
        else 
        {
			fscanf(fp,"%d,%d;\n",&((*src0)[i]),&((*src1)[i]));
			if((*opcode)[i]==ST)
				stuff.ST_ADDR = (*src0)[i];
		}
		i++;
	}
	//fclose(fp);
	stuff.INSTRUCTIONS=i;
	stuff.INPUTS=loads;
	stuff.INPUT_VARIABLES=variables;
	stuff.INPUT_CONSTANTS=constants;
	stuff.OUTPUTS=stores;
	return stuff;
}

typedef struct {
	int opcode;
	int dest;
	int src0;
	int src1;
} instr_format;

void cluster(asm_stuff stuff, thrust::host_vector<int> *opcode, thrust::host_vector<int> *dest,  thrust::host_vector<int> *src0, thrust::host_vector<int> *src1)
{
	int cluster_cnt = stuff.OUTPUTS;
	int j = -1;
	instr_format instr_temp;
	std::vector<instr_format> stuff_atomic[cluster_cnt];
	for(int i=stuff.INSTRUCTIONS-1; i>=0 && (*opcode)[i]!=LD; i--) //initialize the stack
	{
		if( (*opcode)[i] == ST )
		{
			j++;
			instr_temp = (instr_format) { (*opcode)[i], (*dest)[i], (*src0)[i], (*src1)[i] };
			stuff_atomic[j].push_back(instr_temp);
		}
		else
		{
			instr_temp = (instr_format) { (*opcode)[i], (*dest)[i], (*src0)[i], (*src1)[i] };
			stuff_atomic[j].push_back(instr_temp);
		}
	}
	for(int i=0; i<cluster_cnt; i++)
	{
		std::vector<int> source;
		for(int j=0; j<stuff_atomic[i].size(); j++) 
		{
			source.push_back(stuff_atomic[i][j].src0);
			source.push_back(stuff_atomic[i][j].src1);
		}
		source.erase(std::remove(source.begin(), source.end(), -1), source.end()); // remove src=-1, which is due to ST instruction
		//        for(std::vector<int>::iterator iter=source.begin(); iter!=source.end(); iter++)
		//            std::cout << " " << *iter << std::endl;
		for(int j=0; j<stuff_atomic[i].size(); j++) 
			source.erase(std::remove(source.begin(), source.end(), stuff_atomic[i][j].dest), source.end());
		//        for(std::vector<int>::iterator iter=source.begin(); iter!=source.end(); iter++)
		//           std::cout << " " << *iter << std::endl;
		for(int m=0; m<source.size(); m++)
		{
			int n = source[m];
			instr_temp = (instr_format) { (*opcode)[n], (*dest)[n], (*src0)[n], (*src1)[n]};
			stuff_atomic[i].push_back(instr_temp);
		}
	}
	//for(std::vector<instr_format>::iterator iter=stuff_atomic[0].begin(); iter!=stuff_atomic[0].end(); iter++)
	//  std::cout << "hello " << iter->opcode << " " << iter->dest << " " << iter->src0 << " " << iter->src1 << std::endl;  
}

int count_lines(FILE * fp) {
	int lines = 0;
	char line_temp[128] = "";
        if(!fp)
                exit(1);
	fseek(fp,0,SEEK_SET);
	while(!feof(fp)) { 
		fgets(line_temp, 128, fp); lines++;
	}

    return lines;
}


int main(int argc, char** argv)
{
	if(argc!=3) {
		printf("Usage: peace_test <interval splits> <asm file>\n");
		exit(1);
	}

	const int N_intervals = atoi(argv[1]); 

	// Do calculation on Device
	//int block_size = (N<32)?N:32;
	//int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
#ifdef PERF
	cudaEvent_t start, stop;
	float copy_time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	FILE *fp;
        fp = fopen(argv[2],"r");
        if(!fp)
        {
                printf("%s does not exist\n", argv[2]);
                exit(1);
        }
	int max_instruction_lines = count_lines(fp);
	// Deheng asks: can we declare THRUST_VECTOR without specifying SIZE?
	/*HOST(opcode,int, stuff.INSTRUCTIONS);
	    HOST(src0,int, stuff.INSTRUCTIONS);
	    HOST(src1,int, stuff.INSTRUCTIONS);
	    HOST(dest,int, stuff.INSTRUCTIONS);
	//HOST(index,int, MAX_INSTRUCTIONS);
	HOST(in_lo,double, MAX_INPUTS);
	HOST(in_hi,double, MAX_INPUTS);*/
	if(max_instruction_lines <=0) {

		printf("file %s is empty\n", argv[2]);
                exit(1);

	}
	HOST0(opcode,int, MAX_INSTRUCTIONS);
	HOST0(dest,int,MAX_INSTRUCTIONS);
	//HOST(index,int, MAX_INSTRUCTIONS);
	HOST0(src0,int, MAX_INSTRUCTIONS);
	HOST0(src1,int, MAX_INSTRUCTIONS);
	HOST0(in_lo, double, MAX_INPUTS);
	HOST0(in_hi, double, MAX_INPUTS);

	asm_stuff stuff = parse_asm(fp, &hv_opcode, &hv_src0, &hv_src1, &hv_in_lo, &hv_in_hi, &hv_dest);

	fclose(fp);
	// allocate #threads based on #sub_intervals
	int N_threads = pow(N_intervals,stuff.INPUT_VARIABLES);
	int block_size = (N_threads<32)?N_threads:32;
	int n_blocks = N_threads/block_size + (N_threads%block_size == 0 ? 0:1);

	if(stuff.OUTPUTS > 1)
	{
		std::vector<instr_format> stuff_atomic[stuff.OUTPUTS];
		cluster(stuff, &hv_opcode, &hv_dest, &hv_src0, &hv_src1);
	}

	DEVICE_COPY(opcode,int);
	DEVICE_COPY(src0,int);
	DEVICE_COPY(src1,int);
	DEVICE_COPY(dest,int);
	DEVICE_COPY(in_lo,double);
	DEVICE_COPY(in_hi,double);

	/*
	   int MONTE_CARLO = 0; // TO use Monte-Carlo,  make DESIGN=fig3 THREADS=1000
	   if (MONTE_CARLO) // monte-carlo range analysis
	   {
	   float *h_mc_range, *d_mc_range; 
	   int *h_time, *d_time;

	// malloc mc_range, time on CPU 
	h_mc_range = (float *)malloc(sizeof(float) * n_blocks * block_size);
	h_time = (int *)malloc(sizeof(int) * 1);

	// malloc mc_range, time on GPU
	cudaMalloc((void **)&d_mc_range, sizeof(float) * n_blocks * block_size);
	cudaMalloc((void **)&d_time, sizeof(int) * 1);

	h_time[0] = (int)time(NULL); //current CPU time

	cudaMemcpy(d_time, h_time, sizeof(int) * 1, cudaMemcpyHostToDevice);

	monte_carlo_range<<< n_blocks, block_size >>> (d_mc_range, d_time, d_in_lo, d_in_hi, stuff.INPUTS);

	cudaMemcpy(h_mc_range, d_mc_range, sizeof(float)*n_blocks*block_size,cudaMemcpyDeviceToHost);

	double max_value = h_mc_range[0];
	double min_value = h_mc_range[0];
	for(int i=0; i<n_blocks*block_size; i++) // retrive min, max range
	{
	if(h_mc_range[i] > max_value)
	max_value = h_mc_range[i];
	if(h_mc_range[i] < min_value)
	min_value = h_mc_range[i];
	}
	printf("\n -------------***** Range from Monte-Carlo *****---------------\n");
	printf("min_value: %f \n max_value: %f \n", min_value, max_value);
	printf("\n -------------***** Range from Monte-Carlo *****---------------\n");

	free(h_mc_range);
	free(h_time);

	cudaFree(d_mc_range);
	cudaFree(d_time);
	}

	 */
	DEVICE_RAW(out_lo,double,N_threads*REGISTERS);
	DEVICE_RAW(out_hi,double,N_threads*REGISTERS);
	//std::cout<< "#intervals: " << N_intervals << "      #threads:  " << N_threads << std::endl;
	peace_range<<< n_blocks, block_size >>> (d_in_lo, d_in_hi, d_out_lo, d_out_hi, d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, N_threads, N_intervals);
	typedef thrust::device_vector<double>::iterator Iterator;

	HOST0(lomin,double, REGISTERS);
	HOST0(himax,double, REGISTERS);
	//HOST(lomin,double);
	//HOST(himax,double);
	//  int i=stuff.ST_ADDR; 
	for(int i =0; i<REGISTERS; i++)
	{
		strided_range<Iterator> reg_lo(dv_out_lo.begin()+i, dv_out_lo.end(), REGISTERS);
		//hv_lomin[i] = thrust::reduce(reg_lo.begin(), reg_lo.end(), 0, thrust::minimum<double>());
		//hv_lomin[i] = *(thrust::min_element(reg_lo.begin(), reg_lo.end()));
		strided_range<Iterator> reg_hi(dv_out_hi.begin()+i, dv_out_hi.end(), REGISTERS);
		hv_himax[i] = thrust::reduce(reg_hi.begin(), reg_hi.end(), 0, thrust::maximum<double>());
		hv_himax[i] = *(thrust::max_element(reg_hi.begin(), reg_hi.end()));
		printf("----REDUCE------%d [%g,%g]\n",i, hv_lomin[i], hv_himax[i]);
	}
	DEVICE_COPY(lomin,double);
	DEVICE_COPY(himax,double);
#ifdef PERF
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&copy_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU copy time is %f ms\n", copy_time);
#endif

	/*
#ifdef PERF
cudaEvent_t start, stop;
float copy_time;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);
#endif
	//PRINT_THRUST_VECTOR(out_hi, double);
	//PRINT_THRUST_VECTOR(out_lo, double);

	HOST(Na, int, stuff.INSTRUCTIONS);
	HOST(Nb, int, stuff.INSTRUCTIONS);
	HOST(bitwidth_array, int, stuff.INSTRUCTIONS);

	// heuristic for pruning space, limit #precision choices into only 2~3
	for(int i=0; i<stuff.INSTRUCTIONS; i++)
	hv_bitwidth_array[i] = high_bound;
	double fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_bitwidth_array);
	std::cout << fixed_error << std::endl;

	for(int i=0; i<stuff.INSTRUCTIONS-1; i++)
	{
	while( fixed_error <= ERROR_THRESH && hv_bitwidth_array[i] > 0)
	{
	hv_bitwidth_array[i]--;  //std::cout << bitwidth_array[i] << std::endl ;
	fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_bitwidth_array);  
	//std::cout << fixed_error << std::endl;
	} //std::cout << bitwidth_array[i] << std::endl;
	hv_Na[i] = hv_bitwidth_array[i]+1; // smallest bitwidth possible
	hv_bitwidth_array[i] = high_bound;
	fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_bitwidth_array);
	}
	fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_Na); 
	std::cout << fixed_error << std::endl;
	for(int i=0; i<stuff.INSTRUCTIONS-1; i++)
	hv_Nb[i] = hv_Na[i];
	while(fixed_error > ERROR_THRESH)
	{
	for(int i=0; i<stuff.INSTRUCTIONS-1; i++)
	hv_Nb[i]++;
	fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_Nb); 
	}
	for(int i=0; i<stuff.INSTRUCTIONS-1; i++)
	std::cout << hv_Na[i] << " " << hv_Nb[i] << std::endl;
	//fixed_error = fx_error(hv_in_lo, hv_in_hi, hv_opcode, hv_src0, hv_src1, hv_dest, stuff.INSTRUCTIONS, hv_Na);
	//std::cout << fixed_error<< std::endl;

	int base = hv_Nb[0]-hv_Na[0]+1;
	//std::cout << "hello: " <<  base << " " << pow(base, stuff.INSTRUCTIONS-1) << std::endl;
	unsigned long BITWIDTH_SPACE = pow(base, stuff.INSTRUCTIONS-1); // the ST instruction should have same precision as the last computing instruction
	//printf("size of int is %d", sizeof(int));
	//printf("search space size: %lu", BITWIDTH_SPACE);
	block_size = 512;
	n_blocks = BITWIDTH_SPACE/block_size + (BITWIDTH_SPACE%block_size == 0 ? 0:1);
	HOST(pow_bitslice, int, stuff.INSTRUCTIONS-1);
	HOST(pow_error, double, high_bound);
	for(int i=0; i<stuff.INSTRUCTIONS-1; i++) 
	hv_pow_bitslice[i] = pow(base, i);
	for(int i=0; i<high_bound; i++) 
	hv_pow_error[i] = pow(2, -(1 + i));
	DEVICE_COPY(pow_bitslice, int);
	DEVICE_COPY(pow_error, double);
	DEVICE_COPY(Na, int);
	DEVICE_COPY(Nb, int);
#ifdef PERF
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&copy_time, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
	//printf("GPU copy time is %f ms\n", copy_time);
#endif

#ifdef PERF
	cudaEvent_t start1, stop1;
	float kernel_time1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
#endif
	//HOST(out_err,double, BITWIDTH_SPACE);
	//HOST(out_area,double, BITWIDTH_SPACE);

	DEVICE_RAW(result,output_stuff,BITWIDTH_SPACE);
	//DEVICE_RAW(out_area,double,BITWIDTH_SPACE);

	//thrust::device_vector<int> d_valid_index;
	//thrust::device_vector<double> d_valid_err;
	//thrust::device_vector<double> d_valid_area;

	peace_error<<< n_blocks, block_size>>>(d_in_lo, d_in_hi, d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, d_Na, d_Nb, BITWIDTH_SPACE, d_pow_bitslice, d_pow_error, d_result, ERROR_THRESH);

	//typedef thrust::device_vector<output_stuff>::iterator Iterator;
	//for(Iterator i=d_result.begin();i!=d_result.end();i++)
	//    std::cout <<  " ,," ;

	//int N_prime = thrust::count_if(d_result.begin(),d_result.end(),is_smaller_than_thresh());

	//int MONTE_CARLO = 1;
	//if (MONTE_CARLO) // monte-carlo precision analysis
	//{
	//    int sampling_cnt = 1; //65536; // pow(2,16)
	//    n_blocks = sampling_cnt/block_size + (sampling_cnt%block_size == 0 ? 0:1);
	//    int current_time = (int)time(NULL); //current CPU time
	//    monte_carlo_error<<< 1, 1>>>(d_in_lo, d_in_hi, d_out_err,d_opcode, d_src0, d_src1, d_dest, stuff.INSTRUCTIONS, d_Na, d_Nb, d_pow_bitslice, d_pow_error, d_out_area, current_time, ERROR_THRESH);
	//}

#ifdef PERF
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&kernel_time1, start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	//printf("GPU error time is %f ms\n", kernel_time1);
#endif
	*/
		/*
		//double result = thrust::reduce(dv_out_err.begin(), dv_out_err.end(), ERROR_THRESH, thrust::minimum<double>());
		double result = *(thrust::max_element(dv_out_err.begin(), dv_out_err.end()));
		//int result = thrust::count_if(dv_out_err.begin(), dv_out_err.end(), is_nonzero());
		printf("Minimum valid error :  %g\n", result);
		//PRINT_THRUST_VECTOR(out_err, double);
		 */  


		//  peace_area<<< n_blocks, block_size>>>(d_out_area, d_opcode, d_dest, stuff.INSTRUCTIONS, Na, Nb, BITWIDTH_SPACE, d_pow_bitslice);
		//PRINT_THRUST_VECTOR(out_area, double);

} // main

