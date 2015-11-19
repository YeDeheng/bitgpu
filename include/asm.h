#ifndef ASM_H
#define ASM_H

/* Thrust Headers */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/count.h>

/* Thrust defines */
#define HOST(x,y,size) thrust::host_vector<y> hv_##x(size);
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

typedef struct {
	int opcode;
	int dest;
	int src0;
	int src1;
} instr_format;


typedef struct {
	int INSTRUCTIONS;
	int INPUTS;
	int INPUT_VARIABLES;
	int INPUT_CONSTANTS;
	int OUTPUTS;
	int ST_ADDR;
} asm_stuff;

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


// This method parses the ASM file and return a structure with appropriate metadata..
asm_stuff parse_asm(char* filename, thrust::host_vector<int> *opcode, thrust::host_vector<int> *src0, thrust::host_vector<int> *src1, thrust::host_vector<REAL> *in_lo, thrust::host_vector<REAL> *in_hi, thrust::host_vector<int> *dest);

// use a proper clustering framework that builds a DFG-- what is this hacky program?
void cluster(asm_stuff stuff, thrust::host_vector<int> *opcode, thrust::host_vector<int> *dest,  thrust::host_vector<int> *src0, thrust::host_vector<int> *src1);

// You don't have to reinvent the wheel -- there are functions that would do this for you in the C++ APIs
int line_counter(char* filename);

int get_max_reg(thrust::host_vector<int> dest, asm_stuff stuff);

void parse_intervals(char* filename, int max_reg, 
		thrust::host_vector<REAL> *hv_lomin, thrust::host_vector<REAL> *hv_himax);

void parse_bitrange(char* filename, int max_reg, 
		thrust::host_vector<int> *hv_Na, thrust::host_vector<int> *hv_Nb);

void create_pow_array(thrust::host_vector<REAL> *hv_pow_error, int exponent, int upper_bound_arg);
void create_pow_array_intervals(thrust::host_vector<REAL> *hv_pow_error, int exponent, int upper_bound_arg);

void create_pow_bitslice(thrust::host_vector<int> *hv_pow_bitslice, int max_reg,
	                thrust::host_vector<int> *hv_Na, thrust::host_vector<int> *hv_Nb); 

#endif
