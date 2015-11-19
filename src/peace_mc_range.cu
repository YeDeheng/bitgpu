#include "peace.cuh"
#include<stdio.h>
//#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <cuda_runtime.h>

__global__ void peace_mc_range (float *res, int *time, double* in_lo, double* in_hi, int INPUTS) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // current time for random seed generation
    int t = *time; 

    res += idx;

    /* Init curand */
    curandState s;
    curand_init(t, idx, 0, &s);

    double input_array[MAX_INPUTS];
    for(int i=0; i<INPUTS; i++)
    {
        double range = in_hi[i] - in_lo[i]; 
        // curand uniform distribution from [0,1]
        input_array[i] = curand_uniform(&s)*range + in_lo[i];
    }

    // currently use a*b+c-b as example here. 
    //*res = input_array[0]*input_array[1] + input_array[2] - input_array[1];

    // appolonius
    *res = input_array[0]*(input_array[0] - 2*input_array[3]) - input_array[1]*(input_array[1]*input_array[1] - 2*input_array[3]) - input_array[2]*(input_array[2] - 2*input_array[4]);
    //*res = input_array[0]*input_array[0] - input_array[1]*input_array[1]*input_array[1] - input_array[2]*input_array[2] - 2*input_array[3]*(input_array[0]-input_array[1]) + 2*input_array[2]*input_array[4];

    printf("sampling data are : %f, %f, %f, %f, resulting data is : %f\n", input_array[0], input_array[1], input_array[2], input_array[3], input_array[4], *res);
    //poly4
    //*res = input_array[0]*(1 - input_array[0]*(0.5 - input_array[0]*(0.3 - input_array[0]*0.25))); 
    //poly6
    //*res = input_array[0]*(1 - input_array[0]*(0.5 - input_array[0]*(0.3 - input_array[0]*(0.25 - input_array[0]*(0.2 - input_array[0]*0.17)))));
    //poly8
    //*res = input_array[0]*(1 - input_array[0]*(0.5 - input_array[0]*(0.3 - input_array[0]*(0.25 - input_array[0]*(0.2 - input_array[0]*(0.17 - input_array[0]*(0.14 - input_array[0]*0.125)))))));
    //printf("sampling data is : %f, resulting data is : %f\n", input_array[0], *res);
}
