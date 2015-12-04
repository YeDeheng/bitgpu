#include <stdio.h>
#include <assert.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/count.h>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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
    inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", 
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

void profileCopies(float        *h_a, 
        float        *h_b, 
        float        *d, 
        unsigned int  n,
        char         *desc)
{
    printf("%s transfers", desc);

    unsigned int bytes = n * sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent; 

    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    checkCuda( cudaEventRecord(startEvent, 0) );
    checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    float time;
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    //printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
    printf("  Host to Device time: %f\n", time);

    checkCuda( cudaEventRecord(startEvent, 0) );
    checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    //printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
    printf("  Device to Host time: %f\n",  time);

    for (int i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***", desc);
            break;
        }
    }

    // clean up events
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
}

int main()
{
    unsigned int nElements = 16*1024*1024;
    const unsigned int bytes = nElements * sizeof(float);

    // host arrays
    float *h_aPageable, *h_bPageable;   
    float *h_aPinned, *h_bPinned;

    // device array
    float *d_a;

    // allocate and initialize
    h_aPageable = (float*)malloc(bytes);                    // host pageable
    h_bPageable = (float*)malloc(bytes);                    // host pageable
    checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
    checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
    checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

    for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      
    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    // output device info and transfer size
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );

    printf("Device: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

    printf("\n");

  HOST(deheng,unsigned int, );
    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);

    return 0;
}
