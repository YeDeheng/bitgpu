#include<math.h>
#include "opcode.h"
//#include "peace.cuh"

#ifndef RANGE_H2_
#define RANGE_H2_

// yet again no need for stupid replication of code
// http://stackoverflow.com/questions/3085071/how-to-redefine-a-macro-using-its-previous-definition to avoid idiotic redefinition of macro warning
#ifdef __device__
#undef __device__
#endif
#ifdef __forceinline__
#undef __forceinline__
#endif

#define __device__ 
#define __forceinline__

using namespace std;

#ifdef USE_MC
#include "mc_range_host.h"
#else
#include "range.cuh"
#endif


#endif
