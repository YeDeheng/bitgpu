#include<iostream>
#include<stdio.h>
#include<math.h>

#include "opcode.h"
#include<thrust/device_vector.h>
#include "range_host.h"

#ifndef ERROR_H2_
#define ERROR_H2_

// No need to create a copy of the original error equations!
#define __device__ 
#include "error.cuh"

#endif
