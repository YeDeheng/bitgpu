#ifndef BITSLICE_H2
#define BITSLICE_H2

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define REALV thrust::host_vector<REAL>
#define INTV thrust::host_vector<int>
#define ARRAY_ACCESS(x,j) (*x)[j]


void bitslice_core(REALV *temp_lo, REALV *temp_hi,
		INTV *opcode, INTV *src0, INTV *src1, INTV *dest, int INSTRUCTIONS,
		REALV *pow_error,
		REAL *out_area, REAL *out_err, INTV *bits, REAL ERROR_THRESH);

#endif
