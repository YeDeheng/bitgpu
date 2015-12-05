#include <stdio.h>
#include <math.h>

#define LIMIT 100

enum model_type_t {RANGE=0, ERROR=1, AREA=2, RANDOM=3};
enum range_kernel_t {ADD_RANGE=0, SUB_RANGE=1, MUL_RANGE=2, DIV_RANGE=3, EXP_RANGE=4, LOG_RANGE=5};
enum error_kernel_t {ADD_ERROR=0, SUB_ERROR=1, MUL_ERROR=2, DIV_ERROR=3, EXP_ERROR=4, LOG_ERROR=5};
enum area_kernel_t {ADD_AREA=0, SUB_AREA=1, MUL_AREA=2, DIV_AREA=3, EXP_AREA=4, LOG_AREA=5};

// Wrapper Kernels
void wrapper_range_kernel(range_kernel_t type, REAL* in0, REAL* in1, REAL* in2, REAL* in3, REAL* out0, REAL* out1, int tid);
void wrapper_error_kernel(error_kernel_t type, REAL* x0, REAL* x1, REAL* e1, REAL* y0, REAL* y1, REAL* e2, REAL* out0, REAL* out1, REAL* pow_error, REAL* out_error, int t0, int tid);
void wrapper_area_kernel(area_kernel_t type, REAL* x0, REAL* x1, REAL* y0, REAL* y1, REAL* out0, REAL* out1, REAL* out_area, int t0, int tid);
void wrapper_rand_kernel(unsigned long seed, int tid);
