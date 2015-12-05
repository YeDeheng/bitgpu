#ifndef AREA_H_
#define AREA_H_

/* 
 * Area Models
 * Input: bitwidth x (for single-input operators)
 *        bitwidths x1 and x2 (for two-input operators)
 * Output: area (Number of LUTs)
 * Method: Regression fitting from HLS kernels for these operators mapped to ML605 board
 * Bitwidth varied, resulting LUT counts recorded.
 *
 * Shown in Figure 1 of FPGA 2016 paper  (Page 3)
 * "GPU-Accelerated High-Level Synthesis for Bitwidth Optimization of FPGA Datapaths"
 *
*/
__forceinline__ __device__ float sqrt_area(int x) {
    float area = 20.041 * x + 2324.24;
    return area;
}
__forceinline__ __device__ float exp_area(int x) {
    float area = 2718.857 + 18.66216*x + 0.05235931*x*x - 4.570707e-04*x*x*x;
    return area;
}
__forceinline__  __device__ float log_area(int x) {
    float area = 2032.929 + 19.08337*x + 0.05482684*x*x - 5.176768e-04*x*x*x;
    return area;
}
__forceinline__  __device__ int div_area(int x) {
    int area = 252.1429 - 84.09784*x + 8.084491*x*x + 2.272727e-05*x*x*x;
    return area;
}
__forceinline__  __device__ int mult_area(int x1, int x2) {
    int area = x1*x2;
    return area;
}
__forceinline__  __device__ int add_area(int x1, int x2) {
    int area = (x1 > x2) ? x1 : x2;
    return area;
}
#endif 
