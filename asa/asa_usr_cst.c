/************************************************************************
* Adaptive Simulated Annealing (ASA)
* Lester Ingber <ingber@ingber.com>
* Copyright (c) 1987-2013 Lester Ingber.  All Rights Reserved.
* ASA-LICENSE file has the license that must be included with ASA code.
***********************************************************************/

 /* $Id: asa_usr_cst.c,v 29.6 2013/10/19 21:30:59 ingber Exp ingber $ */

 /* asa_usr_cst.c for Adaptive Simulated Annealing */

#include "asa_usr.h"

#include<time.h>
#if COST_FILE
#include<iostream>
#include<vector>
#include<algorithm>

#include "bitslice_core.h"
#include "helper.h"

using namespace std;
using namespace thrust;

//#define ERROR_THRESH pow(2,-9)
int cost_func_cnt=0;
int min_area=100000000;
//int min_area=10;
float total_time=0;
int iter=0;
TIMER start, stop;

 /* Note that this is a trimmed version of the ASA_TEST problem.
    A version of this cost_function with more documentation and hooks for
    various templates is in asa_usr.c. */

 /* If you use this file to define your cost_function (the default),
    insert the body of your cost function just above the line
    "#if ASA_TEST" below.  (The default of ASA_TEST is FALSE.)

    If you read in information via the asa_opt file (the default),
    define *parameter_dimension and
    parameter_lower_bound[.], parameter_upper_bound[.], parameter_int_real[.]
    for each parameter at the bottom of asa_opt.

    The minimum you need to do here is to use
    x[0], ..., x[*parameter_dimension-1]
    for your parameters and to return the value of your cost function.  */

double cost_function (double *x,
               double *parameter_lower_bound,
               double *parameter_upper_bound,
               double *cost_tangents,
               double *cost_curvature,
               ALLOC_INT * parameter_dimension,
               int *parameter_int_real,
               int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS)
{

  /* *** Insert the body of your cost function here, or warnings
   * may occur if COST_FILE = TRUE & ASA_TEST != TRUE ***
   * Include ADAPTIVE_OPTIONS below if required */

		// Attempting to relocate this to asa_usr.c
/*		HOST(opcode,int, MAX_INSTRUCTIONS);
		HOST(src0,int, MAX_INSTRUCTIONS);
		HOST(src1,int, MAX_INSTRUCTIONS);
		HOST(dest,int, MAX_INSTRUCTIONS);
		HOST(in_lo,REAL, MAX_INPUTS);
		HOST(in_hi,REAL, MAX_INPUTS);
*/

	if(0) {
		char filename[100];
		sprintf(filename, "%s", DESIGN);
		asm_stuff stuff = parse_asm(filename, &hv_opcode, &hv_src0, &hv_src1, &hv_in_lo, &hv_in_hi, &hv_dest);
	}
    
	if(cost_func_cnt==0) {
    		record_time(&stop);
	}
    cost_func_cnt++;

    HOST(pow_error, REAL, UPPER_BOUND);
    for(int i=0; i<UPPER_BOUND; i++) {
	    hv_pow_error[i] = pow(2.0, -(i+1));
    }
    HOST(xint,int, MAX_INSTRUCTIONS);
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    if(hv_opcode[i]!=ST) {
	    	hv_xint[hv_dest[i]] = (int)x[i];
	    } else {
	    	hv_xint[hv_src0[i]] = (int)x[i];
	    }
    }

    REAL out_err;
    float out_area;

    //TIMER start, stop;
    //record_time(&start);
    bitslice_core(&hv_in_lo, &hv_in_hi, 
		    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
		    &hv_pow_error,
		    &out_area, &out_err, &hv_xint, ERROR_THRESH);
    start=stop;
    record_time(&stop);
    float time = calculate_time(&start,&stop);
    

    if(out_area == INT_MAX )
    {
        *cost_flag = FALSE;
        return (0);
    }
    else 
    {
        *cost_flag = TRUE;
    }

    // print min_area and accumulated time
    if(out_area < min_area) {
	    min_area = out_area;
    }
    total_time+=time;
    cout << min_area << "," << total_time;
    cout << ",[";
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
    	cout << hv_xint[i] << "-";
    }
    cout << "]," << cost_func_cnt << endl;
    
    double ret = (double)out_area;
    return ret; 

}
#endif /* COST_FILE */
    
/*
    if (iter == TargetIter)
    {
	    printf("Target iterations reached\n");
	    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		    cout << "PART " << parameter_lower_bound[i] << "," << x[i] << endl;
	    }
	    exit(0);
    }

    iter++;
*/

