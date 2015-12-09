/* Common Headers */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
/* Self-defined Headers */
#include "opcode.h"
#include "asm.h"
#include "bitgpu_core.h"

using namespace std;
using namespace thrust;

void create_pow_array(host_vector<REAL> *hv_pow_error, int upper_bound_arg) {
	for(int i=0; i<=upper_bound_arg; i++) {
		(*hv_pow_error)[i] = pow(2.0, -(i+1));
	}   
}       


int main(int argc, char** argv)
{
    if(argc!=4) 
    {
        printf("Usage: prune_driver <asm file> <err-thresh> <upper-bound>\n");
        exit(1);
    }

    // read in #lines, #intervals 
    const int lines = line_counter(argv[1]);
    const REAL ERROR_THRESH = atof(argv[2]);
    const int upper_bound_arg = atoi(argv[3]);
    int N_threads;
    int block_size;
    int n_blocks;

    // allocate space on CPU 
    HOST(opcode, int, lines);
    HOST(src0, int, lines);
    HOST(src1, int, lines);
    HOST(dest, int, lines);
    HOST(in_lo, REAL, lines);
    HOST(in_hi, REAL, lines);

    // parse ASM code 
    asm_stuff stuff = parse_asm(argv[1], &hv_opcode, &hv_src0, &hv_src1, 
		    &hv_in_lo, &hv_in_hi, &hv_dest);

    // Prune search space
    HOST(Na, int, lines);
    HOST(Nb, int, lines);
    HOST(bitwidth_array, int, lines);
    // pre-define pow(2,-i) on CPU
    HOST(pow_error, REAL, upper_bound_arg+1);
    create_pow_array(&hv_pow_error, upper_bound_arg);

    // Pseudo code: bit width(0:N-1) <- upper bound fb;
    for(int i=0; i<stuff.INSTRUCTIONS; i++)
	    hv_bitwidth_array[i] = upper_bound_arg+1; // adjusted by 1 to ensure error satisfies

    float out_area;

    REAL current_error = ERROR_THRESH;
    REAL biggest_error = current_error;
	
    /*
     * Pseudo code:
     * while current error < error constraint do
     *     bit width(0:N-1) --;
     * end
     */
    while(current_error <= ERROR_THRESH)
    {
//	    cout << "uniform_bits=" << hv_bitwidth_array[0] << ", error=" << current_error << endl;
	    
	    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		    hv_bitwidth_array[i] -= 1;
		    if(hv_bitwidth_array[i]==0) {
			    current_error = ERROR_THRESH+1;
		    }
	    }
	    
	    bitgpu_core(&hv_in_lo, &hv_in_hi, 
			    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
			    &hv_pow_error,
			    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);

    }
    // Pseudo code: uniform bit = bit width[0]+1;
    int uniform_bitwidth = hv_bitwidth_array[0] + 4;
    cout << "uniform bitwidth is " << uniform_bitwidth << endl;
    int gap =0, gap_max = 0;

    /* 
     * Pseudo code:
     * foreach n=0:N-1 do
     *    while current error <= error constraint do
     *       bit width(n)--;
     *    end
     *    lowest(n) <- bit width(n);
     *    bit width(n)<- uniform bit;
     * end
     */
    for(int i=0; i<stuff.INSTRUCTIONS; i++)
    {
	    while( current_error<=ERROR_THRESH && hv_bitwidth_array[i]>0 )
	    {
		    if(hv_opcode[i]!=ST) {
		    	hv_bitwidth_array[hv_dest[i]]--;  
		    	if(hv_bitwidth_array[hv_dest[i]]==0) {
			    break;
		    	}
		    } else {
		    	hv_bitwidth_array[hv_src0[i]]--;  
		    	if(hv_bitwidth_array[hv_src0[i]]==0) {
			    break;
		    	}
		    }
		    bitgpu_core(&hv_in_lo, &hv_in_hi, 
				    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
				    &hv_pow_error,
				    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);
	    } 

	    hv_Na[i] = hv_bitwidth_array[i]; // must set lower-limit to something sensible..
	    hv_bitwidth_array[i] = upper_bound_arg;
	    current_error = biggest_error;
    }
	
    /*
     * heuristically adjust the upper bound
     * Pseudo code:
     * while current error <= error constraint do
     *    bit width(0:N-1)++;
     * end
     * highest(0:N-1) <- bit width(0:N-1) + guard bit;
     */
    for(int gap=0; gap<uniform_bitwidth; gap++)  {
	    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		    if(hv_opcode[i]==ST) {
			    hv_Nb[i] = hv_Na[i];
			    hv_bitwidth_array[hv_src0[i]]=hv_Nb[i];
		    } else if(hv_opcode[i]==LD && fabs(hv_in_lo[i]-hv_in_hi[i])<1e-30) {
			    //add a guard band of 2 bits to safely cover potentially better solutions within the search space
			    hv_Nb[i] = hv_Na[i]+8;
			    hv_bitwidth_array[hv_dest[i]]=hv_Nb[i];
		    } else {
			    hv_Nb[i] = std::min(hv_Na[i]+gap, uniform_bitwidth+2);
			    hv_bitwidth_array[hv_dest[i]]=hv_Nb[i];
		    }
	    }
	    // validate whether the bounds actually work
	    bitgpu_core(&hv_in_lo, &hv_in_hi, 
			    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
			    &hv_pow_error,
			    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);

	    // stop at the smallest permissible gap
	    if(current_error<ERROR_THRESH) {
		    break;
	    }
	    cout << "gap=" << gap << ", err=" << current_error << ", thresh=" << ERROR_THRESH << endl;
    }

    // reduce search space allocated to wasted operations..
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    if(hv_opcode[i]==ADD || hv_opcode[i]==SUB) {
//		    hv_Na[i]=hv_Nb[i]-1;
	    } else if(hv_opcode[i]==LD && fabs(hv_in_lo[i]-hv_in_hi[i])<1e-30) {
		    hv_Na[i]=hv_Nb[i];
	    }
    }
    
    // final check
    bitgpu_core(&hv_in_lo, &hv_in_hi, 
		    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
		    &hv_pow_error,
		    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);

    if(current_error>=ERROR_THRESH) {
	    cout << "Pruning failed" << endl;
	    exit(1);
    }


    /* for GPU backend */
    char csv1[100]=""; 
    strcat(csv1,argv[1]);
    strcat(csv1,".bitrange.csv");
    FILE *ifs1 = fopen(csv1,"w");
    long long SearchSpaceSize = 1; 
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    fprintf(ifs1,"%d,%d\n",hv_Na[i],hv_Nb[i]);
	    SearchSpaceSize *= (hv_Nb[i] - hv_Na[i] + 1);
	    cout << hv_Na[i] << "," << hv_Nb[i] << endl;
    }
    cout << "search_space : " << SearchSpaceSize << endl;
    fclose(ifs1);

    /* for ASA backend */
    char csv2[100]=""; 
    strcat(csv2,argv[1]);
    strcat(csv2,".asa");
    FILE *ifs2 = fopen(csv2,"w");
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	fprintf (ifs2,"%d\t%d\t%d\t%d\t\t\t1\n",i,hv_Na[i],hv_Nb[i],hv_Nb[i]);
    }
    fclose(ifs2);


} 
