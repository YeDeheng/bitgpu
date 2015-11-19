/* Common Headers */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

/* Self-defined Headers */
#include "opcode.h"
#include "asm.h"
#include "bitslice_core.h"

using namespace std;
using namespace thrust;

static int upper_bound_arg;
static REAL ERROR_THRESH;

void create_pow_array(host_vector<REAL> *hv_pow_error, int upper_bound_arg) {
	for(int i=0; i<=upper_bound_arg; i++) {
		(*hv_pow_error)[i] = pow(2.0, -(i+1));
	}   
}       

int getRNG(int lo, int hi) {
    if(hi < lo)
    {
        unsigned int tempForSwap = hi;
        hi = lo;
        lo = tempForSwap;
    }
     return rand()%(hi - lo + 1) + lo;
 }

/*
int getRNG(int lo, int hi) {
	float dith = 0.5
	float rng01 = ((float)(rand()) + 1.)/((float)(RAND_MAX) + 1.);
	int range = hi - lo;
	
	return lo + (int)(range * rng01 + dith);
}*/

void get_uniform_bitwidth(int Na, int Nb, int *uniform_bitwidth, REALV *temp_lo, REALV *temp_hi, 
		INTV *opcode, INTV *src0, INTV *src1, INTV *dest, int INSTRUCTIONS, 
		REALV *pow_error, 
		REAL *out_area, REAL *current_error, INTV *bits) {
	
	// Pseudo code: bit width(0:N-1) <- Nb;
    for(int i=0; i<INSTRUCTIONS; i++) {
	    ARRAY_ACCESS(bits, i) = Nb;
	}
	bitslice_core(temp_lo, temp_hi, 
			    opcode, src0, src1, dest, INSTRUCTIONS, 
			    pow_error,
			    out_area, current_error, bits, ERROR_THRESH);
	if(*current_error <= ERROR_THRESH) {
		if( Nb - Na <= 1) {
			*uniform_bitwidth = Na;
		} else {
			//Recursive Stratified Sampling
			get_uniform_bitwidth(Na, getRNG(Na, Nb), uniform_bitwidth, temp_lo, temp_hi, 
						opcode, src0, src1, dest, INSTRUCTIONS, 
						pow_error, 
						out_area, current_error, bits);
		}
	} else {
		if((Nb ==  upper_bound_arg) || (Nb -Na <= 1)) {
			*uniform_bitwidth = Nb;
		} else {
			//Recursive Stratified Sampling
			get_uniform_bitwidth(getRNG(Na, Nb), Nb, uniform_bitwidth, temp_lo, temp_hi, 
						opcode, src0, src1, dest, INSTRUCTIONS, 
						pow_error, 
						out_area, current_error, bits);
		}
	}

}

void get_lower_bound(int index, int Na, int Nb, REALV *temp_lo, REALV *temp_hi, 
		INTV *opcode, INTV *src0, INTV *src1, INTV *dest, int INSTRUCTIONS, 
		REALV *pow_error, 
		REAL *out_area, REAL *current_error, INTV *bits) {
	
	// Pseudo code: bit width(index) <- Nb;
	ARRAY_ACCESS(bits, index) = Nb;
	
	bitslice_core(temp_lo, temp_hi, 
			    opcode, src0, src1, dest, INSTRUCTIONS, 
			    pow_error,
			    out_area, current_error, bits, ERROR_THRESH);
	if(*current_error <= ERROR_THRESH) {
		if( Nb - Na <= 1) {
			ARRAY_ACCESS(bits, index) = Na;
		} else {
			//Recursive Stratified Sampling
			get_lower_bound(index, Na, getRNG(Na, Nb), temp_lo, temp_hi, 
					opcode, src0, src1, dest, INSTRUCTIONS, 
					pow_error, 
					out_area, current_error, bits);
        }
	} else {
		if((Nb ==  upper_bound_arg) || (Nb -Na <= 1)) {
			ARRAY_ACCESS(bits, index) = Nb;
		} else {
			//Recursive Stratified Sampling
			get_lower_bound(index, getRNG(Na, Nb), Nb, temp_lo, temp_hi, 
				opcode, src0, src1, dest, INSTRUCTIONS, 
				pow_error, 
				out_area, current_error, bits);
		}
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
    ERROR_THRESH = atof(argv[2]);
    upper_bound_arg = atoi(argv[3]);
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
    // pre-define pow(2,-i) to avoid wasting GPU time in each ste
    HOST(pow_error, REAL, upper_bound_arg+1);
    create_pow_array(&hv_pow_error, upper_bound_arg);

    // Pseudo code: bit width(0:N-1) <- upper bound fb;
    //for(int i=0; i<stuff.INSTRUCTIONS; i++)
	//    hv_bitwidth_array[i] = upper_bound_arg+1; // adjusted by 1 to ensure error satisfies

    float out_area;

    REAL current_error = ERROR_THRESH;
    REAL biggest_error = current_error;

	int uniform_bitwidth = upper_bound_arg + 1;
	get_uniform_bitwidth(0, upper_bound_arg, &uniform_bitwidth, &hv_in_lo, &hv_in_hi, 
									&hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
									&hv_pow_error,
									&out_area, &current_error, &hv_bitwidth_array);
    
	/* 
	 * Pseudo code:
	 * while current error < error constraint do
     *     bit width(0:N-1) --;
     * end
	 */
    /*while(current_error <= ERROR_THRESH)
    {
//	    cout << "uniform_bits=" << hv_bitwidth_array[0] << ", error=" << current_error << endl;
	    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		    hv_bitwidth_array[i] -= 1;
		    if(hv_bitwidth_array[i]==0) {
			    current_error = ERROR_THRESH+1;
		    }
	    }
	    
	    bitslice_core(&hv_in_lo, &hv_in_hi, 
			    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
			    &hv_pow_error,
			    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);

    }*/
    // Pseudo code: uniform bit = bit width[0]+1;
    //int uniform_bitwidth = hv_bitwidth_array[0] + 1;
    cout << "uniform bitwidth is " << uniform_bitwidth << endl;
    int gap =0, gap_max = 0;
	
	for(int i=0; i<stuff.INSTRUCTIONS; i++)
    {
		if(hv_opcode[i]!=ST) {
	    	get_lower_bound(hv_dest[i], 0, hv_bitwidth_array[hv_dest[i]], &hv_in_lo, &hv_in_hi, 
				&hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
				&hv_pow_error,
				&out_area, &current_error, &hv_bitwidth_array);
	    } else {
	    	get_lower_bound(hv_src0[i], 0, hv_bitwidth_array[hv_src0[i]], &hv_in_lo, &hv_in_hi, 
				&hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
				&hv_pow_error,
				&out_area, &current_error, &hv_bitwidth_array);  
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
				hv_Nb[i] = hv_Na[i]+2;
			    hv_bitwidth_array[hv_dest[i]]=hv_Nb[i];
		    } else {
			    hv_Nb[i] = std::min(hv_Na[i]+gap, uniform_bitwidth+2);
			    hv_bitwidth_array[hv_dest[i]]=hv_Nb[i];
		    }
	    }
	    // validate whether the bounds actually work
	    bitslice_core(&hv_in_lo, &hv_in_hi, 
			    &hv_opcode, &hv_src0, &hv_src1, &hv_dest, stuff.INSTRUCTIONS, 
			    &hv_pow_error,
			    &out_area, &current_error, &hv_bitwidth_array, ERROR_THRESH);

	    // stop at the smallest permissible gap
	    if(current_error<ERROR_THRESH) {
		    break;
	    }
	    cout << "gap=" << gap << ", err=" << current_error << ", thresh=" << ERROR_THRESH << endl;
    }

	
    // Pseudo code: lowest(n) <- bit width(n)
    /*for(int i=0; i<stuff.INSTRUCTIONS; i++)
    {
	    hv_Na[i] = uniform_bitwidth;
    }*/
    
    // heuristically adjust the upper bound
    /*for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    if(hv_opcode[i]==LD && fabs(hv_in_hi[i]-hv_in_lo[i])<1e-30) {
		    //add a guard band of 2 bits to safely cover potentially better solutions within the search space
			hv_Nb[i] = hv_Na[i]+2; 
		    hv_Na[i] = hv_Nb[i];
	    } else if(hv_opcode[i]==ST) {
		    hv_Nb[i] = hv_Na[i];
	    } else {
		    hv_Nb[i] = upper_bound_arg;
	    }
    }*/

	// reduce search space allocated to wasted operations..
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    if(hv_opcode[i]==ADD || hv_opcode[i]==SUB) {
//		    hv_Na[i]=hv_Nb[i]-1;
	    } else if(hv_opcode[i]==LD && fabs(hv_in_lo[i]-hv_in_hi[i])<1e-30) {
		    hv_Na[i]=hv_Nb[i];
	    }
    }
    
    // final check
    bitslice_core(&hv_in_lo, &hv_in_hi, 
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
    for(int i=0; i<stuff.INSTRUCTIONS; i++) {
	    fprintf(ifs1,"%d,%d\n",hv_Na[i],hv_Nb[i]);
//	    cout << hv_Na[i] << "," << hv_Nb[i] << endl;
    }
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
