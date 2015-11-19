/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
/* Self-defined Headers */
#include "opcode.h"
#include "asm.h"

using namespace std;
using namespace thrust;

int main(int argc, char** argv)
{
    if(argc!=2) 
    {
        printf("Usage: test_enum <asm file>\n");
        exit(1);
    }

    // read in #lines, #intervals 
    const int lines = line_counter(argv[1]);
    
    // allocate space on CPU 
    HOST(opcode,int, lines);
    HOST(src0,int, lines);
    HOST(src1,int, lines);
    HOST(dest,int, lines);
    HOST(in_lo,REAL, lines);
    HOST(in_hi,REAL, lines);
    
    // parse ASM code 
    asm_stuff stuff = parse_asm(argv[1], &hv_opcode, &hv_src0, &hv_src1, &hv_in_lo, &hv_in_hi, &hv_dest);

    HOST(Na, int, stuff.INSTRUCTIONS);
    HOST(Nb, int, stuff.INSTRUCTIONS);

    parse_bitrange(argv[1], stuff.INSTRUCTIONS, &hv_Na, &hv_Nb);

    // create the search space and sizes for warps/blocks
    // ST instruction should have same precision as the last computing instruction
    // Ok the gap between Na and Nb is magically the same size...
    unsigned long search_space = 1;
    for(int j=0; j<stuff.INSTRUCTIONS; j++) {
    	search_space *= (hv_Nb[j] - hv_Na[j] + 1);
    }
    
    HOST(pow_bitslice, int, stuff.INSTRUCTIONS);
    create_pow_bitslice(&hv_pow_bitslice, stuff.INSTRUCTIONS, &hv_Na, &hv_Nb);

    for(int position=0; position<search_space; position++) {
	    cout << ",[";
	    for(int j=0; j<stuff.INSTRUCTIONS; j++)
	    {
		    int CommonCode = position/hv_pow_bitslice[j];
		    int t0 = hv_Na[j] + CommonCode%(hv_Nb[j]-hv_Na[j]+1);
		    cout << t0 << "-";
	    }
	    cout << "]" << endl;
    }
} 
