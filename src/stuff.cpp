/* Common Headers */
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <time.h>
/* Self-defined Headers */
#include "opcode.h"
#include "asm.h"

using namespace std;

/*
 * Appropriately named file extracts useful information about dataflow expression
 * generated from GIMPLE pass from the C function
 */
int main(int argc, char** argv)
{
    if(argc!=2) 
    {
        printf("Usage: stuff <asm file>\n");
        exit(1);
    }

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

    cout << "INPUT_VARIABLES=" << stuff.INPUT_VARIABLES << endl;
    cout << "INPUT_CONSTANTS=" << stuff.INPUT_CONSTANTS<< endl;
    cout << "INPUTS=" << stuff.INPUTS<< endl;
    cout << "OUTPUTS=" << stuff.OUTPUTS<< endl;
    cout << "INSTRUCTIONS=" << stuff.INSTRUCTIONS << endl;

}
