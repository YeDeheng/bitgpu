#include <cstdlib>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <algorithm>

#include "opcode.h"
#include "asm.h"

using namespace std;
using namespace thrust;

#define STRINGIFY(s) str(s)
#define str(s) #s

// This method parses the ASM file and return a structure with appropriate metadata..
asm_stuff parse_asm(char* filename, host_vector<int> *opcode, host_vector<int> *src0, host_vector<int> *src1, host_vector<REAL> *in_lo, host_vector<REAL> *in_hi, host_vector<int> *dest) 
{
    FILE *fp;
    fp = fopen(filename,"r");
    if(!fp) 
    {
        printf("%s does not exist\n",filename);
        exit(1);
    }
    int i=0;
    int loads=0, stores=0, variables=0, constants=0;
    asm_stuff stuff;
    while (!feof(fp)) 
    {
        int op, desti;
        (void)fscanf(fp,"%d,%d,",&op,&desti);
        (*opcode)[i]=op;
        (*dest)[i]=desti;
        if((*opcode)[i]==ST) 
            stores++;
        if((*opcode)[i]==LD) 
        {
            loads++;
            REAL val0, val1;
            if(strcmp( STRINGIFY(REAL) , "float" ) == 0) {
                (void)fscanf(fp,"%f,%f;\n",&val0, &val1);
            } else {
                (void)fscanf(fp,"%lf,%lf;\n",&val0, &val1);
            }
            (*in_lo)[desti]=val0;
            (*in_hi)[desti]=val1;
            if( val0 == val1 )
                constants++;
            else
                variables++;
            (*src0)[i]=-1;
            (*src1)[i]=-1;
        } 
        else 
        {
            int src0d, src1d;
            (void)fscanf(fp,"%d,%d;\n", &src0d, &src1d);
            (*src0)[i]=src0d;
            (*src1)[i]=src1d;
            if((*opcode)[i]==ST)
                stuff.ST_ADDR = (*src0)[i];
        }
        /*
        cout << "i=" << i << ", " << (*opcode)[i] << "," << 
					(*dest)[i] << "," << 
					(*in_lo)[i] << "," << 
					(*in_hi)[i] << "," << 
					(*src0)[i] << "," << 
					(*src1)[i] << "," << 
					endl;
					*/
        i++;
    }
    fclose(fp);
    stuff.INSTRUCTIONS=i;
    stuff.INPUTS=loads;
    stuff.INPUT_VARIABLES=variables;
    stuff.INPUT_CONSTANTS=constants;
    stuff.OUTPUTS=stores;
    return stuff;
}

// use a proper clustering framework that builds a DFG-- what is this hacky program?
void cluster(asm_stuff stuff, host_vector<int> *opcode, host_vector<int> *dest,  host_vector<int> *src0, host_vector<int> *src1)
{
    int cluster_cnt = stuff.OUTPUTS;
    int j = -1;
    instr_format instr_temp;
    vector<instr_format>* stuff_atomic = new vector<instr_format>[cluster_cnt];
    for(int i=stuff.INSTRUCTIONS-1; i>=0 && (*opcode)[i]!=LD; i--) //initialize the stack
    {
        if( (*opcode)[i] == ST )
        {
            j++;
            instr_temp = (instr_format) { (*opcode)[i], (*dest)[i], (*src0)[i], (*src1)[i] };
            stuff_atomic[j].push_back(instr_temp);
        }
        else
        {
            instr_temp = (instr_format) { (*opcode)[i], (*dest)[i], (*src0)[i], (*src1)[i] };
            stuff_atomic[j].push_back(instr_temp);
        }
    }
    for(int i=0; i<cluster_cnt; i++)
    {
        std::vector<int> source;
        for(int j=0; j<stuff_atomic[i].size(); j++) 
        {
            source.push_back(stuff_atomic[i][j].src0);
            source.push_back(stuff_atomic[i][j].src1);
        }
        source.erase(std::remove(source.begin(), source.end(), -1), source.end()); // remove src=-1, which is due to ST instruction
        for(int j=0; j<stuff_atomic[i].size(); j++) 
            source.erase(std::remove(source.begin(), source.end(), stuff_atomic[i][j].dest), source.end());
        for(int m=0; m<source.size(); m++)
        {
            int n = source[m];
            instr_temp = (instr_format) { (*opcode)[n], (*dest)[n], (*src0)[n], (*src1)[n]};
            stuff_atomic[i].push_back(instr_temp);
        }
    }
}

// You don't have to reinvent the wheel -- there are functions that would do thsi for you in the C++ APIs
int line_counter(char* filename)
{
    int lines = 0;
    char ch;
    FILE* fp = fopen(filename, "r");
    if(!fp)
    {
        printf("%s does not exist\n",filename);
        exit(1);
    }
    while(!feof(fp))
    {
        ch = fgetc(fp);
        if(ch == '\n')
            lines++;
    }
    return lines;
}

int get_max_reg(host_vector<int> hv_dest, asm_stuff stuff) {
	int max_reg=0;
	for(int i=0; i<stuff.INSTRUCTIONS; i++) {
		if(max_reg < hv_dest[i]) {
			max_reg = hv_dest[i];
		}
	}
	return max_reg;
}

void parse_intervals(char* filename, int max_reg, host_vector<REAL> *hv_lomin, host_vector<REAL> *hv_himax) {
	char csv[100]="";
	strcat(csv,filename);
	strcat(csv,".interval.csv");
	FILE *ifs = fopen(csv,"r");
	for(int i=0; i<max_reg && !feof(ifs); i++) {
		REAL lomin, himax;
		if(strcmp( STRINGIFY(REAL) , "float" ) == 0) {
			(void)fscanf(ifs,"%f,%f\n",&lomin,&himax);
		} else {
			(void)fscanf(ifs,"%lf,%lf\n",&lomin,&himax);
		}
		(*hv_lomin)[i]=lomin;
		(*hv_himax)[i]=himax;
	}
	fclose(ifs);

}

void parse_bitrange(char* filename, int max_reg, host_vector<int> *hv_Na, host_vector<int> *hv_Nb) {
	char csv1[100]="";
	strcat(csv1,filename);
	strcat(csv1,".bitrange.csv");
	FILE *ifs1 = fopen(csv1,"r");
	for(int i=0; i<max_reg && !feof(ifs1); i++) {
		int Na, Nb;
		(void)fscanf(ifs1,"%d,%d\n",&Na,&Nb);
		(*hv_Na)[i]=Na;
		(*hv_Nb)[i]=Nb;
//		cout << "i=" << i << " Na=" << Na << endl;
	}
	fclose(ifs1);
}

void create_pow_array(host_vector<REAL> *hv_pow_error, int exponent, int upper_bound_arg) {
	for(int i=0; i<upper_bound_arg; i++) {
		(*hv_pow_error)[i] = pow((double)exponent, -(i+1));
	}
}

void create_pow_array_intervals(host_vector<REAL> *hv_pow_error, int exponent, int upper_bound_arg) {
	for(int i=0; i<upper_bound_arg; i++) {
		(*hv_pow_error)[i] = pow(exponent, i);
	}
}

void create_pow_bitslice(host_vector<int> *hv_pow_bitslice, int max_reg,
		host_vector<int> *hv_Na, host_vector<int> *hv_Nb) 
{

	(*hv_pow_bitslice)[0] = ((*hv_Nb)[0]-(*hv_Na)[0]+1);
	for(int j=1; j<max_reg; j++) {
		(*hv_pow_bitslice)[j] = (*hv_pow_bitslice)[j-1]*
			((*hv_Nb)[j]-(*hv_Na)[j]+1);
	}
	for(int j=0; j<max_reg; j++) {
		(*hv_pow_bitslice)[j] = (*hv_pow_bitslice)[max_reg-1]/(*hv_pow_bitslice)[j];
	}
}
