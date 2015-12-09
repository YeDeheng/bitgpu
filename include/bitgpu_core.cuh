/* standard headers */
#include <stdio.h>
#include <math.h>
#include <time.h>

/* self-defined headers */
#include "opcode.h"

 
 /* 
  * process the error equations and resource model expressions to determine the cumulative error of the output variable and resource costs, 
  * and save in out_err array and out_area array respectively. Treat bitwidth combinations that result in error
  * larger than the user-supplied error constraint as invalid.
  */
__forceinline__ __device__ void bitgpu_core(REALV *temp_lo, REALV *temp_hi, 
		INTV *opcode, INTV *src0, INTV *src1, INTV *dest, int INSTRUCTIONS, 
		REALV *pow_error, 
		REAL *out_area, REAL *out_err, INTV *bits, REAL ERROR_THRESH)
{
	REAL error[REGISTERS] = {0};

	float area_total=0;

	int integer_bit_src0;
	int integer_bit_src1;

	for(int j=0; j<INSTRUCTIONS; j++)
	{
		int destj = ARRAY_ACCESS(dest, j);
		int src0j = ARRAY_ACCESS(src0, j);
		int src1j = ARRAY_ACCESS(src1, j);
		int t0 = ARRAY_ACCESS(bits, destj);
		REAL powt0 = ARRAY_ACCESS(pow_error, t0);
		REAL powt0m1 = ARRAY_ACCESS(pow_error, t0-1);
		int b0 = ARRAY_ACCESS(bits, src0j);
		int b1 = ARRAY_ACCESS(bits, src1j);
		integer_bit_src0 = integer_bit_calc(ARRAY_ACCESS(temp_lo, src0j), ARRAY_ACCESS(temp_hi, src0j)); // found copy-paste bug
		integer_bit_src1 = integer_bit_calc(ARRAY_ACCESS(temp_lo, src1j), ARRAY_ACCESS(temp_hi, src1j));
		REAL tlo0 = ARRAY_ACCESS(temp_lo, src0j);
		REAL thi0 = ARRAY_ACCESS(temp_hi, src0j);
		REAL tlo1 = ARRAY_ACCESS(temp_lo, src1j);
		REAL thi1 = ARRAY_ACCESS(temp_hi, src1j);
		REAL error0 = error[src0j];
		REAL error1 = error[src1j];
		int code = ARRAY_ACCESS(opcode, j);
		REAL *erroraddr = &(error[destj]);

		REAL tlo, thi;
		if(code == LOG) 
		{
			log_rangerule(tlo0, thi0, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			log_errrule(tlo0, thi0, error0, powt0, erroraddr);
			area_total += log_area(integer_bit_src0 + b0);
		} 
		else if(code == EXP)
		{
			exp_rangerule(tlo0, thi0, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			exp_errrule(tlo0, thi0, error0, powt0, erroraddr);
			area_total += exp_area(integer_bit_src0 + b0);
		}
		else if(code == DIV) 
		{
			div_rangerule(tlo0, thi0, tlo1, thi1, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			div_errrule(tlo0, thi0, error0, tlo1, thi1, error1, powt0, erroraddr);
			area_total += div_area(max(integer_bit_src0 + b0, integer_bit_src1 + b1));
		} 
		else if(code == MUL) 
		{
			mult_rangerule(tlo0, thi0, tlo1, thi1, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			mult_errrule(tlo0, thi0, error0, tlo1, thi1, error1, powt0, erroraddr);
			area_total += mult_area(integer_bit_src0 + b0, integer_bit_src1 + b1);

		} 
		else if(code == ADD)
		{
			add_rangerule(tlo0, thi0, tlo1, thi1, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			add_errrule(tlo0, thi0, error0, tlo1, thi1, error1, powt0, erroraddr);
			area_total += add_area(integer_bit_src0 + b0, integer_bit_src1 + b1);
		} 
		else if(code == SUB)
		{
			sub_rangerule(tlo0, thi0, tlo1, thi1, &tlo, &thi);
			ARRAY_ACCESS(temp_lo, destj) = tlo; ARRAY_ACCESS(temp_hi, destj) = thi;

			sub_errrule(tlo0, thi0, error0, tlo1, thi1, error1, powt0, erroraddr);
			area_total += add_area(integer_bit_src0 + b0, integer_bit_src1 + b1) + 1;
		} 
		else if(code == LD) 
		{
			thi = ARRAY_ACCESS(temp_hi, destj);
			tlo = ARRAY_ACCESS(temp_lo, destj);
			// handling fractional constants again
			if( fabs(thi-tlo) < 1e-30 )
			{
				double shift = tlo/powt0m1;
				double intpart;
				double fractpart = modf(shift, &intpart);
				if(fractpart > 1e-30)
				{
					*erroraddr = powt0;
				}
				else 
				{
					*erroraddr = 0;
				}
			}
			else 
			{
				*erroraddr = powt0;
			}

		}
		else if(code == ST) 
		{
			area_total += b0;
			if( error0 > ERROR_THRESH )
			{
				*out_area = INT_MAX;
			}
			else 
			{
				*out_area = area_total;
			}

			*erroraddr = error0;
			*out_err = error0;
		}
	}
	
} 

