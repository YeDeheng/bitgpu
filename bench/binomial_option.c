// This is a one-step binomial option model, i.e. NUM_STEPS = 1
// Reference: https://software.intel.com/en-us/articles/binomial-options-pricing-model-code-for-intel-xeon-phi-coprocessor
#include<math.h>
binomial_option(double StockPrice, // Stock price of today
				double OptionStrike, 
				double TimePeriod, // time period
				double vol, // volatility
				double risk, // risk free
				double *CallResult
				)
{
    /* Note: for intermediate variables, please define them starting with a capital letter 'T' followed by a digit number */
    /* This is a hacky way for easy SHELL script processing, but is effective */
	double T1 = vol * sqrt(TimePeriod); // variable vDt
	double T2 = risk * TimePeriod;   // variable rDt

	double T3 = (exp(T2) - 1/exp(T1)) / (exp(T1) - 1/exp(T1)); // variable pu
	double T4 = 1 - T3;  // variable pd
	double T5 = StockPrice * exp(T1) - OptionStrike;  // variable d1
	*CallResult = ( T3 * T5 - T4 * T5 ) / exp(T2);
}
