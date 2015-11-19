#include<math.h>

// outputs are denoted with '*'
// Reference: CUDA black-scholes implementation
BlackScholes(
    double StockPrice, //S1tock price
    double OptionStrike, //Option strike
    double TimePeriod, //Option years
    double risk, //Riskless rate
    double vol,  //Volatility rate
    double A1, // = 0.31938153f;
    double A2, // = -0.356563782f;
    double A3, // = 1.781477937f;
    double A4, // = -1.821255978f;
    double A5, // = 1.330274429f;
    double PI, //= 0.398942f;
    
    double const1, 
    double const2, 
    double const3, 

//    double *CallResult,
    double *W1
)
{
    double T0 = sqrt(TimePeriod);  // variable M1
    double T1 = (log(StockPrice / OptionStrike) + (risk + const2 * vol * vol) * TimePeriod) / (vol * T0);  // variable d1
    double T2 = T1 - vol * T0;  // variable d2

    double T3 = const1 / (const1 + const3 * T1);  // variable K1
    double T4 = const1 / (const1 + const3 * T2);  // variable K2

    double T5 = T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * (A4 + T3 * A5))));  // tmp1
    double T6 = T4 * (A2 + T4 * (A2 + T4 * (A3 + T4 * (A4 + T4 * A5))));  // tmp2

    double T7 = PI * exp(const1 - const2 * T1 * T1) * T5; // CNDD1
    double T8 = PI * exp(const1 - const2 * T2 * T2) * T6; // CNDD2

    //Calculate Call and Put simultaneously
    double T9 = const1 / exp(risk * TimePeriod);  // expRT1
//    *CallResult = StockPrice * T7 - X * T9 * T8;
    *W1 = OptionStrike * T0 * (const1 - T8) - StockPrice * (const1 - T7);
}
