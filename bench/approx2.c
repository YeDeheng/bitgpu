#include<math.h>
double approx2(double x1, double x2, double c1, double c2) 
{
    double T1 = log(1+exp(c1*x1));
    double T2 = log(1+exp(c1*x2));

    return c2*(T1*T1-T2*T2);
}
