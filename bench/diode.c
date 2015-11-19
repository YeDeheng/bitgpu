#include<math.h>
double diode(double x1, double x2, double x3, double* y) 
{
    *y = x3*(exp(x1*x2)-1) ;
}
