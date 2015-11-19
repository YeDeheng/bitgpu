// I tried to write a complete Gaussian filter, but we cannot support complex C program. I have to simplify it by pre-defining all the coefficients. 
//
// Assume 3x3 pixels, and only consider one channel of RGB.
void gaussian(double r0, double r1, double r2, 
            double r3, double r4, double r5, 
            double r6, double r7, double r8, 
            double c0, double c1, double c2, 
            double c3, double c4, double c5, 
            double c6, double c7, double c8, 
            double *sum)
{
    double T0 = r0*c0; 
    double T1 = r1*c1; 
    double T2 = r2*c2; 
    double T3 = r3*c3; 
    double T4 = r4*c4; 
    double T5 = r5*c5; 
    double T6 = r6*c6; 
    double T7 = r7*c7; 
    double T8 = r8*c8; 

    *sum = T0+T1+T2+T3+T4+T5+T6+T7+T8;
}
