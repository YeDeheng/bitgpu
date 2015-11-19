// Refer to Deriche Filter wikipedia. 
// only consider one Channel. 
void deriche(double x1, 
            double x2, 
            double x3, 
            double x4, 
            double x5, 
            double x6, 
            double x7, 
            double x8, 

            double y1, 
            double y2, 
            double y3, 
            double y4, 
            double y5, 
            double y6, 
            double y7, 
            double y8, 

            double z1, 
            double z2, 
            double z3, 
            double z4, 
            double z5, 
            double z6, 
            double z7, 
            double z8, 

            double w1, 
            double w2, 
            double w3, 
            double w4, 
            double w5, 
            double w6, 
            double w7, 
            double w8,

            double h1, 
            double h2, 
            double h3, 
            double h4, 

            double c1, 
            double c2, 

            double a1, 
            double a2, 
            double a3, 
            double a4, 
            double a5, 
            double a6, 
            double a7, 
            double a8,

            double b1, 
            double b2, 
            
            double *sum)
{
    double T0 = c1 * ( a1*x1 + a2*x2 + b1*y1 + b2*y2 + a3*x3 + a4*x4 + b1*y3 + b2*y4 ); 
    double T1 = c1 * ( a1*x5 + a2*x6 + b1*y5 + b2*y6 + a3*x7 + a4*x8 + b1*y7 + b2*y8 ); 
    double T2 = c1 * ( a1*z1 + a2*z2 + b1*w1 + b2*w2 + a3*z3 + a4*z4 + b1*w3 + b2*w4 ); 
    double T3 = c1 * ( a1*z5 + a2*z6 + b1*w5 + b2*w6 + a3*z7 + a4*z8 + b1*w7 + b2*w8 ); 

    double T4 = a5*T0 + a6*T1 + b1*h1 + b2*h2; 
    double T5 = a7*T2 + a8*T3 + b1*h3 + b2*h4; 
    *sum = c2*(T4 + T5);
}
