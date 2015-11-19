//void rgb(unsigned int R, unsigned int G, unsigned int B, 
void rgb(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, 
        double a, double b, double c, double d, double e, double f, 
        double g, 
        double* y0, double* y1, double* y2, double* y3, double* y4, double* y5, double* y6, double* y7)
{
    double T1 = x0+x7;
    double T2 = x1+x6;
    double T3 = x2+x5;
    double T4 = x3+x4;
    double T5 = x0-x7;
    double T6 = x1-x6;
    double T7 = x2-x5;
    double T8 = x3-x4;

    *y0 = (T1+T2+T3+T4)*g;
    *y2 = b*(T1-T4) + e*(T2-T3);
    *y4 = ((T1+T4) - (T2+T3))*g;
    *y6 = e*(T1-T4) - b*(T2-T3);

    *y1 = a*T5 - c*T6 + d*T7 - f*T8;
    *y3 = c*T5 + f*T6 - a*T7 + d*T8;
    *y5 = d*T5 + a*T6 + f*T7 - c*T8;
    *y7 = f*T5 + d*T6 + c*T7 + a*T8;

    //*y0 = (x0+x1+x2+x3+x4+x5+x6+x7)/c0;
    //*y1 = c1*(x0-x7) + c3*( x1-x6) + c4 *( x2-x5) + c6*( x3-x4);
    //*y2 = c2*(x0+x7-x3-x4) + c5*(x1+x6-x2-x5);
    //*y3 = c3*(x0-x7) + c6*( x1-x6) + c1*( x2-x5) + c4*( x3-x4);

    //*y4 = (x0+x7+x3+x4 - x1-x2-x5-x6)/c0;
    //*y5 = d*(x0-x7) - a*( x1-x6) + f*( x2-x5) + c*( x3-x4);
    //*y6 = e*(x0+x7-x3-x4) - b*(x1+x6-x2-x5);
    //*y7 = f*(x0-x7) - d*( x1-x6) + c*( x2-x5) - a*( x3-x4);
}
