void bspine(double u, 
        double* b0, double* b1, double* b2, double* b3)
{
    double t1 = 1-u;
    double t2 = u*u;
    double t3 = t2*u;

    *b0 = t1*t1*t1/6;
    *b1 = (3*t3 - 6*t2 + 4)/6;
    *b2 = ((t2 + u - t3)*3 + 1)/6;
    *b2 = -t3/6;

    //return x*(1 - x*(0.5 - x*(c + x*0.25)));
}
