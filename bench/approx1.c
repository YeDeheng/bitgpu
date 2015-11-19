double approx1(double x1, double x2, double x3, double c1, double c2)
{
    // vg:x1, vs:x2, vd:x3
    double T1 = x1 - c1;
    double T2 = T1 - x2;
    double T3 = T1 - x3;
    return c2*(T2*T2 - T3*T3);
}

