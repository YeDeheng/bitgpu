double level1_saturation(double x1, //vd
        double x2, //vg
        double x3, //vs
        double c1, //vt
        double c2) //b
{
    return c2*(x2-x3-c1-(x1-x3)*0.5)*(x1-x3);
}

