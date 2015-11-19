//void rgb(unsigned int R, unsigned int G, unsigned int B, 
void rgb(double R, double G, double B, 
        double c1, double c2, double c3, 
        double c4, double c5, //double c6, 
        //double c7, 
        double c8, double c9, 
        double* Y, double* Cb, double* Cr) {
    *Y = c1*R + c2*G + c3*B;
    *Cb = c4*R + c5*G + B*0.5;
    *Cr = 0.5*R + c8*G + c9*B;

	//return x*(1 - x*(0.5 - x*(c + x*0.25)));
}
