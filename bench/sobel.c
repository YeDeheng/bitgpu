//Note: inputs to Sobel are all unsigned char data type, they are the pixels of input image. Here, we use 'double' instead. 
double sobel(double ul, // upper left
        double um, // upper middle
        double ur, // upper right
        double ml, // middle left
        double mr, // middle right
        double ll, // lower left
        double lm, // lower middle
        double lr, // lower right
        double fscale)
{
    double T1 = ur + 2*mr + lr - ul - 2*ml - ll;
    double T2 = ul + 2*um + ur - ll - 2*lm - lr;
    return fscale*(T1+T2);
}
