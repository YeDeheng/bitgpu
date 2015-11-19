double level1_linear(double x1, double x2, double c1, double c2) 
{
	double T1 = x1 - x2 - c1;
	return c2*T1*T1;
}
