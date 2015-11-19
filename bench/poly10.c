double poly10(double x, double c1, double c2, double c3, double c4, double c5, double c6) {
	return x*(1 - x*(0.5 - x*(c1 - x*(0.25 - x*(c2 - x*(c3 - x*(c4 - x*(0.125 - x*(c5 - x*c6)))))))));
}
