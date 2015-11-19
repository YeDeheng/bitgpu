double poly6(double x, double c1, double c2, double c3) {
	return x*(1 - x*(0.5 - x*(c1 - x*(0.25 - x*(c2 - x*c3)))));
}
