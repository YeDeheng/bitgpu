double poly(double x, double c) {
	return x*(1 - x*(0.5 - x*(c + x*0.25)));
}
