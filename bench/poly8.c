double poly8(double x, double c1, double c2, double c3, double c4) {
	return x*(1 - x*(0.5 - x*(c1 - x*(0.25 - x*(c2 - x*(c3 - x*(c4 - x*0.125 )))))));
}
