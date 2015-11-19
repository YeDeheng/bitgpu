iir8(double a0, 
	 double a1, 
	 double a2, 
	 double a3, 
	 double a4, 
	 double a5, 
	 double a6, 
	 double a7, 
	 double b0, 
	 double b1, 
	 double b2, 
	 double b3, 
	 double b4, 
	 double b5, 
	 double b6, 
	 double x0, 
	 double x1, 
	 double x2, 
	 double x3, 
	 double x4, 
	 double x5, 
	 double x6, 
	 double x7, 
	 double y0, 
	 double y1, 
	 double y2, 
	 double y3, 
	 double y4, 
	 double y5, 
	 double y6, 
	 double y7
	 )
{
	y0 = a0*x0;
	y1 = a1*x1 + y0 - b0*y0;
	y2 = a2*x2 + y1 - b1*y1;
	y3 = a3*x3 + y2 - b2*y2;
	y4 = a4*x4 + y3 - b3*y3;
	y5 = a5*x5 + y4 - b4*y4;
	y6 = a6*x6 + y5 - b5*y5;
	y7 = a7*x7 + y6 - b6*y6;
}