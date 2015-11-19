#ifndef ERROR_H_
#define ERROR_H_


/* error models */
__device__ void sqrt_errrule(REAL& x0, REAL& x1, REAL& e1, const REAL &e3, REAL* rete) 
{
    *rete = e1 / (sqrt(x0) + sqrt(x0+e1)) + e3;
}
__device__ void exp_errrule(REAL& x0, REAL& x1, REAL& e1, const REAL &e3, REAL* rete) 
{
    *rete = exp(x1) * (exp(e1) - 1) + e3;
}
__device__ void log_errrule(REAL& x0, REAL& x1, REAL& e1, const REAL& e3, REAL* rete) 
{
    if(x0==0)
        return;
    *rete = log(1 + e1/x0) + e3;
}
__device__ void div_errrule(REAL& x0, REAL& x1, REAL& e1, REAL& y0, REAL& y1, REAL& e2, REAL& e3, REAL* rete) 
{
    REAL rep_x = (fabs(x1) > fabs(x0)) ? fabs(x1) : fabs(x0); 
    REAL rep_y = (fabs(y1) > fabs(y0)) ? fabs(y1) : fabs(y0); 
    *rete = e3 + (rep_y*e1 + rep_x*e2) / (rep_y*e2 + rep_y*rep_y);
}
__device__ void mult_errrule(REAL& x0, REAL& x1, REAL& e1, REAL& y0, REAL& y1, REAL& e2, REAL& e3, REAL* rete) 
{
    REAL main_err = ((fabs(x1) > fabs(x0)) ? fabs(x1) : fabs(x0))*e2 + ((fabs(y1) > fabs(y0)) ? fabs(y1) : fabs(y0))*e1; 

    //if (e1==0 || e2==0)
    //    *rete = main_err;
    //else 
    //    *rete = main_err + e3 + e1*e2;

    if(fabs(x0-x1)<1e-30 && fabs(y0-y1)<1e-30)
    {
        REAL z = x1*y1;
        //REAL z_shift = z*pow(2,t);
        REAL z_shift = 0.5*z/e3;
        double intpart;
        REAL fractpart = (REAL)modf((double)z_shift, &intpart);
        if(fractpart)
            *rete = main_err + e3;
        else 
            *rete = main_err;
    }
    else if (fabs(x0-x1)<1e-30 && fabs(y0-y1)>1e-30)
    {
        double intpart;
        REAL fracpart = (REAL)modf((double)x1, &intpart);
        if(fracpart) // not an integer
            *rete = main_err + e3;
        else  // it is integer
        {
            int x_int = (int)x1; 
            while (((x_int % 2) == 0) && x_int > 1) // detect power of 2
                x_int /= 2;
            if(x_int == 1) // if it is a power of 2
            {
                int error_ratio = e3/e2;
                if(error_ratio <= x_int)
                    *rete = main_err;
                else 
                    *rete = main_err + e3;
            }
        }
    }
    else if (fabs(x0-x1)>1e-30 && fabs(y0-y1)<1e-30)
    {
        double intpart;
        REAL fracpart = (REAL)modf((double)y1, &intpart);
        if(fracpart) // not an integer
            *rete = main_err + e3;
        else  // it is integer
        {
            int y_int = (int)y1; 
            while (((y_int % 2) == 0) && y_int > 1) // detect power of 2
                y_int /= 2;
            if(y_int == 1) // if it is a power of 2
            {
                int error_ratio = e3/e1;
                if(error_ratio <= y_int)
                    *rete = main_err;
                else 
                    *rete = main_err + e3;
            }
        }
    }
    else
    {
        *rete = main_err + e3;
    }
}

__device__ void add_errrule(REAL& x0, REAL& x1, REAL& e1, REAL& y0, REAL& y1, REAL& e2, REAL& e3, REAL* rete) 
{
    REAL max_e1_e2 = (e1 >= e2) ? e1 : e2;
    if (e3 > max_e1_e2)
        *rete = e3 + e1 + e2;
    else 
        *rete = e2 + e1;
}

__device__ void sub_errrule(REAL& x0, REAL& x1, REAL& e1, REAL& y0, REAL& y1, REAL& e2, REAL& e3, REAL* rete) 
{
    REAL max_e1_e2 = (e1 >= e2) ? e1 : e2;
    if (e3 > max_e1_e2)
        *rete = e3 + e1 + e2;
    else 
        *rete = e2 + e1;
}

#endif
