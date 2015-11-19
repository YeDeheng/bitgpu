#include "ap_fixed.h"
#include "ap_int.h"
#include "math.h"

#define data_t ap_fixed<80,10>

void exponential(data_t a,  data_t *c)
{
    *c = exp(a);
}
    
