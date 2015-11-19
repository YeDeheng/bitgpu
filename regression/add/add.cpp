#include "ap_fixed.h"
#include "ap_int.h"
#include "math.h"

#define data_t ap_fixed<80,10>

void add(data_t a, data_t b, data_t *c)
{
    *c = a + b;
}
    
