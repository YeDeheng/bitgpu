#ifndef MC_RANGE_HOST_H_
#define MC_RANGE_HOST_H_

#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>

struct GenRand
{
    REAL a, b;
    GenRand(REAL _a=0., REAL _b=1.) : a(_a), b(_b) {};
    REAL operator () (int threadid)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<REAL> uniDist(a, b);
        return uniDist(gen);
    }
};

void get_mc_samples(REAL x0, REAL x1, std::vector<REAL> *sample_vector) {
    int sample_num = sample_vector->size() - 2;
    std::transform(
            sample_vector->begin(),
            sample_vector->end(),
            sample_vector->begin(),
            GenRand(x0, x1));
    (*sample_vector)[sample_num] = x0;
    (*sample_vector)[sample_num+1] = x1;
}

__forceinline__ void sqrt_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<0 || x1<0)
        return;
		
    // create samples between [x0, x1]
    int randNum = abs(x1 - x0);
    std::vector<REAL> rVec(randNum + 2);
    get_mc_samples(x0, x1, &rVec);
    for(int i = 0; i < rVec.size(); i++)
        rVec[i] = sqrt(rVec[i]);
    int min_pos = std::min_element(rVec.begin(), rVec.end()) - rVec.begin();
    int max_pos = std::max_element(rVec.begin(), rVec.end()) - rVec.begin();
    *ret0 = rVec[min_pos];
    *ret1 = rVec[max_pos];
}
__forceinline__ void log_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    if(x0<=0 || x1<=0)
        return;
    // create samples between [x0, x1]
    int randNum = abs(x1 - x0);
    std::vector<REAL> rVec(randNum + 2);
    get_mc_samples(x0, x1, &rVec);
    for(int i = 0; i < rVec.size(); i++)
        rVec[i] = log(rVec[i]);
    int min_pos = std::min_element(rVec.begin(), rVec.end()) - rVec.begin();
    int max_pos = std::max_element(rVec.begin(), rVec.end()) - rVec.begin();
    *ret0 = rVec[min_pos];
    *ret1 = rVec[max_pos];

}

__forceinline__ void exp_rangerule(REAL x0, REAL x1, 
        REAL* ret0, REAL* ret1) {
    // create samples between [x0, x1]
    int randNum = abs(x1 - x0);
    std::vector<REAL> rVec(randNum + 2);
    get_mc_samples(x0, x1, &rVec);
    for(int i = 0; i < rVec.size(); i++)
        rVec[i] = exp(rVec[i]);
    int min_pos = std::min_element(rVec.begin(), rVec.end()) - rVec.begin();
    int max_pos = std::max_element(rVec.begin(), rVec.end()) - rVec.begin();
    *ret0 = rVec[min_pos];
    *ret1 = rVec[max_pos];
}

__forceinline__ void div_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    if(y0*y1<=0) // value of divisor could not include 0 
        return;
    // create samples between [x0, x1]	
    std::vector<REAL> rVec1(abs(x1 - x0) + 2);
    get_mc_samples(x0, x1, &rVec1);
    std::vector<REAL> rVec2(abs(y1 - y0) + 2);
    get_mc_samples(y0, y1, &rVec2);
    std::vector<REAL> rVec3(rVec1.size() * rVec2.size());
    for(int i = 0; i < rVec1.size(); i++) {
        for(int j = 0; j < rVec2.size(); j++) {
            rVec3[i * rVec2.size() + j] = rVec1[i] / rVec2[j];
        }
    }
    int min_pos = std::min_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    int max_pos = std::max_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    *ret0 = rVec3[min_pos];
    *ret1 = rVec3[max_pos];
}

__forceinline__ void mult_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    std::vector<REAL> rVec1(abs(x1 - x0) + 2);
    get_mc_samples(x0, x1, &rVec1);
    std::vector<REAL> rVec2(abs(y1 - y0) + 2);
    get_mc_samples(y0, y1, &rVec2);
    std::vector<REAL> rVec3(rVec1.size() * rVec2.size());
    for(int i = 0; i < rVec1.size(); i++) {
        for(int j = 0; j < rVec2.size(); j++) {
            rVec3[i * rVec2.size() + j] = rVec1[i] * rVec2[j];
	    }
    }

    int min_pos = std::min_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    int max_pos = std::max_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    *ret0 = rVec3[min_pos];
    *ret1 = rVec3[max_pos];
}

__forceinline__ void add_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    std::vector<REAL> rVec1(abs(x1 - x0) + 2);
    get_mc_samples(x0, x1, &rVec1);
    std::vector<REAL> rVec2(abs(y1 - y0) + 2);
    get_mc_samples(y0, y1, &rVec2);
    std::vector<REAL> rVec3(rVec1.size() * rVec2.size());
    for(int i = 0; i < rVec1.size(); i++) {
        for(int j = 0; j < rVec2.size(); j++) {
            rVec3[i * rVec2.size() + j] = rVec1[i] + rVec2[j];
	    }
    }
    int min_pos = std::min_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    int max_pos = std::max_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    *ret0 = rVec3[min_pos];
    *ret1 = rVec3[max_pos];
}

__forceinline__ void sub_rangerule(REAL x0, REAL x1, 
        REAL y0, REAL y1, 
        REAL* ret0, REAL* ret1) {
    std::vector<REAL> rVec1(abs(x1 - x0) + 2);
    get_mc_samples(x0, x1, &rVec1);
    std::vector<REAL> rVec2(abs(y1 - y0) + 2);
    get_mc_samples(y0, y1, &rVec2);
    std::vector<REAL> rVec3(rVec1.size() * rVec2.size());
    for(int i = 0; i < rVec1.size(); i++) {
        for(int j = 0; j < rVec2.size(); j++) {
            rVec3[i * rVec2.size() + j] = rVec1[i] * rVec2[j];
	    }
    }
    int min_pos = std::min_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    int max_pos = std::max_element(rVec3.begin(), rVec3.end()) - rVec3.begin();
    *ret0 = rVec3[min_pos];
    *ret1 = rVec3[max_pos];
}

__forceinline__ int integer_bit_calc(REAL low_bound, REAL high_bound)
{
    int lo = abs(low_bound);
    int hi = abs(high_bound);
    int randNum = abs(hi - lo);
    std::vector<REAL> rVec(randNum + 2);
    get_mc_samples(lo, hi, &rVec);
    for(int i = 0; i < rVec.size(); i++)
        rVec[i] = ceil(log2(rVec[i] + 1));
    int max_pos = std::max_element(rVec.begin(), rVec.end()) - rVec.begin();
    int integer_bit = (int)(rVec[max_pos]);

    return integer_bit;
} 
#endif
