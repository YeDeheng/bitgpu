#ifndef TIMER_H
#define TIMER_H

#include "time.h"

#define TIMER struct timespec

void record_time(TIMER *time) {
	clock_gettime(CLOCK_REALTIME, time);
}


float calculate_time(TIMER *time1, TIMER *time2)
{
	return (float)(( time2->tv_sec - time1->tv_sec ) + ( time2->tv_nsec - time1->tv_nsec ) / 1E9);
}

#endif
