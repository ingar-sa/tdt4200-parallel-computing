#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define STEPS (1e8)
#define H (1.0/STEPS)


int
main ( int argc, char **argv )
{
    double pi = 0.0;

    #pragma omp parallel for reduction(+:pi)
    for ( int64_t i=0; i<STEPS; i++ )
        pi += H / (1.0 + (i*H)*(i*H));

    pi *= 4.0;
    printf ( "Estimated %e, missed by %e\n", pi, fabs(pi-M_PI) );
    exit ( EXIT_SUCCESS );
}
