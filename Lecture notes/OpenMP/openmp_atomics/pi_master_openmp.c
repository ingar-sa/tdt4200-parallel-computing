#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define STEPS (1e8)
#define H (1.0/STEPS)

// Global values, accessible by all threads
double
    pi,
    *pi_parts;

int
main ( int argc, char **argv )
{
    pi_parts = malloc ( omp_get_max_threads() * sizeof(double) );

    // Estimate pi in parallel
    #pragma omp parallel
    {
        int
            tid = omp_get_thread_num(),
            n_threads = omp_get_num_threads();
        double
            x = tid * H;
        pi_parts[tid] = 0.0;
        for ( size_t i=0; i<STEPS; i+=n_threads )
        {
            x += n_threads*H;
            pi_parts[tid] += H / (1.0 + x*x);
        }

        // Make sure that everyone's estimate is finished
        #pragma omp barrier

        // Master thread alone adds up the total
        #pragma omp master
        {
            pi = 0.0;
            for ( int i=0; i<n_threads; i++ )
                pi += pi_parts[tid];
        }
    }

    // We're not parallel anymore, print the result and clean up
    pi *= 4.0;
    printf ( "Estimated %e, missed by %e\n", pi, fabs(pi-M_PI) );
    free ( pi_parts );
    exit ( EXIT_SUCCESS );
}
