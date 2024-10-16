#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define STEPS (1e8)
#define H (1.0/STEPS)

// Global values, accessible by all threads
omp_lock_t lock;
double pi = 0.0;

int
main ( int argc, char **argv )
{
    // Make a lock
    omp_init_lock ( &lock );

    // Estimate pi in parallel
    #pragma omp parallel
    {
        int
            tid = omp_get_thread_num(),
            n_threads = omp_get_num_threads();
        double
            x = tid * H,
            pi_local = 0.0;
        for ( size_t i=0; i<STEPS; i+=n_threads )
        {
            x += n_threads*H;
            pi_local += H / (1.0 + x*x);
        }

        // Avoid race conditions for the global pi value
        omp_set_lock ( &lock );
        pi += pi_local;
        omp_unset_lock ( &lock );
    }

    // We're not parallel anymore, print the result and clean up
    pi *= 4.0;
    printf ( "Estimated %e, missed by %e\n", pi, fabs(pi-M_PI) );
    omp_destroy_lock ( &lock );
    exit ( EXIT_SUCCESS );
}
