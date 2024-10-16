#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define STEPS (1e8)
#define H (1.0/STEPS)

// All variables are declared here
double
    pi,
    pi_local,
    x;
int
    tid,
    n_threads;

int
main ( int argc, char **argv )
{
    pi = 0.0;

    // We can explicitly specify data sharing here instead
    #pragma omp parallel shared(pi) private(pi_local,x,tid,n_threads)
    {
        tid = omp_get_thread_num(),
        n_threads = omp_get_num_threads();
        x = tid * H;
        pi_local = 0.0;
        for ( int64_t i=0; i<STEPS; i+=n_threads )
        {
            x += n_threads*H;
            pi_local += H / (1.0 + x*x);
        }

        // Avoid race conditions for the global pi value
        #pragma omp critical
        {
            pi += pi_local;
        }
    }

    // We're not parallel anymore, print the result and clean up
    pi *= 4.0;
    printf ( "Estimated %e, missed by %e\n", pi, fabs(pi-M_PI) );
    exit ( EXIT_SUCCESS );
}
