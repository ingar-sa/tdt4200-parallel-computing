#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define ITERATIONS 10
#define STEPS (1e8)
#define H (1.0/STEPS)


double pi = 0.0;
omp_lock_t lock;


int
main ( int argc, char **argv )
{
    omp_init_lock ( &lock );
    #pragma omp parallel
    {
        int
            tid = omp_get_thread_num(),
            n_threads = omp_get_num_threads();

        for ( size_t iter=0; iter<ITERATIONS; iter++ )
        {
            double x = tid*H, pi_local = 0.0;
            for ( size_t i=0; i<STEPS; i+=n_threads )
            {
                x += n_threads*H;
                pi_local += H / (1.0 + x*x);
            }

            omp_set_lock ( &lock );
            pi += pi_local;
            omp_unset_lock ( &lock );
            #pragma omp barrier

            if ( tid == 0 )
            {
                pi *= 4.0;
                printf ( "Estimated %e, missed by %e\n", pi, fabs(pi-M_PI) );
                pi = 0.0;
            }
            #pragma omp barrier
        }
    }
    omp_destroy_lock ( &lock );
    exit ( EXIT_SUCCESS );
}
