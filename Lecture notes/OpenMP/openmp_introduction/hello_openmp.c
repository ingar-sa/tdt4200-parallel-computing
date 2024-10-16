#include <stdio.h>
#include <stdlib.h>

// OpenMP run time library functions
#include <omp.h>


int
main ()
{
    int n_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    printf ( "Hello, I am thread %d of %d\n", tid, n_threads );

    printf ( "Let us go momentarily parallel...\n" );
    #pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        printf ( "Hello, I am thread %d of %d\n", tid, n_threads );
    }

    printf ( "...aaaand we're back to normal:\n" );
    n_threads = omp_get_num_threads();
    tid = omp_get_thread_num();
    printf ( "Hello, I am thread %d of %d\n", tid, n_threads );

    exit ( EXIT_SUCCESS );
}
