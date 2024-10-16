#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Typedefs for floats and integers
typedef double real_t;
typedef int64_t int_t;

int_t
    N             = 256,        // Number of points in each direction
    max_iter      = 65536,      // Maximum number of iterations
    snapshot_freq = 128;        // How often to save the simulation state

real_t
    *u[2] = { NULL, NULL },     // Two buffers for displacement (now and next)
    vy,                         // Propagation speeds for the wave,
    vx,                         //  in y and x directions
    dx,                         // Space between two points in our domain
    dt,                         // Time between simulation steps
    x_range[2] = { -1.0, 1.0 }; // "Physical" extent of the domain

/* Convenience macros for indexing:
 *  we want to allocate 1 additional boundary point beyond the domain,
 *  in 2D it becomes a whole additional row/column at each end
 */
#define U(i,j)      u[0][((i)+1)*(N+2)+(j)+1]
#define U_next(i,j) u[1][((i)+1)*(N+2)+(j)+1]


int
main ( int argc, char **argv )
{
    // Allocate 2 extra elements for the displacement arrays, so that we
    //  don't have to make exceptions for first and last points in the domain
    u[0] = malloc ( (N+2) * (N+2) * sizeof(real_t) );
    u[1] = malloc ( (N+2) * (N+2) * sizeof(real_t) );

    // Work out the simulation parameters
    dx = (x_range[1]-x_range[0]) / (real_t) N;
    vx = 1.0;
    vy = 0.5;
    dt = dx/(4.0*sqrt(vy*vy+vx*vx));    // Careful time step value

    // Initial state: start with a big disturbance at one end of the domain
    for ( int_t i=0; i<N; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            real_t
                y = x_range[0] + i * dx,
                x = j * dx;
            real_t
                norm = sqrt ( x*x + y*y );
            U(i,j) = pow(exp(-pow(norm,2.0)),4.0);
        }
    }

    // Record the time spent in our parallel region
    double
        t_timesteps = 0.0,
        t_start,
        t_end;
    // Time integration loop
    for ( int_t iter=0; iter<max_iter; iter++ )
    {

        // Periodic boundary condition
        for ( int_t i=0; i<N; i++ )
        {
            U(i,-1) = U(i,N-1);
            U(i,N)  = U(i,0);
        }
        for ( int_t j=0; j<N; j++ )
        {
            U(-1,j) = U(N-1,j);
            U(N,j)  = U(0,j);
        }

        // Start the clock
        t_start = omp_get_wtime();
        // Solve for next time step, using Lax-Friedrichs approximation
        #pragma omp parallel for
        for ( int_t i=0; i<N; i++ )
        {
            for ( int_t j=0; j<N; j++ )
            {
                U_next(i,j) = 0.25 * (U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1))
                  - vy * (dt/(2.0*dx)) * (U(i+1,j) - U(i-1,j))
                  - vx * (dt/(2.0*dx)) * (U(i,j+1) - U(i,j-1));
            }
        }
        // Stop the clock and accumulate total time
        t_end = omp_get_wtime();
        t_timesteps += (t_end - t_start);

        if ( (iter % snapshot_freq) == 0 )
        {
            int_t step = iter / snapshot_freq;
            char filename[256];
            memset ( filename, 0, 256*sizeof(char) );
            sprintf ( filename, "data/%.5ld.dat", step );
            FILE *out = fopen ( filename, "w" );
            for ( int_t i=0; i<N; i++ )
                fwrite ( &U(i,0), sizeof(real_t), N, out );
            fclose ( out );
            printf ( "Saved state at iteration %ld\n", iter );
        }

        // Swap the now-step with the next-step and continue
        real_t *tmp = u[0];
        u[0] = u[1];
        u[1] = tmp;
    }

    // Print out the total recorded integration time
    printf ( "%lf seconds of parallelized calculus\n", t_timesteps );

    // Clean up and go home
    free ( u[0] );
    free ( u[1] );
    exit ( EXIT_SUCCESS );
}
