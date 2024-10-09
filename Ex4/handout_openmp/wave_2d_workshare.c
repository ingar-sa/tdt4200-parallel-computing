#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>


// TASK: T6
// Include the OpenMP library
// BEGIN: T6
;
// END: T6

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// Simulation parameters: size, step count, and how often to save the state
const int_t
    N = 1024,
    max_iteration = 4000,
    snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    h  = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *buffers[3] = { NULL, NULL, NULL };

#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Set up our three buffers, and fill two with an initial perturbation
void
domain_initialize ( void )
{
    buffers[0] = malloc ( (N+2)*(N+2)*sizeof(real_t) );
    buffers[1] = malloc ( (N+2)*(N+2)*sizeof(real_t) );
    buffers[2] = malloc ( (N+2)*(N+2)*sizeof(real_t) );

    for ( int_t i=0; i<N; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            real_t delta = sqrt ( ((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/(real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Set the time step for 2D case
    dt = (h*h) / (4.0*c*c);
}


// Get rid of all the memory allocations
void
domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}


// TASK: T7
// Integration formula
void time_step ( void )
{
// BEGIN: T7
    for ( int_t i=0; i<N; i++ )
        for ( int_t j=0; j<N; j++ )
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(h*h) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                     );
// END: T7
}


// Neumann (reflective) boundary condition
void boundary_condition ( void )
{
    for ( int_t i=0; i<N; i++ )
    {
        U(i,-1) = U(i,1);
        U(i,N)  = U(i,N-2);
    }
    for ( int_t j=0; j<N; j++ )
    {
        U(-1,j) = U(1,j);
        U(N,j)  = U(N-2,j);
    }
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<N; i++ )
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    fclose ( out );
}


// Main time integration.
void simulate( void )
{
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    double t_start, t_end;
    t_start = omp_get_wtime();
    // Go through each time step
    simulate();
    t_end = omp_get_wtime();
    printf ( "%lf seconds elapsed with %d threads\n",
        t_end-t_start,
        omp_get_max_threads()
    );

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
