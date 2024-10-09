#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DO_DEBUG 1
#if DO_DEBUG
#define LogDebug( ... ) printf ( __VA_ARGS__ )
#else
#define LogDebug( ... )
#endif

#include "argument_utils.h"
#include "datatypes.h"

// TASK: T1a
// Include the MPI hederfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

// Convert 'struct timeval' into seconds in double prec. floating point
// NOTE: I use MPI's timing functionality instead
// #define WALLTIME( t ) ( (double)( t ).tv_sec + 1e-6 * (double)( t ).tv_usec )

#define U_prv( i, j ) time_steps.prev_step[( ( i ) + 1 ) * ( mpi_ctx.N + 2 ) + ( j ) + 1]
#define U( i, j )     time_steps.curr_step[( ( i ) + 1 ) * ( mpi_ctx.N + 2 ) + ( j ) + 1]
#define U_nxt( i, j ) time_steps.next_step[( ( i ) + 1 ) * ( mpi_ctx.N + 2 ) + ( j ) + 1]

// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
// END: T1b

static MpiCtx mpi_ctx = {};

// END: T1b

static SimParams          sim_params           = { 512, 512, 4000, 20 };
static WaveEquationParams wave_equation_params = { .c = 1.0, .dx = 1.0, .dy = 1.0 };
static TimeSteps          time_steps           = {};

// Rotate the time step buffers.
static void
move_buffer_window ( void )
{
    real_t *prev_step    = time_steps.prev_step;
    time_steps.prev_step = time_steps.curr_step;
    time_steps.curr_step = time_steps.next_step;
    time_steps.next_step = prev_step;
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
static void
domain_save ( int_t step )
{
    // BEGIN: T8

    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    MPI_File out;
    MPI_File_open ( mpi_ctx.cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                    &out );

    int global_grid_dims[2] = { sim_params.M, sim_params.N };
    int local_grid_dims[2]  = { mpi_ctx.M, mpi_ctx.N };
    int local_coords[2]     = { mpi_ctx.y * mpi_ctx.M, mpi_ctx.x * mpi_ctx.N };

    if ( step == 0 ) {
        LogDebug ( "Rank (%ld, %ld): global grid (%d, %d), local grid (%d, %d), "
                   "local coords (%d, %d)\n",
                   mpi_ctx.y, mpi_ctx.x, global_grid_dims[0], global_grid_dims[1],
                   local_grid_dims[0], local_grid_dims[1], local_coords[0], local_coords[1] );
    }

    MPI_Datatype my_area;
    MPI_Type_create_subarray ( 2, global_grid_dims, local_grid_dims, local_coords, MPI_ORDER_C,
                               MPI_DOUBLE, &my_area );
    MPI_Type_commit ( &my_area );

    MPI_File_set_view ( out, 0, MPI_DOUBLE, my_area, "native", MPI_INFO_NULL );
    MPI_File_write_all ( out, &U ( 0, 0 ), 1, mpi_ctx.MpiGrid, MPI_STATUS_IGNORE );

    MPI_File_close ( &out );

    // END: T8
}

static void
find_neighbors ( int *north, int *south, int *east, int *west )
{
    MPI_Cart_shift ( mpi_ctx.cart_comm, 0, 1, north, south );
    MPI_Cart_shift ( mpi_ctx.cart_comm, 1, 1, west, east );
}

// TASK: T6
// Communicate the border between processes.
static void
border_exchange ( void )
{
    // BEGIN: T6
    int north, south, east, west;
    find_neighbors ( &north, &south, &east, &west );

    // Send top row to north, receive top row from south in bottom ghost row
    MPI_Sendrecv ( &U ( 0, 0 ), 1, mpi_ctx.MpiRow, north, 0, &U ( mpi_ctx.M, 0 ), 1, mpi_ctx.MpiRow,
                   south, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE );

    // Send bottom row to south, receive bottom row from north in top ghost row
    MPI_Sendrecv ( &U ( mpi_ctx.M - 1, 0 ), 1, mpi_ctx.MpiRow, south, 0, &U ( -1, 0 ), 1,
                   mpi_ctx.MpiRow, north, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE );

    // Send right col to east, reveive right row from west into left ghost col
    MPI_Sendrecv ( &U ( 0, mpi_ctx.N - 1 ), 1, mpi_ctx.MpiCol, east, 0, &U ( 0, -1 ), 1,
                   mpi_ctx.MpiCol, west, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE );

    // Send left col to west, receive left col from east into right ghost col
    MPI_Sendrecv ( &U ( 0, 0 ), 1, mpi_ctx.MpiCol, west, 0, &U ( 0, mpi_ctx.N ), 1, mpi_ctx.MpiCol,
                   east, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE );

    //  END: T6
}

// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
static void
domain_initialize ( void )
{
    // BEGIN: T4
    size_t alloc_size = ( mpi_ctx.M + 2 ) * ( mpi_ctx.N + 2 ) * sizeof ( real_t );
    LogDebug ( "Allocating %zd bytes for each timestep\n", alloc_size );

    time_steps.prev_step = malloc ( alloc_size );
    time_steps.curr_step = malloc ( alloc_size );
    time_steps.next_step = malloc ( alloc_size );

    real_t c        = wave_equation_params.c;
    real_t dx       = wave_equation_params.dx;
    real_t dy       = wave_equation_params.dy;
    int_t  M_offset = mpi_ctx.M * mpi_ctx.y;
    int_t  N_offset = mpi_ctx.N * mpi_ctx.x;
    int_t  global_M = sim_params.M;
    int_t  global_N = sim_params.N;
    LogDebug ( "Rank (%ld, %ld) has offsets M(%ld) N(%ld)\n", mpi_ctx.y, mpi_ctx.x, M_offset,
               N_offset );

    for ( int_t i = 0; i < mpi_ctx.M; i++ ) {
        for ( int_t j = 0; j < mpi_ctx.N; j++ ) {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta
                = sqrt ( ( ( i + M_offset - global_M / 2.0 ) * ( i + M_offset - global_M / 2.0 ) )
                             / (real_t)global_M
                         + ( ( j + N_offset - global_N / 2.0 ) * ( j + N_offset - global_N / 2.0 ) )
                               / (real_t)global_N );

            U_prv ( i, j ) = U ( i, j ) = exp ( -4.0 * delta * delta );
        }
    }

    // Set the time step for 2D case
    wave_equation_params.dt = dx * dy / ( c * sqrt ( dx * dx + dy * dy ) );
    // END: T4
}

// Get rid of all the memory allocations
static void
domain_finalize ( void )
{
    free ( time_steps.prev_step );
    free ( time_steps.curr_step );
    free ( time_steps.next_step );
}

// TASK: T5
// Integration formula
static void
time_step ( void )
{
    int_t  M  = mpi_ctx.M;
    int_t  N  = mpi_ctx.N;
    real_t c  = wave_equation_params.c;
    real_t dx = wave_equation_params.dx;
    real_t dy = wave_equation_params.dy;
    real_t dt = wave_equation_params.dt;

    // BEGIN: T5
    for ( int_t i = 0; i < M; i++ ) {
        for ( int_t j = 0; j < N; j++ ) {
            U_nxt ( i, j ) = -U_prv ( i, j ) + 2.0 * U ( i, j )
                           + ( dt * dt * c * c ) / ( dx * dy )
                                 * ( U ( i - 1, j ) + U ( i + 1, j ) + U ( i, j - 1 )
                                     + U ( i, j + 1 ) - 4.0 * U ( i, j ) );
        }
    }
    // END: T5
}

// TASK: T7
// Neumann (reflective) boundary condition
static void
boundary_condition ( void )
{
    int_t M = mpi_ctx.M;
    int_t N = mpi_ctx.N;

    // BEGIN: T7

    for ( int_t i = 0; i < M; i++ ) {
        if ( mpi_ctx.x == 0 ) {
            U ( i, -1 ) = U ( i, 1 );
        }
        if ( mpi_ctx.x == ( mpi_ctx.cart_cols - 1 ) ) {
            U ( i, N ) = U ( i, N - 2 );
        }
    }
    for ( int_t j = 0; j < N; j++ ) {
        if ( mpi_ctx.y == 0 ) {
            U ( -1, j ) = U ( 1, j );
        }
        if ( mpi_ctx.y == ( mpi_ctx.cart_rows - 1 ) ) {
            U ( M, j ) = U ( M - 2, j );
        }
    }
    // END: T7
}

// Main time integration.
static void
simulate ( void )
{
    int_t max_iteration      = sim_params.max_iteration;
    int_t snapshot_frequency = sim_params.snapshot_frequency;

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ ) {
        if ( ( iteration % snapshot_frequency ) == 0 ) {
            domain_save ( iteration / snapshot_frequency );
        }

        border_exchange ();
        if ( mpi_ctx.on_boundary ) {
            boundary_condition ();
        }
        time_step ();
        move_buffer_window ();
    }
}

static void
mpi_types_free ( void )
{
    MPI_Type_free ( &mpi_ctx.MpiCol );
    MPI_Type_free ( &mpi_ctx.MpiRow );
    MPI_Type_free ( &mpi_ctx.MpiGrid );
}

static bool
on_boundary ( void )
{
    bool on_boundary
        = ( ( mpi_ctx.y == 0 ) || ( mpi_ctx.x == 0 ) || ( mpi_ctx.y == ( mpi_ctx.cart_rows - 1 ) )
            || ( mpi_ctx.x == ( mpi_ctx.cart_cols - 1 ) ) );

    return on_boundary;
}

static void
mpi_ctx_initialize ( int argc, char **argv )
{
    size_t param_send_buf_size = 4 * sizeof ( int_t );
    void  *param_send_buffer   = malloc ( param_send_buf_size );
    if ( mpi_ctx.rank == 0 ) {
        OPTIONS *options = parse_args ( argc, argv );
        if ( !options ) {
            fprintf ( stderr, "Argument parsing failed\n" );
            exit ( EXIT_FAILURE );
        }

        sim_params.M                  = options->M;
        sim_params.N                  = options->N;
        sim_params.max_iteration      = options->max_iteration;
        sim_params.snapshot_frequency = options->snapshot_frequency;

        int buffer_pos = 0;
        MPI_Pack ( &sim_params.M, 1, MPI_INT64_T, param_send_buffer, param_send_buf_size,
                   &buffer_pos, MPI_COMM_WORLD );
        MPI_Pack ( &sim_params.N, 1, MPI_INT64_T, param_send_buffer, param_send_buf_size,
                   &buffer_pos, MPI_COMM_WORLD );
        MPI_Pack ( &sim_params.max_iteration, 1, MPI_INT64_T, param_send_buffer,
                   param_send_buf_size, &buffer_pos, MPI_COMM_WORLD );
        MPI_Pack ( &sim_params.snapshot_frequency, 1, MPI_INT64_T, param_send_buffer,
                   param_send_buf_size, &buffer_pos, MPI_COMM_WORLD );
    }

    MPI_Bcast ( param_send_buffer, param_send_buf_size, MPI_PACKED, 0, MPI_COMM_WORLD );

    if ( !( mpi_ctx.rank == 0 ) ) {
        int buffer_pos = 0;
        MPI_Unpack ( param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.M, 1,
                     MPI_INT64_T, MPI_COMM_WORLD );
        MPI_Unpack ( param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.N, 1,
                     MPI_INT64_T, MPI_COMM_WORLD );
        MPI_Unpack ( param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.max_iteration,
                     1, MPI_INT64_T, MPI_COMM_WORLD );
        MPI_Unpack ( param_send_buffer, param_send_buf_size, &buffer_pos,
                     &sim_params.snapshot_frequency, 1, MPI_INT64_T, MPI_COMM_WORLD );
    }
    free ( param_send_buffer );

    LogDebug ( "Rank %ld has sim_params:\n M=%ld\n N=%ld\n max_iteration=%ld\n "
               "snapshot_frequency=%ld\n",
               mpi_ctx.rank, sim_params.M, sim_params.N, sim_params.max_iteration,
               sim_params.snapshot_frequency );

    int      n_cart_dims  = 2;
    int      cart_dims[2] = { 0 };
    int      periods[2]   = { 0 };
    int      reorder      = 0;
    MPI_Comm cart_comm;
    MPI_Dims_create ( mpi_ctx.commsize, n_cart_dims, cart_dims );
    MPI_Cart_create ( MPI_COMM_WORLD, n_cart_dims, cart_dims, periods, reorder, &cart_comm );

    int cart_rank;
    int coords[2];
    MPI_Comm_rank ( cart_comm, &cart_rank );
    MPI_Cart_coords ( cart_comm, cart_rank, n_cart_dims, coords );

    mpi_ctx.rank        = cart_rank;
    mpi_ctx.cart_rows   = cart_dims[0];
    mpi_ctx.cart_cols   = cart_dims[1];
    mpi_ctx.y           = coords[0];
    mpi_ctx.x           = coords[1];
    mpi_ctx.cart_comm   = cart_comm;
    mpi_ctx.on_boundary = on_boundary ();
    mpi_ctx.M           = sim_params.M / mpi_ctx.cart_rows;
    mpi_ctx.N           = sim_params.N / mpi_ctx.cart_cols;

    MPI_Datatype MpiCol;
    MPI_Type_vector ( mpi_ctx.M, 1, mpi_ctx.N + 2, MPI_DOUBLE, &MpiCol );
    MPI_Type_commit ( &MpiCol );
    mpi_ctx.MpiCol = MpiCol;

    MPI_Datatype MpiRow;
    MPI_Type_contiguous ( mpi_ctx.N, MPI_DOUBLE, &MpiRow );
    MPI_Type_commit ( &MpiRow );
    mpi_ctx.MpiRow = MpiRow;

    MPI_Datatype MpiGrid;
    MPI_Type_vector ( mpi_ctx.M, mpi_ctx.N, mpi_ctx.N + 2, MPI_DOUBLE, &MpiGrid );
    MPI_Type_commit ( &MpiGrid );
    mpi_ctx.MpiGrid = MpiGrid;

    LogDebug ( "Process %ld in MPI_COMM_WORLD is now process %d in cart_comm "
               "with coordinates "
               "(%ld, %ld) is %son the boundary\n",
               mpi_ctx.rank, cart_rank, mpi_ctx.y, mpi_ctx.x, mpi_ctx.on_boundary ? "" : "not " );
}

int
main ( int argc, char **argv )
{
    // TASK: T1c
    // Initialise MPI
    // BEGIN: T1c

    int commsize, my_rank;
    MPI_Init ( &argc, &argv );
    MPI_Comm_size ( MPI_COMM_WORLD, &commsize );
    MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );

    // END: T1c

    // TASK: T3
    // Distribute the user arguments to all the processes
    // BEGIN: T3
    mpi_ctx.rank     = my_rank;
    mpi_ctx.commsize = commsize;
    mpi_ctx_initialize ( argc, argv );

    // END: T3

    // Set up the initial state of the domain
    domain_initialize ();

    // TASK: T2
    // Time your code
    // BEGIN: T2

    double time_start = 0.0;
    double time_end   = 0.0;

    MPI_Barrier ( MPI_COMM_WORLD );
    if ( mpi_ctx.rank == 0 ) {
        time_start = MPI_Wtime ();
    }

    simulate ();
    MPI_Barrier ( MPI_COMM_WORLD );

    if ( mpi_ctx.rank == 0 ) {
        time_end = MPI_Wtime ();
        printf ( "Simulation time: %f\n", time_end - time_start );
    }

    // END: T2

    // Clean up and shut down
    domain_finalize ();
    mpi_types_free ();
    MPI_Comm_free ( &mpi_ctx.cart_comm );

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Finalize ();

    // END: T1d

    exit ( EXIT_SUCCESS );
}
