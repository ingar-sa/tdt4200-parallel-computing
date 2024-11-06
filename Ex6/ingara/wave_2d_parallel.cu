// #define _XOPEN_SOURCE 600
// I get a compiler warning that the macro is already defined so i've commented
// it out

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
//  END: T1

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME( t ) ( (double)( t ).tv_sec + 1e-6 * (double)( t ).tv_usec )

// Option to change numerical precision
typedef int64_t int_t;
typedef float   real_t; // I have been unable to pass make check if I use double,
                        // even though the implementation works with float.

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

// Based on the output from the device info, max threads per block is 1024,
// which is 32x32 This could be determined dynamically in init_cuda, but I don't
// see the point since we know which GPUs we're using
#define BLOCKY 32
#define BLOCKX 32

// Simulation parameters: size, step count, and how often to save the state
int_t h_N = 128, h_M = 128, h_max_iteration = 1e6, h_snapshot_freq = 1e3;
#define SIM_DATA_SIZE ( ( h_M + 2 ) * ( h_N + 2 ) * sizeof ( real_t ) )
// I forgot to multiply by sizeof(real_t) one time, so I made this macro to
// avoid that mistake in the future

// Wave equation parameters, time step is derived from the space step
const real_t h_c = 1.0, h_dx = 1.0, h_dy = 1.0;
real_t       h_dt;

// Since the simulation and equation parameters are constant throughout the
// simulation, we can keep them in the device's constant memory for fast access
// by all threads. The h_ equivalents are copied to these in h_domain_initialize
__constant__ int_t  d_N, d_M;
__constant__ real_t d_c, d_dx, d_dy, d_dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
// I've decided to make a wrapper struct to make it more clear which timestep is
// being used. The struct points the device memory allocated with cudaMalloc,
// not host memory, but I've prepended it with h_ since it is used by the host
// and not the device
typedef struct Timesteps
{
    real_t *prv;
    real_t *cur;
    real_t *nxt;

} Timesteps;
Timesteps h_timesteps;

// Buffer for the host to copy device memory into when writing the results to
// file and an access macro for it
real_t *h_out;
#define h_U( i, j ) h_out[( ( i ) + 1 ) * ( h_N + 2 ) + ( j ) + 1]

// I have changed the macros to work with buffers being passed in to functions
// instead of being accessed globally. This is because I couldn't figure out how
// to have a globally accessible Timesteps struct on the device, so the host
// passes in the individual buffers to kernel calls as arguments, and the kernel
// will pass them along to any device functions that need them
#define U_prv( i, j ) prv[( ( i ) + 1 ) * ( d_N + 2 ) + ( j ) + 1]
#define U( i, j )     cur[( ( i ) + 1 ) * ( d_N + 2 ) + ( j ) + 1]
#define U_nxt( i, j ) nxt[( ( i ) + 1 ) * ( d_N + 2 ) + ( j ) + 1]

// Used to convert bytes to the more readable kibi-, mebi-, and gibibytes
#define BYTES_TO_KiB( bytes ) ( ( bytes ) / 1024.0 )
#define BYTES_TO_MiB( bytes ) ( ( bytes ) / ( 1024.0 * 1024.0 ) )
#define BYTES_TO_GiB( bytes ) ( ( bytes ) / ( 1024.0 * 1024.0 * 1024.0 ) )

// END: T1b

// I didn't end up using these
#define cudaErrorCheck( ans )                                                                      \
    {                                                                                              \
        gpuAssert ( ( ans ), __FILE__, __LINE__ );                                                 \
    }

inline void
gpuAssert ( cudaError_t code, const char *file, int line, bool abort = true )
{
    if ( code != cudaSuccess ) {
        fprintf ( stderr, "GPUassert: %s %s %d\n", cudaGetErrorString ( code ), file, line );
        if ( abort ) {
            exit ( code );
        }
    }
}

// Save the present time step in a numbered file under 'data/'
void
h_domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i = 0; i < h_M; i++ ) {
        fwrite ( &h_U ( i, 0 ), sizeof ( real_t ), h_N, out );
    }
    fclose ( out );
}

// TASK: T4
// Get rid of all the memory allocations
void
h_domain_finalize ( void )
{
    // BEGIN: T4
    free ( h_out );
    cudaFree ( h_timesteps.prv );
    cudaFree ( h_timesteps.cur );
    cudaFree ( h_timesteps.nxt );
    // END: T4
}

// Rotate the time step buffers.
void
h_move_buffer_window ( void )
{
    real_t *temp    = h_timesteps.prv;
    h_timesteps.prv = h_timesteps.cur;
    h_timesteps.cur = h_timesteps.nxt;
    h_timesteps.nxt = temp;
}

// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
__device__ void
d_boundary_condition ( int t_i, int t_j, real_t *cur )
{
    if ( t_j == 0 ) {
        U ( t_i, -1 ) = U ( t_i, 1 );
    }

    if ( t_j == ( d_N - 1 ) ) {
        U ( t_i, d_N ) = U ( t_i, d_N - 2 );
    }

    if ( t_i == 0 ) {
        U ( -1, t_j ) = U ( 1, t_j );
    }

    if ( t_i == ( d_M - 1 ) ) {
        U ( d_M, t_j ) = U ( d_M - 2, t_j );
    }
}
// END: T6

// TASK: T5
// Integration formula

__global__ void
d_time_step ( real_t *prv, real_t *cur, real_t *nxt )
{
    // BEGIN; T5
    cg::thread_block tb = cg::this_thread_block ();

    dim3 b_idx = tb.group_index ();
    dim3 b_dim = tb.group_dim ();
    dim3 t_idx = tb.thread_index ();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    d_boundary_condition ( t_i, t_j, cur );

    // Make sure all boundary conditions have been applied before calculating the
    // timestep
    tb.sync ();

    U_nxt ( t_i, t_j ) = -U_prv ( t_i, t_j ) + 2.0 * U ( t_i, t_j )
                       + ( d_dt * d_dt * d_c * d_c ) / ( d_dx * d_dy )
                             * ( U ( t_i - 1, t_j ) + U ( t_i + 1, t_j ) + U ( t_i, t_j - 1 )
                                 + U ( t_i, t_j + 1 ) - 4.0 * U ( t_i, t_j ) );
    // END: T5
}

// TASK: T7
// Main time integration.
void
h_simulate ( void )
{
    // BEGIN: T7
    // Go through each time step
    int_t grid_x = h_N / BLOCKX;
    int_t grid_y = h_M / BLOCKY;
    dim3  block ( BLOCKX, BLOCKY );
    dim3  grid ( grid_x, grid_y );

    // Originally, I wanted to perform all of the simulation steps in a loop on
    // the GPU, i.e. boundary condition, time step, and move buffer window, only
    // exiting device execution after snapshot_freq iterations to write the data
    // to file, but I couldn't get it to work, unfortunately.
    for ( int_t iteration = 0; iteration <= h_max_iteration; iteration += 1 ) {
        if ( ( iteration % h_snapshot_freq ) == 0 ) {
            cudaMemcpy ( h_out, h_timesteps.cur, SIM_DATA_SIZE, cudaMemcpyDeviceToHost );
            h_domain_save ( iteration / h_snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        d_time_step<<<grid, block>>> ( h_timesteps.prv, h_timesteps.cur, h_timesteps.nxt );

        // Rotate the time step buffers
        h_move_buffer_window ();
    }
    // END: T7
}

// TASK: T8
// GPU occupancy
void
h_occupancy ( void )
{
    // BEGIN: T8
    int block_size;
    int min_grid_size;
    int max_active_blocks;

    cudaOccupancyMaxPotentialBlockSize ( &min_grid_size, &block_size, d_time_step, 0, 0 );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &max_active_blocks, d_time_step, block_size,
                                                    0 );

    int            device;
    cudaDeviceProp props;
    cudaGetDevice ( &device );
    cudaGetDeviceProperties ( &props, device );

    int   grid_size = ( h_M + 2 ) * ( h_N + 2 );
    float occupancy = ( (float)max_active_blocks * block_size / props.warpSize )
                    / ( (float)props.maxThreadsPerMultiProcessor / props.warpSize );

    printf ( "\nGrid size set to %d\nMin grid size is %d\nBlock size is %d\nMax "
             "active blocks is "
             "%d\nDevice's max thread count per SM is %d\nDevice's warp size is "
             "%d\nTheoretical "
             "occupancy: %f\n",
             grid_size, min_grid_size, block_size, max_active_blocks,
             props.maxThreadsPerMultiProcessor, props.warpSize, occupancy );
    // END: T8
}

// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool
h_init_cuda ( void )
{
    // BEGIN: T2
    int dev_count;
    cudaGetDeviceCount ( &dev_count );
    printf ( "Device count: %d\n", dev_count );

    if ( dev_count <= 0 ) {
        return false;
    } else {
        cudaSetDevice ( 0 );
    }

    cudaDeviceProp dev_props;
    for ( int i = 0; i < dev_count; ++i ) {
        cudaError_t ret = cudaGetDeviceProperties ( &dev_props, i );
        if ( ret == cudaErrorInvalidDevice ) {
            return false;
        }

        printf ( "\nName: %s\n", dev_props.name );
        printf ( "Compute capability: %d.%d\n", dev_props.major, dev_props.minor );
        printf ( "Multiprocessors: %d\n", dev_props.multiProcessorCount );
        printf ( "Warp size: %d\n", dev_props.warpSize );
        printf ( "Global memory: %.3fGiB\n", BYTES_TO_GiB ( dev_props.totalGlobalMem ) );
        printf ( "Per-block shared memory: %.3fKiB\n",
                 BYTES_TO_KiB ( dev_props.sharedMemPerBlock ) );
        printf ( "Per-block registers: %d\n", dev_props.regsPerBlock );
        printf ( "\nMax threads per block: %d\n", dev_props.maxThreadsPerBlock );
        printf ( "Max threads dim: %d x %d x %d\n", dev_props.maxThreadsDim[0],
                 dev_props.maxThreadsDim[1], dev_props.maxThreadsDim[2] );
        printf ( "Max gid size: %d x %d x %d\n", dev_props.maxGridSize[0], dev_props.maxGridSize[1],
                 dev_props.maxGridSize[2] );
    }

    return true;
    // END: T2
}

// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
__global__ void
d_init_timesteps ( real_t *prv, real_t *cur )
{
    // Calculating the initial values on the device, instead of on the host and
    // then copying it to device memory, should be faster (though it might not
    // necessarily be, but the necessary profiling to find out is outside the
    // scope of the exercise, in my opinion)
    cg::thread_block tb = cg::this_thread_block ();

    dim3 b_idx = tb.group_index ();
    dim3 b_dim = tb.group_dim ();
    dim3 t_idx = tb.thread_index ();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    real_t delta = sqrt ( ( ( t_i - d_M / 2.0 ) * ( t_i - d_M / 2.0 ) ) / (real_t)d_M
                          + ( ( t_j - d_N / 2.0 ) * ( t_j - d_N / 2.0 ) ) / (real_t)d_N );

    real_t val         = exp ( -4.0 * delta * delta );
    U_prv ( t_i, t_j ) = U ( t_i, t_j ) = val;
}

void
h_domain_initialize ( void )
{
    // BEGIN: T3

    bool locate_cuda = h_init_cuda ();
    if ( !locate_cuda ) {
        printf ( "Failed to init CUDA\n" );
        exit ( EXIT_FAILURE );
    }

    // Buffer on the host for writing the results to file
    h_out = (real_t *)malloc ( SIM_DATA_SIZE );

    cudaMalloc ( (void **)&h_timesteps.prv, SIM_DATA_SIZE );
    cudaMalloc ( (void **)&h_timesteps.cur, SIM_DATA_SIZE );
    cudaMalloc ( (void **)&h_timesteps.nxt, SIM_DATA_SIZE );

    // Set the time step for 2D case
    h_dt = h_dx * h_dy / ( h_c * sqrt ( h_dx * h_dx + h_dy * h_dy ) );

    // Copy all relevant values for the simulation and equation to the device
    cudaMemcpyToSymbol ( d_N, &h_N, sizeof ( int_t ) );
    cudaMemcpyToSymbol ( d_M, &h_M, sizeof ( int_t ) );
    cudaMemcpyToSymbol ( d_c, &h_c, sizeof ( real_t ) );
    cudaMemcpyToSymbol ( d_dx, &h_dx, sizeof ( real_t ) );
    cudaMemcpyToSymbol ( d_dy, &h_dy, sizeof ( real_t ) );
    cudaMemcpyToSymbol ( d_dt, &h_dt, sizeof ( real_t ) );

    int_t grid_x = h_N / BLOCKX;
    int_t grid_y = h_M / BLOCKY;

    dim3 grid ( grid_x, grid_y );
    dim3 block ( BLOCKX, BLOCKY );
    d_init_timesteps<<<grid, block>>> ( h_timesteps.prv, h_timesteps.cur );

    // END: T3
}

int
main ( void )
{
    // Set up the initial state of the domain
    h_domain_initialize ();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    h_simulate ();

    gettimeofday ( &t_end, NULL );

    printf ( "\nTotal elapsed time: %lf seconds\n", WALLTIME ( t_end ) - WALLTIME ( t_start ) );

    h_occupancy ();

    // Clean up and shut down
    h_domain_finalize ();
    exit ( EXIT_SUCCESS );
}
