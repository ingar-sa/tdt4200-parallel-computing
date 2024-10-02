#define _XOPEN_SOURCE 600

// NOTE(ingar): isa.h is a personal library that has some functionality I find useful,
// primarily the logging. Which logging calls that are compiled can be set by changing the
// ISA_LOG_LEVEL define. The levels are: 0, no logging; 1, only errors; 2, errors and warnings; 3,
// errors, warnings, and info; 4, the previous and debug
#ifndef ISA_LOG_LEVEL
#define ISA_LOG_LEVEL 4
#endif

#include "isa.h"

ISA_LOG_REGISTER(MPI2DWaveEquation);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"
#include "datatypes.h"
#include "mpi_utils.h"

// TASK: T1a
// Include the MPI hederfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
// Buffers for three time steps, indexed with 2 ghost points for the boundary

#define UPrev(i, j) time_steps.prev_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define UCurr(i, j) time_steps.curr_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define UNext(i, j) time_steps.next_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]

// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
// END: T1b

static MpiCtx mpi_ctx = {};

// END: T1b

// Simulation parameters: size, step count, and how often to save the state
// Wave equation parameters, time step is derived from the space step
// Rotate the time step buffers.

static SimParams sim_params
    = { .M = 256, .N = 256, .max_iteration = 4000, .snapshot_frequency = 20 };

static WaveEquationParams wave_equation_params = { .c = 1.0, .dx = 1.0, .dy = 1.0 };

static TimeSteps time_steps = {};

static void
rotate_buffers(void)
{
    f64 *prev_step       = time_steps.prev_step;
    time_steps.prev_step = time_steps.curr_step;
    time_steps.curr_step = time_steps.next_step;
    time_steps.next_step = prev_step;
}

// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void
initialize_domain(void)
{
    // BEGIN: T4
    i64 M  = sim_params.M;
    i64 N  = sim_params.N;
    f64 c  = wave_equation_params.c;
    f64 dx = wave_equation_params.dx;
    f64 dy = wave_equation_params.dy;

    time_steps.prev_step = malloc((M + 2) * (N + 2) * sizeof(f64));
    time_steps.curr_step = malloc((M + 2) * (N + 2) * sizeof(f64));
    time_steps.next_step = malloc((M + 2) * (N + 2) * sizeof(f64));

    for(i64 i = 0; i < M; i++) {
        for(i64 j = 0; j < N; j++) {
            // Calculate delta (radial distance) adjusted for M x N grid
            f64 delta   = sqrt(((i - M / 2.0) * (i - M / 2.0)) / (f64)M
                               + ((j - N / 2.0) * (j - N / 2.0)) / (f64)N);
            UPrev(i, j) = UCurr(i, j) = exp(-4.0 * delta * delta);
        }
    }

    // Set the time step for 2D case
    wave_equation_params.dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
    // END: T4
}

// Get rid of all the memory allocations
void
finalize_domain(void)
{
    free(time_steps.prev_step);
    free(time_steps.curr_step);
    free(time_steps.next_step);
}

// TASK: T5
// Integration formula
void
perform_time_step(void)
{
    i64 m  = sim_params.M;
    i64 n  = sim_params.N;
    f64 c  = wave_equation_params.c;
    f64 dx = wave_equation_params.dx;
    f64 dy = wave_equation_params.dy;
    f64 dt = wave_equation_params.dt;

    // BEGIN: T5
    for(i64 i = 0; i < m; i++) {
        for(i64 j = 0; j < n; j++) {
            UNext(i, j) = -UPrev(i, j) + 2.0 * UCurr(i, j)
                        + (dt * dt * c * c) / (dx * dy)
                              * (UCurr(i - 1, j) + UCurr(i + 1, j) + UCurr(i, j - 1)
                                 + UCurr(i, j + 1) - 4.0 * UCurr(i, j));
        }
    }
    // END: T5
}

// TASK: T6
// Communicate the border between processes.
void
perform_border_exchange(void)
{
    // BEGIN: T6
    ;
    // END: T6
}

// TASK: T7
// Neumann (reflective) boundary condition
void
perform_boundary_condition(void)
{
    i64 M = sim_params.M;
    i64 N = sim_params.N;

    // BEGIN: T7
    for(i64 i = 0; i < M; i++) {
        UCurr(i, -1) = UCurr(i, 1);
        UCurr(i, N)  = UCurr(i, N - 2);
    }
    for(i64 j = 0; j < N; j++) {
        UCurr(-1, j) = UCurr(1, j);
        UCurr(M, j)  = UCurr(M - 2, j);
    }
    // END: T7
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
void
save_domain(i64 step)
{
    i64 M = sim_params.M;
    i64 N = sim_params.N;

    // BEGIN: T8
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    for(i64 i = 0; i < M; i++) {
        fwrite(&UCurr(i, 0), sizeof(f64), N, out);
    }
    fclose(out);
    // END: T8
}

// Main time integration.
void
simulate(void)
{
    i64 n_time_steps       = sim_params.max_iteration;
    i64 snapshot_frequency = sim_params.snapshot_frequency;
    // Go through each time step
    for(i64 iteration = 0; iteration <= n_time_steps; iteration++) {
        if((iteration % snapshot_frequency) == 0) {
            save_domain(iteration / snapshot_frequency);
        }

        // Derive step t+1 from steps t and t-1
        perform_border_exchange();
        perform_boundary_condition();
        perform_time_step();

        // Rotate the time step buffers
        rotate_buffers();
    }
}

int
main(int argc, char **argv)
{
    // TASK: T1c
    // Initialise MPI
    // BEGIN: T1c

    int commsize, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // END: T1c

    // TASK: T3
    // Distribute the user arguments to all the processes
    // BEGIN: T3

    mpi_ctx.my_rank         = my_rank;
    mpi_ctx.commsize        = commsize;
    mpi_ctx.n_children      = commsize - 1;
    mpi_ctx.i_am_root_rank  = (mpi_ctx.my_rank == 0);
    mpi_ctx.i_am_first_rank = (mpi_ctx.my_rank == 1);
    mpi_ctx.i_am_last_rank  = (mpi_ctx.my_rank == (commsize - 1));
    mpi_ctx.i_am_only_child = (mpi_ctx.i_am_first_rank && mpi_ctx.i_am_last_rank);

    size_t send_buf_size = 4 * sizeof(i64);
    void  *send_buffer   = malloc(send_buf_size);
    if(mpi_ctx.i_am_root_rank) {
        OPTIONS *options = parse_args(argc, argv);
        if(!options) {
            fprintf(stderr, "Argument parsing failed\n");
            exit(EXIT_FAILURE);
        }

        sim_params.M                  = options->M;
        sim_params.N                  = options->N;
        sim_params.max_iteration      = options->max_iteration;
        sim_params.snapshot_frequency = options->snapshot_frequency;

        int buffer_pos = 0;
        MPI_Pack(&sim_params.M, 1, MPI_INT64_T, send_buffer, send_buf_size, &buffer_pos,
                 MPI_COMM_WORLD);
        MPI_Pack(&sim_params.N, 1, MPI_INT64_T, send_buffer, send_buf_size, &buffer_pos,
                 MPI_COMM_WORLD);
        MPI_Pack(&sim_params.max_iteration, 1, MPI_INT64_T, send_buffer, send_buf_size, &buffer_pos,
                 MPI_COMM_WORLD);
        MPI_Pack(&sim_params.snapshot_frequency, 1, MPI_INT64_T, send_buffer, send_buf_size,
                 &buffer_pos, MPI_COMM_WORLD);
    }

    MPI_Bcast(send_buffer, send_buf_size, MPI_PACKED, 0, MPI_COMM_WORLD);

    if(!mpi_ctx.i_am_root_rank) {
        int buffer_pos = 0;
        MPI_Unpack(send_buffer, send_buf_size, &buffer_pos, &sim_params.M, 1, MPI_INT64_T,
                   MPI_COMM_WORLD);
        MPI_Unpack(send_buffer, send_buf_size, &buffer_pos, &sim_params.N, 1, MPI_INT64_T,
                   MPI_COMM_WORLD);
        MPI_Unpack(send_buffer, send_buf_size, &buffer_pos, &sim_params.max_iteration, 1,
                   MPI_INT64_T, MPI_COMM_WORLD);
        MPI_Unpack(send_buffer, send_buf_size, &buffer_pos, &sim_params.snapshot_frequency, 1,
                   MPI_INT64_T, MPI_COMM_WORLD);
    }
    free(send_buffer);

    IsaLogDebug(
        "Rank %ld has sim_params:\n M=%ld\n N=%ld\n max_iteration=%ld\n snapshot_frequency=%ld\n",
        mpi_ctx.my_rank, sim_params.M, sim_params.N, sim_params.max_iteration,
        sim_params.snapshot_frequency);

    int      n_cart_dims    = 2;
    int      cart_dims[2]   = { 0 };
    int      periodicity[2] = { 0 };
    int      reorder        = 0;
    MPI_Comm cart_comm;
    // TODO(ingar): Set up cartesian communicator

    // END: T3

    // Set up the initial state of the domain
    // initialize_domain();

    // TASK: T2
    // Time your code
    // BEGIN: T2

    // struct timeval t_start, t_end;
    // I think it's appropriate to use MPI's timing functionality in an MPI program
    f64 time_start = 0.0;
    f64 time_end   = 0.0;

    if(mpi_ctx.i_am_root_rank) {
        time_start = MPI_Wtime();
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // simulate();
    // MPI_Barrier(MPI_COMM_WORLD);

    if(mpi_ctx.i_am_root_rank) {
        time_end = MPI_Wtime();
        printf("Simulation time: %f\n", time_end - time_start);
    }

    // END: T2

    // Clean up and shut down
    // finalize_domain();

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d

    MPI_Finalize();

    // END: T1d

    exit(EXIT_SUCCESS);
}
