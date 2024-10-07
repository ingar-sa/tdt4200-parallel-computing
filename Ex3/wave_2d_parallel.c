#define _XOPEN_SOURCE 600
// TODO(ingar): Change f64 and i64 to real_t and int_t at the end

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
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"
#include "datatypes.h"

// TASK: T1a
// Include the MPI hederfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
// Buffers for three time steps, indexed with 2 ghost points for the boundary

#define UPrev(i, j) time_steps.prev_step[((i) + 1) * (mpi_ctx.N + 2) + (j) + 1]
#define UCurr(i, j) time_steps.curr_step[((i) + 1) * (mpi_ctx.N + 2) + (j) + 1]
#define UNext(i, j) time_steps.next_step[((i) + 1) * (mpi_ctx.N + 2) + (j) + 1]

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
    = {}; //= { .M = 512, .N = 512, .max_iteration = 4000, .snapshot_frequency = 20 };

static WaveEquationParams wave_equation_params = { .c = 1.0, .dx = 1.0, .dy = 1.0 };

static TimeSteps time_steps = {};

static void
move_buffer_window(void)
{
    f64 *prev_step       = time_steps.prev_step;
    time_steps.prev_step = time_steps.curr_step;
    time_steps.curr_step = time_steps.next_step;
    time_steps.next_step = prev_step;
}

// TASK: T8
// Save the present time step in a numbered file under 'data/'
void
domain_save(i64 step)
{
    // BEGIN: T8

    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    MPI_File out;
    MPI_File_open(mpi_ctx.cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                  &out);

    int global_grid_dims[2] = { sim_params.M, sim_params.N };
    int local_grid_dims[2]  = { mpi_ctx.M, mpi_ctx.N };
    int local_coords[2]     = { mpi_ctx.y * mpi_ctx.M, mpi_ctx.x * mpi_ctx.N };

    IsaLogDebug("Rank (%ld, %ld): global grid (%d, %d), local grid (%d, %d), local coords (%d, %d)",
                mpi_ctx.y, mpi_ctx.x, global_grid_dims[0], global_grid_dims[1], local_grid_dims[0],
                local_grid_dims[1], local_coords[0], local_coords[1]);

    MPI_Datatype my_area;
    MPI_Type_create_subarray(2, global_grid_dims, local_grid_dims, local_coords, MPI_ORDER_C,
                             MPI_DOUBLE, &my_area);
    MPI_Type_commit(&my_area);

    MPI_File_set_view(out, 0, MPI_DOUBLE, my_area, "native", MPI_INFO_NULL);
    MPI_Barrier(mpi_ctx.cart_comm);
    MPI_File_write_all(out, &UCurr(0, 0), 1, mpi_ctx.MpiGrid, MPI_STATUS_IGNORE);

    MPI_File_close(&out);
    if(mpi_ctx.i_am_root_rank && (step % 100) == 0) {
        IsaLogDebug("Saved to file %s", filename);
    }

    // END: T8
}

void
find_neighbors(int *north, int *south, int *east, int *west)
{
    MPI_Cart_shift(mpi_ctx.cart_comm, 0, 1, north, south);
    MPI_Cart_shift(mpi_ctx.cart_comm, 1, 1, west, east);
}

// TASK: T6
// Communicate the border between processes.
void
border_exchange(void)
{
    // BEGIN: T6
    int north, south, east, west;
    find_neighbors(&north, &south, &east, &west);

    // Send top row to north, receive top row from south in bottom ghost row
    MPI_Sendrecv(&UCurr(0, 0), 1, mpi_ctx.MpiRow, north, 0, &UCurr(mpi_ctx.M, 0), 1, mpi_ctx.MpiRow,
                 south, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE);

    // Send bottom row to south, receive bottom row from north in top ghost row
    MPI_Sendrecv(&UCurr(mpi_ctx.M - 1, 0), 1, mpi_ctx.MpiRow, south, 0, &UCurr(-1, 0), 1,
                 mpi_ctx.MpiRow, north, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE);

    // Send right col to east, reveive right row from west into left ghost col
    MPI_Sendrecv(&UCurr(0, mpi_ctx.N - 1), 1, mpi_ctx.MpiCol, east, 0, &UCurr(0, -1), 1,
                 mpi_ctx.MpiCol, west, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE);

    // Send left col to west, receive left col from east into right ghost col
    MPI_Sendrecv(&UCurr(0, 0), 1, mpi_ctx.MpiCol, west, 0, &UCurr(0, mpi_ctx.N), 1, mpi_ctx.MpiCol,
                 east, 0, mpi_ctx.cart_comm, MPI_STATUS_IGNORE);

    //  END: T6
}

// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void
domain_initialize(void)
{
    // BEGIN: T4
    f64 c  = wave_equation_params.c;
    f64 dx = wave_equation_params.dx;
    f64 dy = wave_equation_params.dy;

    size_t alloc_size = (mpi_ctx.M + 2) * (mpi_ctx.N + 2) * sizeof(f64);
    IsaLogDebug("Allocating %zd bytes for each timestep", alloc_size);

    time_steps.prev_step = malloc(alloc_size);
    time_steps.curr_step = malloc(alloc_size);
    time_steps.next_step = malloc(alloc_size);

    i64 M_offset = mpi_ctx.M * mpi_ctx.y;
    i64 N_offset = mpi_ctx.N * mpi_ctx.x;

    IsaLogDebug("Rank (%ld, %ld) has offsets M(%ld) N(%ld)", mpi_ctx.y, mpi_ctx.x, M_offset,
                N_offset);

    for(i64 i = 0; i < mpi_ctx.M; i++) {
        for(i64 j = 0; j < mpi_ctx.N; j++) {
            // Calculate delta (radial distance) adjusted for M x N grid
            f64 delta
                = sqrt(((i + M_offset - sim_params.M / 2.0) * (i + M_offset - sim_params.M / 2.0))
                           / (f64)sim_params.M
                       + ((j + N_offset - sim_params.N / 2.0) * (j + N_offset - sim_params.N / 2.0))
                             / (f64)sim_params.N);

            UPrev(i, j) = UCurr(i, j) = exp(-4.0 * delta * delta);
        }
    }

    // Set the time step for 2D case
    wave_equation_params.dt = dx * dy / (c * sqrt(dx * dx + dy * dy));
    // END: T4
}

// Get rid of all the memory allocations
void
domain_finalize(void)
{
    free(time_steps.prev_step);
    free(time_steps.curr_step);
    free(time_steps.next_step);
}

// TASK: T5
// Integration formula
void
time_step(void)
{
    i64 M  = mpi_ctx.M;
    i64 N  = mpi_ctx.N;
    f64 c  = wave_equation_params.c;
    f64 dx = wave_equation_params.dx;
    f64 dy = wave_equation_params.dy;
    f64 dt = wave_equation_params.dt;

    // BEGIN: T5
    for(i64 i = 0; i < M; i++) {
        for(i64 j = 0; j < N; j++) {
            UNext(i, j) = -UPrev(i, j) + 2.0 * UCurr(i, j)
                        + (dt * dt * c * c) / (dx * dy)
                              * (UCurr(i - 1, j) + UCurr(i + 1, j) + UCurr(i, j - 1)
                                 + UCurr(i, j + 1) - 4.0 * UCurr(i, j));
        }
    }
    // END: T5
}

// TASK: T7
// Neumann (reflective) boundary condition
void
boundary_condition(void)
{
    i64 M = mpi_ctx.M;
    i64 N = mpi_ctx.N;

    // BEGIN: T7

    for(i64 i = 0; i < M; i++) {
        if(mpi_ctx.x == 0) {
            UCurr(i, -1) = UCurr(i, 1);
        }
        if(mpi_ctx.x == (mpi_ctx.cart_cols - 1)) {
            UCurr(i, N) = UCurr(i, N - 2);
        }
    }
    for(i64 j = 0; j < N; j++) {
        if(mpi_ctx.y == 0) {
            UCurr(-1, j) = UCurr(1, j);
        }
        if(mpi_ctx.y == (mpi_ctx.cart_rows - 1)) {
            UCurr(M, j) = UCurr(M - 2, j);
        }
    }
    // END: T7
}

// Main time integration.
void
simulate(void)
{
    i64 max_iteration      = sim_params.max_iteration;
    i64 snapshot_frequency = sim_params.snapshot_frequency;

    for(i64 iteration = 0; iteration <= max_iteration; iteration++) {
        if((iteration % snapshot_frequency) == 0) {
            domain_save(iteration / snapshot_frequency);
        }

        border_exchange();
        if(mpi_ctx.on_boundary) {
            if(iteration % 100 == 0) {
                IsaLogDebug("Rank %ld performing boundary condition", mpi_ctx.rank);
            }
            boundary_condition();
        }
        time_step();
        move_buffer_window();
    }
}

void
mpi_types_free(void)
{
    MPI_Type_free(&mpi_ctx.MpiCol);
    MPI_Type_free(&mpi_ctx.MpiRow);
    MPI_Type_free(&mpi_ctx.MpiGrid);
}

bool
on_boundary(void)
{
    bool on_boundary
        = ((mpi_ctx.y == 0) || (mpi_ctx.x == 0) || (mpi_ctx.y == (mpi_ctx.cart_rows - 1))
           || (mpi_ctx.x == (mpi_ctx.cart_cols - 1)));

    return on_boundary;
}

void
mpi_ctx_initialize(int argc, char **argv)
{
    size_t param_send_buf_size = 4 * sizeof(i64);
    void  *param_send_buffer   = malloc(param_send_buf_size);
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
        MPI_Pack(&sim_params.M, 1, MPI_INT64_T, param_send_buffer, param_send_buf_size, &buffer_pos,
                 MPI_COMM_WORLD);
        MPI_Pack(&sim_params.N, 1, MPI_INT64_T, param_send_buffer, param_send_buf_size, &buffer_pos,
                 MPI_COMM_WORLD);
        MPI_Pack(&sim_params.max_iteration, 1, MPI_INT64_T, param_send_buffer, param_send_buf_size,
                 &buffer_pos, MPI_COMM_WORLD);
        MPI_Pack(&sim_params.snapshot_frequency, 1, MPI_INT64_T, param_send_buffer,
                 param_send_buf_size, &buffer_pos, MPI_COMM_WORLD);
    }

    MPI_Bcast(param_send_buffer, param_send_buf_size, MPI_PACKED, 0, MPI_COMM_WORLD);

    if(!mpi_ctx.i_am_root_rank) {
        int buffer_pos = 0;
        MPI_Unpack(param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.M, 1,
                   MPI_INT64_T, MPI_COMM_WORLD);
        MPI_Unpack(param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.N, 1,
                   MPI_INT64_T, MPI_COMM_WORLD);
        MPI_Unpack(param_send_buffer, param_send_buf_size, &buffer_pos, &sim_params.max_iteration,
                   1, MPI_INT64_T, MPI_COMM_WORLD);
        MPI_Unpack(param_send_buffer, param_send_buf_size, &buffer_pos,
                   &sim_params.snapshot_frequency, 1, MPI_INT64_T, MPI_COMM_WORLD);
    }
    free(param_send_buffer);

    int      n_cart_dims    = 2;
    int      cart_dims[2]   = { 0 };
    int      periodicity[2] = { 0 };
    int      reorder        = 0;
    MPI_Comm cart_comm;
    MPI_Dims_create(mpi_ctx.commsize, n_cart_dims, cart_dims);
    MPI_Cart_create(MPI_COMM_WORLD, n_cart_dims, cart_dims, periodicity, reorder, &cart_comm);

    int cart_rank;
    int coords[2];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, n_cart_dims, coords);

    mpi_ctx.rank        = cart_rank;
    mpi_ctx.cart_rows   = cart_dims[0];
    mpi_ctx.cart_cols   = cart_dims[1];
    mpi_ctx.y           = coords[0];
    mpi_ctx.x           = coords[1];
    mpi_ctx.cart_comm   = cart_comm;
    mpi_ctx.on_boundary = on_boundary();
    mpi_ctx.M           = sim_params.M / mpi_ctx.cart_rows;
    mpi_ctx.N           = sim_params.N / mpi_ctx.cart_cols;

    IsaLogDebug("Rank %ld has sim_params:\n M=%ld\n N=%ld\n max_iteration=%ld\n "
                "snapshot_frequency=%ld\n",
                mpi_ctx.rank, mpi_ctx.M, mpi_ctx.N, sim_params.max_iteration,
                sim_params.snapshot_frequency);

    MPI_Datatype MpiCol;
    MPI_Type_vector(mpi_ctx.M, 1, mpi_ctx.N + 2, MPI_DOUBLE, &MpiCol);
    MPI_Type_commit(&MpiCol);
    mpi_ctx.MpiCol = MpiCol;

    MPI_Datatype MpiRow;
    MPI_Type_contiguous(mpi_ctx.N, MPI_DOUBLE, &MpiRow);
    MPI_Type_commit(&MpiRow);
    mpi_ctx.MpiRow = MpiRow;

    MPI_Datatype MpiGrid;
    MPI_Type_vector(mpi_ctx.M, mpi_ctx.N, mpi_ctx.N + 2, MPI_DOUBLE, &MpiGrid);
    MPI_Type_commit(&MpiGrid);
    mpi_ctx.MpiGrid = MpiGrid;

    IsaLogDebug("Process %d in MPI_COMM_WORLD is now process %d in cart_comm with coordinates "
                "(%d, %d) is %son the boundary",
                mpi_ctx.rank, cart_rank, mpi_ctx.y, mpi_ctx.x, mpi_ctx.on_boundary ? "" : "not ");
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
    mpi_ctx.rank           = my_rank;
    mpi_ctx.commsize       = commsize;
    mpi_ctx.i_am_root_rank = (mpi_ctx.rank == 0);
    mpi_ctx_initialize(argc, argv);

    // END: T3

    // Set up the initial state of the domain
    domain_initialize();

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
    simulate();
    // MPI_Barrier(MPI_COMM_WORLD);

    if(mpi_ctx.i_am_root_rank) {
        time_end = MPI_Wtime();
        printf("Simulation time: %f\n", time_end - time_start);
    }

    // END: T2

    // Clean up and shut down
    domain_finalize();

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Comm_free(&mpi_ctx.cart_comm);
    mpi_types_free();
    MPI_Finalize();

    // END: T1d

    exit(EXIT_SUCCESS);
}
