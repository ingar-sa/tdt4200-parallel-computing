#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <mpi.h>
#include <stdint.h>
#include <stdbool.h>

// Option to change numerical precision
typedef int64_t int_t;
typedef double  real_t;

// Context for each MPI process
typedef struct
{
    int_t rank;
    int_t commsize;
    bool  on_boundary;

    int_t M, N;
    int_t y, x;
    int_t cart_cols, cart_rows;

    MPI_Comm     cart_comm;
    MPI_Datatype MpiCol;
    MPI_Datatype MpiRow;
    MPI_Datatype MpiGrid;
} MpiCtx;

// NOTE: I use wrapper structs for the global state because I think it improves the readability of
// the code

//  Simulation parameters: size, step count, and how often to save the state.
typedef struct
{
    int_t M;
    int_t N;
    int_t max_iteration;
    int_t snapshot_frequency;
} SimParams;

// Wave equation parameters, time step is derived from the space step.
typedef struct
{
    const real_t c;
    const real_t dx;
    const real_t dy;
    real_t       dt;
} WaveEquationParams; // wave_equation_params;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
typedef struct
{
    real_t *prev_step;
    real_t *curr_step;
    real_t *next_step;
} TimeSteps;

#endif
