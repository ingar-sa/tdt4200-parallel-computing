#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <mpi.h>
#include "isa.h"

typedef struct
{
    i64 rank;
    i64 commsize;
    i64 cells_per_rank;

    bool i_am_root_rank;
    bool on_boundary;

    i64 M, N;
    int y, x;
    int cart_cols, cart_rows;

    MPI_Comm     cart_comm;
    MPI_Datatype MpiCol;
    MPI_Datatype MpiRow;
    MPI_Datatype MpiGrid;
} MpiCtx;

// Simulation parameters: size, step count, and how often to save the state.
typedef struct
{
    i64 M;
    i64 N;
    i64 max_iteration;
    i64 snapshot_frequency;
} SimParams;

// Wave equation parameters, time step is derived from the space step.

typedef struct
{
    const f64 c;
    const f64 dx;
    const f64 dy;
    f64       dt;
} WaveEquationParams; // wave_equation_params;

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
typedef struct
{
    f64 *prev_step;
    f64 *curr_step;
    f64 *next_step;
} TimeSteps;

#endif
