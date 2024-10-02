#ifndef DATATYPES_H_
#define DATATYPES_H_

#include "isa.h"

typedef struct
{
    i64 my_rank;
    i64 commsize;
    i64 n_children;

    bool i_am_root_rank;
    bool i_am_first_rank;
    bool i_am_last_rank;
    bool i_am_only_child;
    bool there_is_one_child;

    i64 cells_per_rank;
    i64 remaining_cells;
    i64 n_my_cells;

    int *recv_counts;
    int *displacements;
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
