#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

// Option to change numerical precision.
typedef int64_t i64;
typedef double  f64;

// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
;
// END: T1b

// Simulation parameters: size, step count, and how often to save the state.
typedef struct
{
    const i64 NCells;
    const i64 NTimeSteps;
    const i64 SnapshotFrequency;
} sim_params;

static sim_params SimParams = { .NCells = 65536, .NTimeSteps = 100000, .SnapshotFrequency = 500 };

// Wave equation parameters, time step is derived from the space step.

typedef struct
{
    const f64 c;
    const f64 dx;
    f64       dt;
} wave_equation_params;

static wave_equation_params WaveEquationParams = { .c = 1.0, .dx = 1.0, .dt = 0.0 };

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
typedef struct
{
    f64 *PrevStep;
    f64 *CurrStep;
    f64 *NextStep;
} time_steps;

static time_steps TimeSteps = {};

#define UPrev(i) TimeSteps.PrevStep[(i) + 1]
#define UCurr(i) TimeSteps.CurrStep[(i) + 1]
#define UNext(i) TimeSteps.NextStep[(i) + 1]

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALL_TIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// TASK: T8
// Save the present time step in a numbered file under 'data/'.
void
SaveDomain(i64 step)
{
    // BEGIN: T8
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    fwrite(&UCurr(0), sizeof(f64), SimParams.NTimeSteps, out);
    fclose(out);
    // END: T8
}

// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void
InitializeDomain(int CommSize, int MyRank)
{
    // BEGIN: T3
    TimeSteps.PrevStep = malloc((SimParams.NCells + 2) * sizeof(f64));
    TimeSteps.CurrStep = malloc((SimParams.NCells + 2) * sizeof(f64));
    TimeSteps.NextStep = malloc((SimParams.NCells + 2) * sizeof(f64));

    for(i64 i = 0; i < SimParams.NCells; i++)
    {
        f64 Point = cos(M_PI * i / (f64)SimParams.NCells);
        UPrev(i)  = Point;
        UCurr(i)  = Point;
    }
    // END: T3

    // Set the time step for 1D case.
    WaveEquationParams.dt = WaveEquationParams.dx / WaveEquationParams.c;
}

// Return the memory to the OS.
void
domain_finalize(void)
{
    // Unnecessary. The memory is live for the entire program and will be reclaimed on process exit
    // free(buffers[0]);
    // free(buffers[1]);
    // free(buffers[2]);
}

// Rotate the time step buffers.
static inline void
RotateBuffers(void)
{
    f64 *PrevStep      = TimeSteps.PrevStep;
    TimeSteps.PrevStep = TimeSteps.CurrStep;
    TimeSteps.CurrStep = TimeSteps.NextStep;
    TimeSteps.NextStep = PrevStep; // Existing values will be overwritten the next timestep
}

// TASK: T4
// Derive step t+1 from steps t and t-1.
static inline void
PerformTimeStep(void)
{
    f64 dt = WaveEquationParams.dt, c = WaveEquationParams.c, dx = WaveEquationParams.dx;
    for(i64 i = 0; i < SimParams.NCells; ++i)
    {
        UNext(i) = -UPrev(i) + 2.0 * UCurr(i)
                 + (dt * dt * c * c) / (dx * dx) * (UCurr(i - 1) + UCurr(i + 1) - 2.0 * UCurr(i));
    }
}

// TASK: T6
// Neumann (reflective) boundary condition.
void
PerformBoundaryCondition(void)
{
    // BEGIN: T6
    UCurr(-1)               = UCurr(1);
    UCurr(SimParams.NCells) = UCurr(SimParams.NCells - 2);
    // END: T6
}

// TASK: T5
// Communicate the border between processes.
void
PerformBorderExchange(void)
{
    // BEGIN: T5
    ;
    // END: T5
}

// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void
SendDataToRoot()
{
    // BEGIN: T7
    ;
    // END: T7
}

// Main time integration.
void
Simulate(void)
{
    // Go through each time step.
    for(i64 Iteration = 0; Iteration <= SimParams.NTimeSteps; Iteration++)
    {
        if(0 == (Iteration % SimParams.SnapshotFrequency))
        {
            SendDataToRoot();
            SaveDomain(Iteration / SimParams.SnapshotFrequency);
        }

        // Derive step t+1 from steps t and t-1.
        PerformBorderExchange();
        PerformBoundaryCondition();
        PerformTimeStep();
        RotateBuffers();
    }
}

int
main(int ArgCount, char **ArgV)
{
    // TASK: T1c
    // Initialise MPI
    // BEGIN: T1c

    int CommSize;
    int MyRank;
    MPI_Init(&ArgCount, &ArgV);
    MPI_Comm_size(MPI_COMM_WORLD, &CommSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    // END: T1c

    // TODO(ingar): Figure out why they included this instead of doubles, which is what is used by
    // MPI timing
    // struct timeval TimeStart, TimeEnd;
    f64 TimeStart = 0.0;
    f64 TimeEnd   = 0.0;

    InitializeDomain(CommSize, MyRank);

    // TASK: T2
    // Time your code
    // BEGIN: T2

    // TODO(ingar): Only rank 0 should perform the timing, since we're interested in the time the
    // entire simulation takes, not how long each process uses for its slice of it. I don't think we
    // need to synchronize the ranks by using barriers in this case, but I'm unsure
    TimeStart = MPI_Wtime();

    Simulate();

    TimeEnd = MPI_Wtime();
    // END: T2

    // domain_finalize();

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Finalize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
