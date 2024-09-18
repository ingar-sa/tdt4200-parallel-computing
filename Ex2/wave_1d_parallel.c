#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
typedef struct
{
    i64 MyRank;
    i64 CommSize;

    bool IAmRootRank;
    bool IAmFirstRank;
    bool IAmLastRank;

    i64 CellsPerRank;
    i64 RemainingCells;

    i64 NMyCells;
    // TODO(ingar): These two might not be needed
    i64 StartingCell;
    i64 EndingCell;
} mpi_ctx;

static mpi_ctx MpiCtx = {};
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
static void
SaveDomain(i64 Step)
{
    // BEGIN: T8
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", Step);
    FILE *out = fopen(filename, "wb");
    fwrite(&UCurr(0), sizeof(f64), SimParams.NCells, out);
    fclose(out);
    // END: T8
}

// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
static void
InitializeDomain(void)
{
    // BEGIN: T3
    // TODO(ingar): Verify that giving the remainder of the cells to the final rank works/is correct
    MpiCtx.CellsPerRank   = SimParams.NCells / (MpiCtx.CommSize - 1);
    MpiCtx.RemainingCells = SimParams.NCells % (MpiCtx.CommSize - 1);
    if(MpiCtx.IAmRootRank)
    {
        MpiCtx.NMyCells = SimParams.NCells;
        // TODO(ingar): Verify that this works / move root's buffer for itself?
        TimeSteps.CurrStep = malloc((SimParams.NCells + 2) * sizeof(f64));
        return;
    }
    else if(MpiCtx.IAmLastRank)
    {
        MpiCtx.NMyCells = MpiCtx.CellsPerRank + MpiCtx.RemainingCells;
    }
    else
    {
        MpiCtx.NMyCells = MpiCtx.CellsPerRank;
    }

    TimeSteps.PrevStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    TimeSteps.CurrStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    TimeSteps.NextStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));

    // TODO(ingar): This might only be usefule here
    MpiCtx.StartingCell = (MpiCtx.MyRank - 1) * MpiCtx.CellsPerRank;
    MpiCtx.EndingCell   = MpiCtx.StartingCell + MpiCtx.NMyCells;

    // NOTE(ingar): We must offset i to the correct part of the entire domain for the initialization
    // values to be correct
    for(i64 i = MpiCtx.StartingCell; i < MpiCtx.EndingCell; ++i)
    {
        f64 Point = cos(M_PI * i / (f64)SimParams.NCells);

        UPrev(i - MpiCtx.StartingCell) = Point;
        UCurr(i - MpiCtx.StartingCell) = Point;
    }
    // END: T3

    // Set the time step for 1D case.
    WaveEquationParams.dt = WaveEquationParams.dx / WaveEquationParams.c;
}

// Return the memory to the OS.
void
domain_finalize(void)
{
    // TODO(ingar): Might not be unnecessary since every process allocates their own memory
    // Unnecessary. The memory is live for the entire program and will be reclaimed on process exit
    // free(buffers[0]);
    // free(buffers[1]);
    // free(buffers[2]);
}

// Rotate the time step buffers.
static inline void
RotateBuffers(void)
{
    if(!MpiCtx.IAmRootRank)
    {
        f64 *PrevStep      = TimeSteps.PrevStep;
        TimeSteps.PrevStep = TimeSteps.CurrStep;
        TimeSteps.CurrStep = TimeSteps.NextStep;
        TimeSteps.NextStep = PrevStep; // Existing values will be overwritten the next timestep
    }
}

// TASK: T4
// Derive step t+1 from steps t and t-1.
static inline void
PerformTimeStep(void)
{
    f64 dt = WaveEquationParams.dt;
    f64 c  = WaveEquationParams.c;
    f64 dx = WaveEquationParams.dx;

    for(i64 i = 0; i < MpiCtx.NMyCells; ++i)
    {
        UNext(i) = -UPrev(i) + 2.0 * UCurr(i)
                 + (dt * dt * c * c) / (dx * dx) * (UCurr(i - 1) + UCurr(i + 1) - 2.0 * UCurr(i));
    }
}

// TASK: T6
// Neumann (reflective) boundary condition.
static inline void
PerformBoundaryCondition(void)
{
    // BEGIN: T6
    if(MpiCtx.IAmFirstRank)
    {
        UCurr(-1) = UCurr(1);
    }
    else if(MpiCtx.IAmLastRank)
    {
        UCurr(SimParams.NCells) = UCurr(SimParams.NCells - 2);
    }
    // END: T6
}

// TASK: T5
// Communicate the border between processes.
static inline void
PerformBorderExchange(void)
{
    // BEGIN: T5
    // NOTE(ingar): To ensure that the program doesn't hang, the border exchange goes from left to
    // right. The first rank starts by sending, all other ranks start with receiving. Once a rank
    // has received the value from its left neighbor, it sends its value to it. It then sends its
    // value to its right neighbor, and, finally, receives from its right neighbor. The exceptions
    // are the first and last rank, which only exchange to the right and to the left, respectively.
    if(!MpiCtx.IAmRootRank)
    {
        // NOTE(ingar): The first rank has no neighbor to the left
        if(!MpiCtx.IAmFirstRank)
        {
            f64 *LeftBorderRecv = &UCurr(-1);
            MPI_Recv(LeftBorderRecv, 1, MPI_DOUBLE, MpiCtx.MyRank - 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            f64 *LeftBorderSend = &UCurr(0);
            MPI_Send(LeftBorderSend, 1, MPI_DOUBLE, MpiCtx.MyRank - 1, 0, MPI_COMM_WORLD);
        }

        // NOTE(ingar): The last rank has no neighbor to the right
        if(!MpiCtx.IAmLastRank)
        {
            f64 *RightBorderSend = &UCurr(MpiCtx.NMyCells - 1);
            MPI_Send(RightBorderSend, 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0, MPI_COMM_WORLD);

            f64 *RightBorderRecv = &UCurr(MpiCtx.NMyCells);
            MPI_Recv(RightBorderRecv, 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    // END: T5
}

// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void
SendDataToRoot()
{
    // BEGIN: T7

    // END: T7
}

// Main time integration.
void
Simulate(void)
{
    // Go through each time step.
    for(i64 Iteration = 0; Iteration <= SimParams.NTimeSteps; ++Iteration)
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

    if(CommSize > SimParams.NCells)
    {
        printf("Cannot use more processes than simulation cells!\n");
        exit(EXIT_FAILURE);
    }

    // TODO(ingar): Verify that this is correct
    MpiCtx.MyRank       = MyRank;
    MpiCtx.CommSize     = CommSize;
    MpiCtx.IAmRootRank  = MyRank == 0;
    MpiCtx.IAmFirstRank = MyRank == 1;
    MpiCtx.IAmLastRank  = MyRank == (CommSize - 1);

    // END: T1c

    // TODO(ingar): Figure out why they included this instead of doubles, which is what is used by
    // MPI timing
    // struct timeval TimeStart, TimeEnd;
    f64 TimeStart = 0.0;
    f64 TimeEnd   = 0.0;

    InitializeDomain();

    // TASK: T2
    // Time your code
    // BEGIN: T2

    // TODO(ingar): Only rank 0 should perform the timing, since we're interested in the time the
    // entire simulation takes, not how long each process uses for its slice of it. I don't think we
    // need to synchronize the ranks by using barriers in this case, but I'm unsure
    if(MpiCtx.IAmRootRank)
    {
        TimeStart = MPI_Wtime();
    }
    else
    {
        // TODO(ingar): Does the root rank need to do the simulation? I don't think so, which means
        // all of the logic for handling the root in the procedures used in simulate is unnecessary
        Simulate();
    }

    if(MpiCtx.IAmRootRank)
    {
        TimeEnd = MPI_Wtime();
    }
    // END: T2

    // domain_finalize();

    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Finalize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
