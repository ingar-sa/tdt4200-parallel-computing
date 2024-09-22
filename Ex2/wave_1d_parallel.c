#define _XOPEN_SOURCE 600

#ifndef SDB_LOG_LEVEL
#define SDB_LOG_LEVEL 0
#endif
#include "Sdb.h"

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a

SDB_LOG_REGISTER(Mpi1dWaveEquation);

// Option to change numerical precision.
// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
typedef struct
{
    i64 MyRank;
    i64 CommSize;
    i64 NChildren;

    bool IAmRootRank;
    bool IAmFirstRank;
    bool IAmLastRank;
    bool IAmOnlyChild;
    bool ThereIsOneChild;

    i64 CellsPerRank;
    i64 RemainingCells;
    i64 NMyCells;

    int *RecvCounts;
    int *Displacements;
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

void
PrintMpiContext(const mpi_ctx *Context)
{
    printf("MyRank: %ld\n", Context->MyRank);
    printf("CommSize: %ld\n", Context->CommSize);
    printf("NChildren: %ld\n", Context->NChildren);
    printf("IAmRootRank: %s\n", Context->IAmRootRank ? "true" : "false");
    printf("IAmFirstRank: %s\n", Context->IAmFirstRank ? "true" : "false");
    printf("IAmLastRank: %s\n", Context->IAmLastRank ? "true" : "false");
    printf("IAmOnlyChild: %s\n", Context->IAmOnlyChild ? "true" : "false");
    printf("ThereIsOneChild: %s\n", Context->ThereIsOneChild ? "true" : "false");
    printf("CellsPerRank: %ld\n", Context->CellsPerRank);
    printf("RemainingCells: %ld\n", Context->RemainingCells);
    printf("NMyCells: %ld\n", Context->NMyCells);
    printf("RecvCounts: %p\n", (void *)Context->RecvCounts);
    printf("Displacements: %p\n", (void *)Context->Displacements);
    printf("\n");
}

void
PrintAllMpiContexts(void)
{
    MPI_Datatype Mpi_mpi_ctx;
    int          NStructMembers = 13; // - RecvCounts and Displacements
    int          MemberBlocks[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    MPI_Aint     MemberDisplacements[]
        = { offsetof(mpi_ctx, MyRank),       offsetof(mpi_ctx, CommSize),
            offsetof(mpi_ctx, NChildren),    offsetof(mpi_ctx, IAmRootRank),
            offsetof(mpi_ctx, IAmFirstRank), offsetof(mpi_ctx, IAmLastRank),
            offsetof(mpi_ctx, IAmOnlyChild), offsetof(mpi_ctx, ThereIsOneChild),
            offsetof(mpi_ctx, CellsPerRank), offsetof(mpi_ctx, RemainingCells),
            offsetof(mpi_ctx, NMyCells),     offsetof(mpi_ctx, RecvCounts),
            offsetof(mpi_ctx, Displacements) };

    MPI_Datatype MemberTypes[]
        = { MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_C_BOOL,  MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL,
            MPI_C_BOOL,  MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_AINT,   MPI_AINT };

    MPI_Type_create_struct(NStructMembers, MemberBlocks, MemberDisplacements, MemberTypes,
                           &Mpi_mpi_ctx);
    MPI_Type_commit(&Mpi_mpi_ctx);

    if(MpiCtx.IAmRootRank) {
        PrintMpiContext(&MpiCtx);

        mpi_ctx Context = { 0 };
        for(int i = 1; i < MpiCtx.CommSize; ++i) {
            MPI_Recv(&Context, 1, Mpi_mpi_ctx, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            PrintMpiContext(&Context);
        }
    } else {
        MPI_Send(&MpiCtx, 1, Mpi_mpi_ctx, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&Mpi_mpi_ctx);
}

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
    SdbLogDebug("Rank %d allocating memory for %ld cells", MpiCtx.MyRank, MpiCtx.NMyCells + 2);
    TimeSteps.PrevStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    TimeSteps.CurrStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    TimeSteps.NextStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));

    i64 StartingCell = 0;
    i64 EndingCell   = 0;

    if(MpiCtx.IAmRootRank) {
        EndingCell = MpiCtx.NMyCells;
    } else {
        StartingCell = (MpiCtx.MyRank - 1) * MpiCtx.CellsPerRank;
        EndingCell   = StartingCell + MpiCtx.NMyCells;
    }
    SdbLogDebug("Rank %d has starting cell %ld and ending cell %ld", MpiCtx.MyRank, StartingCell,
                EndingCell);

    // NOTE(ingar): We must offset i to the correct part of the entire domain for the initialization
    // values to be correct
    for(i64 i = StartingCell; i < EndingCell; ++i) {
        f64 Point = cos(M_PI * i / (f64)SimParams.NCells);

        UPrev(i - StartingCell) = Point;
        UCurr(i - StartingCell) = Point;
    }
    // END: T3

    // Set the time step for 1D case.
    WaveEquationParams.dt = WaveEquationParams.dx / WaveEquationParams.c;
}

// Return the memory to the OS.
void
FinalizeDomain(void)
{
    free(TimeSteps.PrevStep);
    free(TimeSteps.CurrStep);
    free(TimeSteps.NextStep);
}

// Rotate the time step buffers.
static void
RotateBuffers(void)
{
    if(MpiCtx.IAmRootRank && MpiCtx.ThereIsOneChild) {
        return;
    }

    f64 *PrevStep      = TimeSteps.PrevStep;
    TimeSteps.PrevStep = TimeSteps.CurrStep;
    TimeSteps.CurrStep = TimeSteps.NextStep;
    TimeSteps.NextStep = PrevStep;
}

// TASK: T4
// Derive step t+1 from steps t and t-1.
static void
PerformTimeStep(void)
{
    // TODO(ingar): Moving this out to simulate would be ideal, since then the root will not perform
    // the function call at all

    f64 dt = WaveEquationParams.dt;
    f64 c  = WaveEquationParams.c;
    f64 dx = WaveEquationParams.dx;

    SdbLogDebug("Hello from rank %ld in PerformTimeStep!\n", MpiCtx.MyRank);
    for(i64 i = 0; i < MpiCtx.NMyCells; ++i) {
        UNext(i) = -UPrev(i) + 2.0 * UCurr(i)
                 + (dt * dt * c * c) / (dx * dx) * (UCurr(i - 1) + UCurr(i + 1) - 2.0 * UCurr(i));
    }
}

// TASK: T6
// Neumann (reflective) boundary condition.
static void
PerformBoundaryCondition(void)
{
    // BEGIN: T6

    SdbLogDebug("Hello from rank %ld in PerformBoundaryCondition!\n", MpiCtx.MyRank);

    if(MpiCtx.IAmRootRank || MpiCtx.IAmOnlyChild) {
        SdbLogDebug("Rank %d performing boundary condition", MpiCtx.MyRank);
        // NOTE(ingar): Needed in case the program is run with only the root rank or one child rank.
        // If the program is run with more ranks it has no effect, since then the ghost values in
        // the root are not used for anything
        UCurr(-1)              = UCurr(1);
        UCurr(MpiCtx.NMyCells) = UCurr(MpiCtx.NMyCells - 2);
    } else if(MpiCtx.IAmFirstRank) {
        UCurr(-1) = UCurr(1);
    } else if(MpiCtx.IAmLastRank) {
        UCurr(MpiCtx.NMyCells) = UCurr(MpiCtx.NMyCells - 2);
    }

    // END: T6
}
// TASK: T5
// Communicate the border between processes.

static void
PerformBorderExchange(void)
{
    if(!MpiCtx.IAmFirstRank) {
        MPI_Sendrecv(&UCurr(0), 1, MPI_DOUBLE, MpiCtx.MyRank - 1, 0, &UCurr(-1), 1, MPI_DOUBLE,
                     MpiCtx.MyRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Right exchange
    if(!MpiCtx.IAmLastRank) {
        MPI_Sendrecv(&UCurr(MpiCtx.NMyCells - 1), 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0,
                     &UCurr(MpiCtx.NMyCells), 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
    }
}

// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void
SendDataToRoot(void)
{
    // BEGIN: T7

    SdbLogDebug("Hello from rank %ld in SendDataToRoot!\n", MpiCtx.MyRank);

    f64 *SendBuf = (MpiCtx.IAmRootRank) ? NULL : &UCurr(0);
    if(MpiCtx.IAmRootRank) {
        MPI_Gatherv(SendBuf, 0, MPI_DOUBLE, &UCurr(0), MpiCtx.RecvCounts, MpiCtx.Displacements,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(SendBuf, MpiCtx.NMyCells, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    SdbLogDebug("Goodbye from rank %ld in SendDataToRoot!\n", MpiCtx.MyRank);

    // END: T7
}

// Main time integration.
void
Simulate(void)
{
    SdbLogDebug("Hello from rank %ld in Simulate!\n", MpiCtx.MyRank);

    for(i64 Iteration = 0; Iteration <= SimParams.NTimeSteps; ++Iteration) {
        if(false && 0 == (Iteration % SimParams.SnapshotFrequency)) {
            SdbLogDebug("Iteration %ld\n", Iteration);
            if(MpiCtx.NChildren > 0) {
                SendDataToRoot();
            }
            if(MpiCtx.IAmRootRank) {
                SdbLogDebug("Rank %d saving domain", MpiCtx.MyRank);
                SaveDomain(Iteration / SimParams.SnapshotFrequency);
            }
        }

        // Derive step t+1 from steps t and t-1.
        if(!(MpiCtx.IAmRootRank || MpiCtx.IAmOnlyChild)) {
            PerformBorderExchange();
        }

        if(!(MpiCtx.IAmRootRank && MpiCtx.ThereIsOneChild)) {
            PerformBoundaryCondition();
        }

        if(!(MpiCtx.IAmRootRank && MpiCtx.ThereIsOneChild)) {
            PerformTimeStep();
        }

        RotateBuffers();
    }
}

int
main(int ArgCount, char **ArgV)
{
    // TASK: T1c
    // Initialise MPI
    // BEGIN: T1c

    int CommSize, MyRank;
    MPI_Init(&ArgCount, &ArgV);
    MPI_Comm_size(MPI_COMM_WORLD, &CommSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    if(CommSize > SimParams.NCells) {
        SdbLogError("Cannot use more processes than simulation cells!\n");
        goto exit;
    }

    MpiCtx.MyRank   = MyRank;
    MpiCtx.CommSize = CommSize;

    // NOTE(ingar): There are two corner-cases we must handle separately: if the program is
    // run with only the root rank, or if the program is run with the root rank and one
    // child process. In the first case, the root rank itself must perform the calculations.
    // This requires some
    if(MpiCtx.CommSize > 1) {
        MpiCtx.NChildren       = CommSize - 1;
        MpiCtx.IAmRootRank     = (MyRank == 0);
        MpiCtx.IAmFirstRank    = (MyRank == 1);
        MpiCtx.IAmLastRank     = (MyRank == (CommSize - 1));
        MpiCtx.IAmOnlyChild    = false;
        MpiCtx.ThereIsOneChild = (MpiCtx.NChildren == 1);
        MpiCtx.CellsPerRank    = SimParams.NCells / MpiCtx.NChildren;
        MpiCtx.RemainingCells  = SimParams.NCells % MpiCtx.NChildren;

        if(MpiCtx.IAmFirstRank && MpiCtx.IAmLastRank) {
            MpiCtx.IAmOnlyChild = true;
        }

        if(MpiCtx.IAmRootRank) {
            MpiCtx.NMyCells = SimParams.NCells;
        } else if(MpiCtx.IAmLastRank) {
            MpiCtx.NMyCells = MpiCtx.CellsPerRank + MpiCtx.RemainingCells;
        } else {
            MpiCtx.NMyCells = MpiCtx.CellsPerRank;
        }

        if(MpiCtx.IAmRootRank) {
            MpiCtx.RecvCounts    = malloc((MpiCtx.CommSize) * sizeof(*MpiCtx.RecvCounts));
            MpiCtx.Displacements = malloc((MpiCtx.CommSize) * sizeof(*MpiCtx.Displacements));

            for(int i = 1; i < MpiCtx.CommSize; ++i) {
                MpiCtx.RecvCounts[i]    = MpiCtx.CellsPerRank;
                MpiCtx.Displacements[i] = (i - 1) * MpiCtx.CellsPerRank;
            }
            MpiCtx.RecvCounts[MpiCtx.CommSize - 1] += MpiCtx.RemainingCells;

            MpiCtx.RecvCounts[0]    = 0;
            MpiCtx.Displacements[0] = 0;
#if 0
            for(int i = 0; i < MpiCtx.CommSize; ++i) {
                SdbLogInfo("Rank %d i recv count (%d) displacement (%d)\n", i + 1,
                           MpiCtx.RecvCounts[i], MpiCtx.Displacements[i]);
            }
#endif
        }

        // PrintAllMpiContexts();
    } else {
        // NOTE(ingar): Program is run with only root rank, so we don't need to do all of
        // the above stuff
        MpiCtx.NChildren       = 0;
        MpiCtx.IAmRootRank     = true;
        MpiCtx.IAmFirstRank    = false;
        MpiCtx.IAmLastRank     = false;
        MpiCtx.IAmOnlyChild    = false;
        MpiCtx.ThereIsOneChild = false;
        MpiCtx.CellsPerRank    = SimParams.NCells;
        MpiCtx.RemainingCells  = 0;
        MpiCtx.NMyCells        = SimParams.NCells;
    }

    InitializeDomain();

    // END: T1c

    // TODO(ingar): Figure out why they included this instead of doubles, which is what is
    // used by MPI timing struct timeval TimeStart, TimeEnd;

    // TASK: T2
    // Time your code
    // BEGIN: T2

    // TODO(ingar): Only rank 0 should perform the timing, since we're interested in the
    // time the entire simulation takes, not how long each process uses for its slice of it.
    f64 TimeStart = 0.0;
    f64 TimeEnd   = 0.0;

    if(MpiCtx.IAmRootRank) {
        TimeStart = MPI_Wtime();
    }

    Simulate();

    MPI_Barrier(MPI_COMM_WORLD);
    if(MpiCtx.IAmRootRank) {
        TimeEnd = MPI_Wtime();
        SdbLogInfo("Simulation time: %f", TimeEnd - TimeStart);
    }

    // END: T2

exit:
    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    FinalizeDomain();
    MPI_Finalize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
