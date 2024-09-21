#include <stddef.h>
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
    i64 NChildren;

    bool IAmRootRank;
    bool IAmFirstRank;
    bool IAmLastRank;

    i64 CellsPerRank;
    i64 RemainingCells;

    i64 NMyCells;
    // TODO(ingar): These two might not be needed
    i64 StartingCell;
    i64 EndingCell;

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
    TimeSteps.CurrStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    if(MpiCtx.IAmRootRank) {
        return;
    }
    TimeSteps.PrevStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));
    TimeSteps.NextStep = malloc((MpiCtx.NMyCells + 2) * sizeof(f64));

    // TODO(ingar): This might only be usefule here
    // NOTE(ingar): We must offset i to the correct part of the entire domain for the initialization
    // values to be correct
    for(i64 i = MpiCtx.StartingCell; i < MpiCtx.EndingCell; ++i) {
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
    if(!MpiCtx.IAmRootRank) {
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

    // printf("Hello from rank %ld in PerformTimeStep!\n", MpiCtx.MyRank);
    for(i64 i = 0; i < MpiCtx.NMyCells; ++i) {
        UNext(i) = -UPrev(i) + 2.0 * UCurr(i)
                 + (dt * dt * c * c) / (dx * dx) * (UCurr(i - 1) + UCurr(i + 1) - 2.0 * UCurr(i));
    }
}

// TASK: T6
// Neumann (reflective) boundary condition.
static inline void
PerformBoundaryCondition(void)
{
    // printf("Hello from rank %ld in PerformBoundaryCondition!\n", MpiCtx.MyRank);
    //  BEGIN: T6
    if(MpiCtx.IAmFirstRank) {
        UCurr(-1) = UCurr(1);
    } else if(MpiCtx.IAmLastRank) {
        UCurr(MpiCtx.NMyCells) = UCurr(MpiCtx.NMyCells - 2);
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
    // printf("Hello from rank %ld in PerformBorderExchange!\n", MpiCtx.MyRank);
    // NOTE(ingar): The first rank has no neighbor to the left
    if(!MpiCtx.IAmFirstRank) {
        f64 *LeftBorderRecv = &UCurr(-1);
        MPI_Recv(LeftBorderRecv, 1, MPI_DOUBLE, MpiCtx.MyRank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        f64 *LeftBorderSend = &UCurr(0);
        MPI_Send(LeftBorderSend, 1, MPI_DOUBLE, MpiCtx.MyRank - 1, 0, MPI_COMM_WORLD);
    }

    // NOTE(ingar): The last rank has no neighbor to the right
    if(!MpiCtx.IAmLastRank) {
        f64 *RightBorderSend = &UCurr(MpiCtx.NMyCells - 1);
        MPI_Send(RightBorderSend, 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0, MPI_COMM_WORLD);

        f64 *RightBorderRecv = &UCurr(MpiCtx.NMyCells);
        MPI_Recv(RightBorderRecv, 1, MPI_DOUBLE, MpiCtx.MyRank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    // END: T5
}

// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void
SendDataToRoot(void)
{
    // BEGIN: T7
    // printf("Hello from rank %ld in SendDataToRoot!\n", MpiCtx.MyRank);
    f64 *SendBuf = (MpiCtx.IAmRootRank) ? MPI_IN_PLACE : &UCurr(0);
    MPI_Gatherv(SendBuf, MpiCtx.NMyCells, MPI_DOUBLE, &UCurr(0), MpiCtx.RecvCounts,
                MpiCtx.Displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("Goodbye from rank %ld in SendDataToRoot!\n", MpiCtx.MyRank);
    //  END: T7
}

// Main time integration.
void
Simulate(void)
{
    // printf("Hello from rank %ld in Simulate!\n", MpiCtx.MyRank);
    //  Go through each time step.
    for(i64 Iteration = 0; Iteration <= SimParams.NTimeSteps; ++Iteration) {
        if(0 == (Iteration % SimParams.SnapshotFrequency)) {
            SendDataToRoot();
            if(MpiCtx.IAmRootRank) {
                // printf("Iteration %ld\n", Iteration);
                SaveDomain(Iteration / SimParams.SnapshotFrequency);
            }
        }

        // Derive step t+1 from steps t and t-1.
        if(!MpiCtx.IAmRootRank) {
            PerformBorderExchange();
            PerformBoundaryCondition();
            PerformTimeStep();
            RotateBuffers();
        }
    }
}

void
PrintMpiContext(const mpi_ctx *Context)
{
    printf("MyRank: %ld\n", Context->MyRank);
    printf("CommSize: %ld\n", Context->CommSize);
    printf("NChildren: %ld\n", Context->NChildren);
    printf("IAmRootRank: %s\n", Context->IAmRootRank ? "true" : "false");
    printf("IAmFirstRank: %s\n", Context->IAmFirstRank ? "true" : "false");
    printf("IAmLastRank: %s\n", Context->IAmLastRank ? "true" : "false");
    printf("CellsPerRank: %ld\n", Context->CellsPerRank);
    printf("RemainingCells: %ld\n", Context->RemainingCells);
    printf("NMyCells: %ld\n", Context->NMyCells);
    printf("StartingCell: %ld\n", Context->StartingCell);
    printf("EndingCell: %ld\n", Context->EndingCell);
    printf("RecvCounts: %p\n", (void *)Context->RecvCounts);
    printf("Displacements: %p\n", (void *)Context->Displacements);
    printf("\n");
}

int MPI_Type_create_structasdf(int count, int array_of_blocklengths[],
                               const MPI_Aint     array_of_displacements[],
                               const MPI_Datatype array_of_types[], MPI_Datatype *newtype);

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
            offsetof(mpi_ctx, CellsPerRank), offsetof(mpi_ctx, RemainingCells),
            offsetof(mpi_ctx, NMyCells),     offsetof(mpi_ctx, StartingCell),
            offsetof(mpi_ctx, EndingCell),   offsetof(mpi_ctx, RecvCounts),
            offsetof(mpi_ctx, Displacements) };

    MPI_Datatype MemberTypes[]
        = { MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_C_BOOL,  MPI_C_BOOL, MPI_C_BOOL, MPI_INT64_T,
            MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_AINT,   MPI_AINT };

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
        printf("Cannot use more processes than simulation cells!\n");
        goto exit;
    }

    MpiCtx.MyRank         = MyRank;
    MpiCtx.CommSize       = CommSize;
    MpiCtx.NChildren      = CommSize - 1;
    MpiCtx.IAmRootRank    = (MyRank == 0);
    MpiCtx.IAmFirstRank   = (MyRank == 1);
    MpiCtx.IAmLastRank    = (MyRank == (CommSize - 1));
    MpiCtx.CellsPerRank   = SimParams.NCells / MpiCtx.NChildren;
    MpiCtx.RemainingCells = SimParams.NCells % MpiCtx.NChildren;

    if(MpiCtx.IAmRootRank) {
        MpiCtx.NMyCells = SimParams.NCells;
    } else if(MpiCtx.IAmLastRank) {
        MpiCtx.NMyCells = MpiCtx.CellsPerRank + MpiCtx.RemainingCells;
    } else {
        MpiCtx.NMyCells = MpiCtx.CellsPerRank;
    }

    MpiCtx.StartingCell = (MpiCtx.MyRank - 1) * MpiCtx.CellsPerRank;
    MpiCtx.EndingCell   = MpiCtx.StartingCell + MpiCtx.NMyCells - 1;

    if(MpiCtx.IAmRootRank) {
        MpiCtx.RecvCounts    = malloc((MpiCtx.NChildren) * sizeof(*MpiCtx.RecvCounts));
        MpiCtx.Displacements = malloc((MpiCtx.NChildren) * sizeof(*MpiCtx.Displacements));

        for(int i = 0; i < MpiCtx.NChildren; ++i) {
            MpiCtx.RecvCounts[i] = MpiCtx.CellsPerRank;
        }

        MpiCtx.RecvCounts[MpiCtx.NChildren - 1] += MpiCtx.RemainingCells;

        MpiCtx.Displacements[0] = 0;
        for(int i = 1; i < MpiCtx.NChildren; ++i) {
            MpiCtx.Displacements[i] = MpiCtx.Displacements[i - 1] + MpiCtx.RecvCounts[i - 1];
        }

        for(int i = 0; i < MpiCtx.NChildren; ++i) {
            printf("Rank %d i recv count (%d) displacement (%d)\n", i, MpiCtx.RecvCounts[i],
                   MpiCtx.Displacements[i]);
        }
    }

    PrintAllMpiContexts();

    // END: T1c

    // TODO(ingar): Figure out why they included this instead of doubles, which is what is used
    // by MPI timing struct timeval TimeStart, TimeEnd;
    f64 TimeStart = 0.0;
    f64 TimeEnd   = 0.0;

    InitializeDomain();

    // TASK: T2
    // Time your code
    // BEGIN: T2

    // TODO(ingar): Only rank 0 should perform the timing, since we're interested in the time
    // the entire simulation takes, not how long each process uses for its slice of it. I don't
    // think we need to synchronize the ranks by using barriers in this case, but I'm unsure
    if(MpiCtx.IAmRootRank) {
        TimeStart = MPI_Wtime();
    }
    Simulate();

    if(MpiCtx.IAmRootRank) {
        TimeEnd = MPI_Wtime();
    }
    // END: T2

    // domain_finalize();

exit:
    // TASK: T1d
    // Finalise MPI
    // BEGIN: T1d
    MPI_Finalize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
