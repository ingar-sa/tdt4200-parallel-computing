#define _XOPEN_SOURCE 600
#include "isa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ISA_LOG_REGISTER(wave_1d);

// Simulation parameters: size, step count, and how often to save the state.
static struct
{
    const i64 N;
    const i64 MaxIterations;
    const i64 SnapshotFrequency;
} SimParams = { .N = 1024, .MaxIterations = 4000, .SnapshotFrequency = 10 };

// Wave equation parameters, time step is derived from the space step.
static struct
{
    const f64 c;
    const f64 dx;
    f64       dt;
} WaveEquationParams = { .c = 1.0, .dx = 1.0 };

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
// Each buffer contains all of the values for the entire wave at that time step
static struct
{
    f64 *PrevStep;
    f64 *CurrStep;
    f64 *NextStep;
} TimeSteps;

#define UPrev(i) TimeSteps.PrevStep[(i) + 1]
#define UCurr(i) TimeSteps.CurrStep[(i) + 1]
#define UNext(i) TimeSteps.NextStep[(i) + 1]

// Save the present time step in a numbered file under 'data/'.
void
DomainSave(i64 Step)
{
    char Filename[256];
    sprintf(Filename, "data/%.5ld.dat", Step);
    FILE *Out = fopen(Filename, "wb");
    fwrite(&UCurr(0), sizeof(f64), SimParams.N, Out);
    fclose(Out);
}

// TASK: T1
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void
DomainInitialize(isa_arena *Arena)
{
    // +2 for the ghost points (???)
    TimeSteps.PrevStep = IsaPushArray(Arena, f64, SimParams.N + 2);
    TimeSteps.CurrStep = IsaPushArray(Arena, f64, SimParams.N + 2);
    TimeSteps.NextStep = IsaPushArray(Arena, f64, SimParams.N + 2);

    for(int i = 0; i < SimParams.N; ++i)
    {
        f64 Point = cos(M_PI * ((f64)i / (f64)SimParams.N));
        UPrev(i)  = Point;
        UCurr(i)  = Point;
        // UNext(i) = Point; ???
    }

    WaveEquationParams.dt = WaveEquationParams.c / WaveEquationParams.dx; //???
}

// TASK T2:
// Return the memory to the OS.
// BEGIN: T2
void
DomainFinalize(void)
{

    /*
     * Since I use an arena allocator, and its memory is
     * alive for the entire program, this step is unnecessary.
     * Explicitly freeing memory that will live for the entire
     * program is also redundant since it will be reclaimed by the
     * OS when the process exits regardless. If there are security
     * concerns, then explicityly clearing the memory before the
     * program is finished could be done, but the OS *should*
     * do this as well.
     */

} // END: T2

// TASK: T3
// Rotate the time step buffers.
// BEGIN: T3
void
PerformTimeStep(void)
{
    // Simply rotate the pointers. No need to copy data.
    f64 *PrevStep      = TimeSteps.PrevStep;
    TimeSteps.PrevStep = TimeSteps.CurrStep;
    TimeSteps.CurrStep = TimeSteps.NextStep;
    TimeSteps.NextStep = PrevStep;
}
// END: T3

// TASK: T4
// Derive step t+1 from steps t and t-1.
// BEGIN: T4
inline f64
Integrate(i64 i)
{
    // Formula U(t+1, i) = -U(t-1, i) + 2*U(t, i) + [((dt^2 * c^2)/(h^2=dx^2) * (U(t, i-1) + U(t, i+1) - 2*U(t, i))]

    f64 NewUNext;
    f64 UCurrPrevPoint, UCurrThisPoint, UCurrNextPoint;
    f64 UPrev;
    f64 dt = WaveEquationParams.dt, c = WaveEquationParams.c, dx = WaveEquationParams.dx;

    UCurrPrevPoint = UCurr(i - 1);
    UCurrThisPoint = UCurr(i);
    UCurrNextPoint = UCurr(i + 1);
    UPrev          = UPrev(i);

    NewUNext = -UPrev + 2 * UCurrThisPoint
             + (((dt * dt * c * c) / (dx * dx)) * (UCurrPrevPoint + UCurrNextPoint - 2 * UCurrThisPoint));

    return NewUNext;
}
// END: T4

// TASK: T5
// Neumann (reflective) boundary condition.
// BEGIN: T5
void
NeumannBoundary(void)
{
}
// END: T5

// TASK: T6
// Main time integration.
void
Simulate(void)
{
    // BEGIN: T6
    i64 Iteration = 0;
    DomainSave(Iteration / SimParams.SnapshotFrequency);

#if 0
    {
        printf("%f %s", NewUNext, (i % 10 == 0) ? "\n" : "");
    }
#endif

    // END: T6
}

int
main(void)
{
    isa_arena Arena;
    u64       ArenaSize = IsaKibiByte(64);
    u8       *Mem       = malloc(ArenaSize);
    if(NULL == Mem)
    {
        perror("Failed to allocate memory!");
        exit(EXIT_FAILURE);
    }
    IsaArenaInit(&Arena, Mem, ArenaSize);

    DomainInitialize(&Arena);
    Simulate();
    // DomainFinalize();

    exit(EXIT_SUCCESS);
}
