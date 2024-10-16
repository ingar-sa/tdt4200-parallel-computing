#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// TASK: T6
// Include the OpenMP library
// BEGIN: T6
#include <omp.h>
// END: T6

// Option to change numerical precision
typedef int64_t int_t;
typedef double  real_t;

typedef struct
{
    int_t N;
    int_t max_iteration;
    int_t snapshot_freq;
} SimParams;
static SimParams sim_params = { 1024, 4000, 20 };

// Wave equation parameters, time step is derived from the space step.
typedef struct
{
    const real_t c;
    const real_t h;
    real_t       dt;
} WaveEquationParams; // wave_equation_params;
static WaveEquationParams weq_params = { 1.0, 1.0 };

// Buffers for three time steps, indexed with 2 ghost points for the boundary
typedef struct
{
    real_t *prev_step;
    real_t *curr_step;
    real_t *next_step;
} TimeSteps;
static TimeSteps time_steps;

#define U_prv(i, j) time_steps.prev_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define U(i, j)     time_steps.curr_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define U_nxt(i, j) time_steps.next_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]

// Rotate the time step buffers.
static void
move_buffer_window(void)
{
    real_t *prev_step    = time_steps.prev_step;
    time_steps.prev_step = time_steps.curr_step;
    time_steps.curr_step = time_steps.next_step;
    time_steps.next_step = prev_step;
}

// Set up our three buffers, and fill two with an initial perturbation
void
domain_initialize(void)
{
    int_t  N = sim_params.N;
    real_t h = weq_params.h;
    real_t c = weq_params.c;

    size_t time_step_sz = (N + 2) * (N + 2) * sizeof(real_t);

    time_steps.prev_step = malloc(time_step_sz);
    time_steps.curr_step = malloc(time_step_sz);
    time_steps.next_step = malloc(time_step_sz);

    for(int_t i = 0; i < N; i++) {
        for(int_t j = 0; j < N; j++) {
            real_t delta
                = sqrt(((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) / (real_t)N);
            real_t val  = exp(-4.0 * delta * delta);
            U_prv(i, j) = U(i, j) = val;
        }
    }

    // Set the time step
    weq_params.dt = (h * h) / (4.0 * c * c);
}

// Get rid of all the memory allocations
static void
domain_finalize(void)
{
    free(time_steps.prev_step);
    free(time_steps.curr_step);
    free(time_steps.next_step);
}

// TASK: T7
// Integration formula
void
time_step(void)
{
    int_t  N  = sim_params.N;
    real_t dt = weq_params.dt;
    real_t h  = weq_params.h;
    real_t c  = weq_params.c;

    // BEGIN: T7
#pragma omp parallel for
    for(int_t i = 0; i < N; i++) {
        for(int_t j = 0; j < N; j++) {
            U_nxt(i, j)
                = -U_prv(i, j) + 2.0 * U(i, j)
                + (dt * dt * c * c) / (h * h)
                      * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) - 4.0 * U(i, j));
        }
    }
    // END: T7
}

// Neumann (reflective) boundary condition
void
boundary_condition(void)
{
    int_t N = sim_params.N;

    for(int_t i = 0; i < N; i++) {
        U(i, -1) = U(i, 1);
        U(i, N)  = U(i, N - 2);
    }
    for(int_t j = 0; j < N; j++) {
        U(-1, j) = U(1, j);
        U(N, j)  = U(N - 2, j);
    }
}

// Save the present time step in a numbered file under 'data/'
void
domain_save(int_t step)
{
    int_t N = sim_params.N;

    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    for(int_t i = 0; i < N; i++) {
        fwrite(&U(i, 0), sizeof(real_t), N, out);
    }
    fclose(out);
}

// Main time integration.
void
simulate(void)
{
    int_t max_iteration = sim_params.max_iteration;
    int_t snapshot_freq = sim_params.snapshot_freq;

    // Go through each time step
    for(int_t iteration = 0; iteration <= max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            domain_save(iteration / snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}

int
main(void)
{
    // Set up the initial state of the domain
    domain_initialize();

    double t_start, t_end;
    t_start = omp_get_wtime();
    // Go through each time step
    simulate();
    t_end = omp_get_wtime();
    printf("%lf seconds elapsed with %d threads\n", t_end - t_start, omp_get_max_threads());

    // Clean up and shut down
    domain_finalize();
    exit(EXIT_SUCCESS);
}
