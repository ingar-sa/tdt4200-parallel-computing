#define _XOPEN_SOURCE 600
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// TASK: T1a
// Include the pthreads library
// BEGIN: T1a
#include <pthread.h>
// END: T1a

// Option to change numerical precision
typedef int64_t int_t;
typedef double  real_t;

// TASK: T1b
// Pthread management
// BEGIN: T1b
typedef struct
{
    int_t t_id;
    int_t row_start, row_end;
} PthreadSimContext;

typedef struct
{
    int_t              n_threads;
    pthread_barrier_t  barrier;
    pthread_t         *pthreads;
    PthreadSimContext *sim_contexts;
} PthreadContext;
static PthreadContext pt_ctx = {};

// END: T1b

// Performance measurement
struct timeval t_start, t_end;
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

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

// TASK: T3
// Integration formula
void
time_step(int_t row_start, int_t row_end)
{
    int_t  N  = sim_params.N;
    real_t dt = weq_params.dt;
    real_t h  = weq_params.h;
    real_t c  = weq_params.c;

    // BEGIN: T3
    for(int_t i = row_start; i < row_end; i += 1) {
        for(int_t j = 0; j < N; j++) {
            real_t Next
                = -U_prv(i, j) + 2.0 * U(i, j)
                + (dt * dt * c * c) / (h * h)
                      * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) - 4.0 * U(i, j));
            U_nxt(i, j) = Next;
        }
    }
    // END: T3
}

// TASK: T4
// Neumann (reflective) boundary condition
void
boundary_condition(PthreadSimContext sim_context)
{
    int_t row_start = sim_context.row_start;
    int_t row_end   = sim_context.row_end;
    int_t N         = sim_params.N;

    // BEGIN: T4
    // Left/right
    for(int_t i = row_start; i < row_end; i += 1) {
        U(i, -1) = U(i, 1);
        U(i, N)  = U(i, N - 2);
    }

    // Top/bottom
    if(sim_context.t_id == 1) {
        for(int_t j = 0; j < N; j += 1) {
            U(-1, j) = U(1, j);
        }
    }
    if(sim_context.t_id == pt_ctx.n_threads) {
        for(int_t j = 0; j < N; j += 1) {
            U(N, j) = U(N - 2, j);
        }
    }
    // END: T4
}
// Save the present time step in a numbered file under 'data/'
void
domain_save(int_t step)
{
    int_t N = sim_params.N;
    char  filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    for(int_t i = 0; i < N; i++) {
        fwrite(&U(i, 0), sizeof(real_t), N, out);
    }
    fclose(out);
}

// TASK: T5
// Main loop
void *
simulate(void *arg)
{
    PthreadSimContext sim_ctx;
    memcpy(&sim_ctx, arg, sizeof(PthreadSimContext)); // Move sim context onto the stack

    // BEGIN: T5
    // Go through each time step
    for(int_t iteration = 0; iteration <= sim_params.max_iteration; iteration++) {
        pthread_barrier_wait(&pt_ctx.barrier);
        if(sim_ctx.t_id == 1) {
            if((iteration % sim_params.snapshot_freq) == 0) {
                domain_save(iteration / sim_params.snapshot_freq);
            }
        }

        // Derive step t+1 from steps t and t-1
        pthread_barrier_wait(&pt_ctx.barrier);
        boundary_condition(sim_ctx);

        pthread_barrier_wait(&pt_ctx.barrier);
        time_step(sim_ctx.row_start, sim_ctx.row_end);

        // Rotate the time step buffers
        pthread_barrier_wait(&pt_ctx.barrier);
        if(sim_ctx.t_id == 1) {
            move_buffer_window();
        }
    }
    // END: T5

    return NULL;
}

static void
pt_ctx_initialize(void)
{
    pthread_barrier_init(&pt_ctx.barrier, NULL, pt_ctx.n_threads);
    pt_ctx.pthreads     = malloc(pt_ctx.n_threads * sizeof(pthread_t));
    pt_ctx.sim_contexts = malloc(pt_ctx.n_threads * sizeof(PthreadSimContext));

    int_t rows_per_thread = sim_params.N / pt_ctx.n_threads;
    int_t remaining_rows  = sim_params.N % pt_ctx.n_threads;

    for(int_t i = 0; i < pt_ctx.n_threads; ++i) {
        pt_ctx.sim_contexts[i].t_id      = i + 1;
        pt_ctx.sim_contexts[i].row_start = i * rows_per_thread;
        pt_ctx.sim_contexts[i].row_end   = i * rows_per_thread + rows_per_thread;
    }
    pt_ctx.sim_contexts[pt_ctx.n_threads - 1].row_end += remaining_rows;
}

static void
pt_ctx_deinitialize(void)
{
    pthread_barrier_destroy(&pt_ctx.barrier);
}

static void
run_simulation(void)
{
    for(int_t i = 0; i < pt_ctx.n_threads; ++i) {
        pthread_create(&pt_ctx.pthreads[i], NULL, simulate, &pt_ctx.sim_contexts[i]);
    }
    for(int_t i = 0; i < pt_ctx.n_threads; ++i) {
        pthread_join(pt_ctx.pthreads[i], NULL);
    }
}

int
main(int argc, char **argv)
{
    // Number of threads is an optional argument, sanity check its value
    if(argc > 1) {
        pt_ctx.n_threads = strtol(argv[1], NULL, 10);
        if(errno == EINVAL) {
            fprintf(stderr, "'%s' is not a valid thread count\n", argv[1]);
        }
        if(pt_ctx.n_threads < 1) {
            fprintf(stderr, "Number of threads must be >0\n");
            exit(EXIT_FAILURE);
        }
    }

    // TASK: T1c
    // Initialise pthreads
    // BEGIN: T1c
    pt_ctx_initialize();
    // END: T1b

    // Set up the initial state of the domain
    domain_initialize();

    // Time the execution
    gettimeofday(&t_start, NULL);

    // TASK: T2
    // Run the integration loop
    // BEGIN: T2
    run_simulation();
    // END: T2

    // Report how long we spent in the integration stage
    gettimeofday(&t_end, NULL);
    printf("%lf seconds elapsed with %ld threads\n", WALLTIME(t_end) - WALLTIME(t_start),
           pt_ctx.n_threads);

    // Clean up and shut down
    domain_finalize();
    // TASK: T1d
    // Finalise pthreads
    // BEGIN: T1d
    pt_ctx_deinitialize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
