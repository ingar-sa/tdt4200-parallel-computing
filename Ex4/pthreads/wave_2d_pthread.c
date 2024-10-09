#define _XOPEN_SOURCE 600
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../isa.h"

ISA_LOG_REGISTER(Pthreads2DWeq);

// TASK: T1a
// Include the pthreads library
// BEGIN: T1a
#include <pthread.h>
// END: T1a

// Option to change numerical precision
// typedef int64_t int_t;
// typedef double  real_t;

// TASK: T1b
// Pthread management
// BEGIN: T1b
typedef struct
{
    i64 t_id;
    i64 row_start, row_end;
} PtSimContext;

typedef struct
{
    i64               n_threads;
    pthread_barrier_t barrier;
    pthread_t        *pthreads;
    PtSimContext     *sim_contexts;
} PtContext;
static PtContext pt_ctx = {};

// END: T1b

// Performance measurement
struct timeval t_start, t_end;
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef struct
{
    i64 N;
    i64 max_iteration;
    i64 snapshot_freq;
} SimParams;
static SimParams sim_params = { 1024, 4000, 20 };

// Wave equation parameters, time step is derived from the space step.
typedef struct
{
    const f64 c;
    const f64 h;
    f64       dt;
} WaveEquationParams; // wave_equation_params;
static WaveEquationParams weq_params = { 1.0, 1.0 };

// Buffers for three time steps, indexed with 2 ghost points for the boundary
typedef struct
{
    f64 *prev_step;
    f64 *curr_step;
    f64 *next_step;
} TimeSteps;
static TimeSteps time_steps;

#define U_prv(i, j) time_steps.prev_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define U(i, j)     time_steps.curr_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]
#define U_nxt(i, j) time_steps.next_step[((i) + 1) * (sim_params.N + 2) + (j) + 1]

// Rotate the time step buffers.

static void
move_buffer_window(void)
{
    f64 *prev_step       = time_steps.prev_step;
    time_steps.prev_step = time_steps.curr_step;
    time_steps.curr_step = time_steps.next_step;
    time_steps.next_step = prev_step;
}

// Set up our three buffers, and fill two with an initial perturbation
void
domain_initialize(void)
{
    i64 N  = sim_params.N;
    f64 dt = weq_params.dt;
    f64 h  = weq_params.h;
    f64 c  = weq_params.c;

    size_t time_step_sz = (N + 2) * (N + 2) * sizeof(f64);

    time_steps.prev_step = malloc(time_step_sz);
    time_steps.curr_step = malloc(time_step_sz);
    time_steps.next_step = malloc(time_step_sz);

    for(i64 i = 0; i < N; i++) {
        for(i64 j = 0; j < N; j++) {
            f64 delta   = sqrt(((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) / (f64)N);
            U_prv(i, j) = U(i, j) = exp(-4.0 * delta * delta);
        }
    }

    // Set the time step
    dt = (h * h) / (4.0 * c * c);
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
time_step(i64 thread_id)
{
    i64 N  = sim_params.N;
    f64 dt = weq_params.dt;
    f64 h  = weq_params.h;
    f64 c  = weq_params.c;

    // BEGIN: T3
    for(i64 i = 0; i < N; i += 1) {
        for(i64 j = 0; j < N; j++) {
            U_nxt(i, j)
                = -U_prv(i, j) + 2.0 * U(i, j)
                + (dt * dt * c * c) / (h * h)
                      * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) - 4.0 * U(i, j));
        }
    }
    // END: T3
}

// TASK: T4
// Neumann (reflective) boundary condition
void
boundary_condition(i64 thread_id)
{
    i64 N = sim_params.N;
    // TODO(ingar): Logic for only copying outer edges

    // BEGIN: T4
    for(i64 i = 0; i < N; i += 1) {
        U(i, -1) = U(i, 1);
        U(i, N)  = U(i, N - 2);
    }
    for(i64 j = 0; j < N; j += 1) {
        U(-1, j) = U(1, j);
        U(N, j)  = U(N - 2, j);
    }
    // END: T4
}

// Save the present time step in a numbered file under 'data/'
void
domain_save(i64 step)
{
    i64 N = sim_params.N;

    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    for(i64 i = 0; i < N; i++) {
        fwrite(&U(i, 0), sizeof(f64), N, out);
    }
    fclose(out);
}

// TASK: T5
// Main loop
void *
// simulate(void *id)
simulate(void *arg)
{
    PtSimContext *sim_ctx = (PtSimContext *)arg;
    IsaLogDebug("Hello from thread %ld!", sim_ctx->t_id);

    pthread_barrier_wait(&pt_ctx.barrier);
    // BEGIN: T5
    // Go through each time step
    for(i64 iteration = 0; iteration <= sim_params.max_iteration; iteration++) {
        if(sim_ctx->t_id == 1) {
            if((iteration % sim_params.snapshot_freq) == 0) {
                domain_save(iteration / sim_params.snapshot_freq);
            }
        }

        // Derive step t+1 from steps t and t-1

        pthread_barrier_wait(&pt_ctx.barrier);
        boundary_condition(0);

        pthread_barrier_wait(&pt_ctx.barrier);
        time_step(0);

        // Rotate the time step buffers
        pthread_barrier_wait(&pt_ctx.barrier);
        if(sim_ctx->t_id == 1) {
            move_buffer_window();
        }
        pthread_barrier_wait(&pt_ctx.barrier);
    }
    // END: T5

    IsaLogDebug("Goodbye from thread %ld", sim_ctx->t_id);
    return NULL;
}

long
get_cache_size(int level)
{
    char filename[256];
    char line[256];
    long cache_size = -1;

    snprintf(filename, sizeof(filename), "/sys/devices/system/cpu/cpu0/cache/index%d/size",
             level - 1);

    FILE *file = fopen(filename, "r");
    if(file == NULL) {
        return -1; // Cache level not found
    }

    if(fgets(line, sizeof(line), file) != NULL) {
        cache_size = strtol(line, NULL, 10);
        // Convert to bytes if necessary
        if(strstr(line, "K") != NULL) {
            cache_size *= 1024;
        } else if(strstr(line, "M") != NULL) {
            cache_size *= 1024 * 1024;
        }
    }

    fclose(file);
    return cache_size;
}

int
get_cache_line_size(int level)
{
    char filename[256];
    char line[256];
    int  line_size = -1;

    snprintf(filename, sizeof(filename),
             "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", level - 1);

    FILE *file = fopen(filename, "r");
    if(file == NULL) {
        return -1; // Cache level not found
    }

    if(fgets(line, sizeof(line), file) != NULL) {
        line_size = atoi(line);
    }

    fclose(file);
    return line_size;
}

#if 0

  g g g g g g g g g g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g 0 0 0 0 0 0 0 0 g
  g g g g g g g g g g

#endif

static void
pt_ctx_initialize(void)
{
    pthread_barrier_init(&pt_ctx.barrier, NULL, pt_ctx.n_threads);
    pt_ctx.pthreads = malloc(pt_ctx.n_threads * sizeof(pthread_t));

    i64 L1_size           = get_cache_size(1);
    L1_size               = (L1_size < 0) ? 4096 : L1_size;
    i64 cache_line_size   = get_cache_line_size(1);
    cache_line_size       = (cache_line_size < 0) ? 64 : cache_line_size;
    i64 n_lines_per_cache = L1_size / cache_line_size;
    i64 domain_size       = (sim_params.N + 2) * (sim_params.N + 2);
    i64 n_cache_fills     = (domain_size / L1_size) + ((domain_size % L1_size != 0) ? 1 : 0);

    size_t bytes_per_row  = (sim_params.N + 2) * sizeof(f64);
    i64    rows_per_cache = L1_size / bytes_per_row + ((L1_size % bytes_per_row != 0) ? 1 : 0);
    i64    lines_per_row
        = bytes_per_row / cache_line_size + ((bytes_per_row % cache_line_size != 0) ? 1 : 0);

    i64 rows_per_thread = (sim_params.N + 2) / pt_ctx.n_threads;
    i64 remaining_rows  = (sim_params.N + 2) % pt_ctx.n_threads;

    IsaLogDebug("L1 size (%ld), cache line size (%ld), lines per cache (%ld), cache fills for "
                "domain (%ld)\n\tbytes per row (%zd), rows per cache (%ld), lines per row (%ld), "
                "rows per thread (%ld), remaining rows (%ld)",
                L1_size, cache_line_size, n_lines_per_cache, n_cache_fills, bytes_per_row,
                rows_per_cache, lines_per_row, rows_per_thread, remaining_rows);

    pt_ctx.sim_contexts = malloc(pt_ctx.n_threads * sizeof(PtSimContext));
    for(i64 i = 0; i < pt_ctx.n_threads; ++i) {
        pt_ctx.sim_contexts[i].t_id      = i + 1;
        pt_ctx.sim_contexts[i].row_start = i * rows_per_thread;
        pt_ctx.sim_contexts[i].row_end   = i * rows_per_thread + rows_per_thread;
        IsaLogDebug("Thread %ld starts at row %ld and ends at row %ld", i + 1, i * rows_per_thread,
                    i * rows_per_thread + rows_per_thread);
    }
}

static void
pt_ctx_deinitialize(void)
{
    pthread_barrier_destroy(&pt_ctx.barrier);
}

static void
start_simulation(void)
{
    for(i64 i = 0; i < pt_ctx.n_threads; ++i) {
        pthread_create(&pt_ctx.pthreads[i], NULL, simulate, &pt_ctx.sim_contexts[i]);
    }
    for(i64 i = 0; i < pt_ctx.n_threads; ++i) {
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

#if 0
    // Set up the initial state of the domain
    domain_initialize();

    // Time the execution
    gettimeofday(&t_start, NULL);

    // TASK: T2
    // Run the integration loop
    // BEGIN: T2
    start_simulation();
    // END: T2

    // Report how long we spent in the integration stage
    gettimeofday(&t_end, NULL);
    printf("%lf seconds elapsed with %ld threads\n", WALLTIME(t_end) - WALLTIME(t_start),
           pt_ctx.n_threads);

    // Clean up and shut down
    domain_finalize();
#endif
    // TASK: T1d
    // Finalise pthreads
    // BEGIN: T1d
    pt_ctx_deinitialize();
    // END: T1d

    exit(EXIT_SUCCESS);
}
