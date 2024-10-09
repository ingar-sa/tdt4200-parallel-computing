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

// END: T1a

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// TASK: T1b
// Pthread management
// BEGIN: T1b
int_t n_threads = 1;
// END: T1b

// Performance measurement
struct timeval t_start, t_end;
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Simulation parameters: size, step count, and how often to save the state
const int_t N = 1024, max_iteration = 4000, snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t c = 1.0, h = 1.0;
real_t dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t *buffers[3] = {NULL, NULL, NULL};

#define U_prv(i, j) buffers[0][((i) + 1) * (N + 2) + (j) + 1]
#define U(i, j) buffers[1][((i) + 1) * (N + 2) + (j) + 1]
#define U_nxt(i, j) buffers[2][((i) + 1) * (N + 2) + (j) + 1]

// Rotate the time step buffers.
void move_buffer_window(void) {
  real_t *temp = buffers[0];
  buffers[0] = buffers[1];
  buffers[1] = buffers[2];
  buffers[2] = temp;
}

// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize(void) {
  buffers[0] = malloc((N + 2) * (N + 2) * sizeof(real_t));
  buffers[1] = malloc((N + 2) * (N + 2) * sizeof(real_t));
  buffers[2] = malloc((N + 2) * (N + 2) * sizeof(real_t));

  for (int_t i = 0; i < N; i++) {
    for (int_t j = 0; j < N; j++) {
      real_t delta = sqrt(
          ((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) / (real_t)N);
      U_prv(i, j) = U(i, j) = exp(-4.0 * delta * delta);
    }
  }

  // Set the time step
  dt = (h * h) / (4.0 * c * c);
}

// Get rid of all the memory allocations
void domain_finalize(void) {
  free(buffers[0]);
  free(buffers[1]);
  free(buffers[2]);
}

// TASK: T3
// Integration formula
void time_step(int_t thread_id) {
  // BEGIN: T3
  for (int_t i = 0; i < N; i += 1)
    for (int_t j = 0; j < N; j++)
      U_nxt(i, j) = -U_prv(i, j) + 2.0 * U(i, j) +
                    (dt * dt * c * c) / (h * h) *
                        (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1) -
                         4.0 * U(i, j));
  // END: T3
}

// TASK: T4
// Neumann (reflective) boundary condition
void boundary_condition(int_t thread_id) {
  // BEGIN: T4
  for (int_t i = 0; i < N; i += 1) {
    U(i, -1) = U(i, 1);
    U(i, N) = U(i, N - 2);
  }
  for (int_t j = 0; j < N; j += 1) {
    U(-1, j) = U(1, j);
    U(N, j) = U(N - 2, j);
  }
  // END: T4
}

// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
  char filename[256];
  sprintf(filename, "data/%.5ld.dat", step);
  FILE *out = fopen(filename, "wb");
  for (int_t i = 0; i < N; i++)
    fwrite(&U(i, 0), sizeof(real_t), N, out);
  fclose(out);
}

// TASK: T5
// Main loop
void *simulate(void *id) {
  // BEGIN: T5
  // Go through each time step
  for (int_t iteration = 0; iteration <= max_iteration; iteration++) {
    if ((iteration % snapshot_freq) == 0) {
      domain_save(iteration / snapshot_freq);
    }

    // Derive step t+1 from steps t and t-1
    boundary_condition(0);
    time_step(0);

    // Rotate the time step buffers
    move_buffer_window();
  }
  // END: T5
}

// Main time integration loop
int main(int argc, char **argv) {
  // Number of threads is an optional argument, sanity check its value
  if (argc > 1) {
    n_threads = strtol(argv[1], NULL, 10);
    if (errno == EINVAL)
      fprintf(stderr, "'%s' is not a valid thread count\n", argv[1]);
    if (n_threads < 1) {
      fprintf(stderr, "Number of threads must be >0\n");
      exit(EXIT_FAILURE);
    }
  }

  // TASK: T1c
  // Initialise pthreads
  // BEGIN: T1c
  ;
  // END: T1b

  // Set up the initial state of the domain
  domain_initialize();

  // Time the execution
  gettimeofday(&t_start, NULL);

  // TASK: T2
  // Run the integration loop
  // BEGIN: T2
  simulate(NULL);
  // END: T2

  // Report how long we spent in the integration stage
  gettimeofday(&t_end, NULL);
  printf("%lf seconds elapsed with %ld threads\n",
         WALLTIME(t_end) - WALLTIME(t_start), n_threads);

  // Clean up and shut down
  domain_finalize();

  // TASK: T1d
  // Finalise pthreads
  // BEGIN: T1d
  ;
  // END: T1d

  exit(EXIT_SUCCESS);
}
