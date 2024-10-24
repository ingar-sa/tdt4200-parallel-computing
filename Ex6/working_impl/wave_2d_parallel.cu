// #define _XOPEN_SOURCE 600
// I get a compiler warning that macro is already defined in the cuda headers, so i've commented it
// out
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if DO_DEBUG == 1
#define PrintfDbg(...) printf(__VA_ARGS__)
#else
#define PrintfDbg(...)
#endif

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
//  END: T1

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef float   real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

// Based on the output from the device info, max threads per block is 1024, which is 32x32
// This could be determined dynamically in init_cuda, but I don't see the point since we know which
// GPUs we're using
#define BLOCKY 32
#define BLOCKX 32

// Simulation parameters: size, step count, and how often to save the state
int_t h_N = 128, h_M = 128, h_max_iteration = 10000, h_snapshot_freq = 200;
#define SIM_DATA_SIZE ((h_M + 2) * (h_N + 2) * sizeof(real_t))
// I forgot to multiply by sizeof(real_t) one time, so I made this macro to avoid that mistake in
// the future

// Wave equation parameters, time step is derived from the space step
const real_t h_c = 1.0, h_dx = 1.0, h_dy = 1.0;
real_t       h_dt;

// Since the simulation and equation parameters are constant throughout the simulation, we can keep
// them in constant memory for fast access by all threads
__constant__ int_t  d_N, d_M, d_max_iteration, d_snapshot_freq;
__constant__ real_t d_c, d_dx, d_dy, d_dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
// real_t *buffers[3] = { NULL, NULL, NULL };
typedef struct Timesteps
{
    real_t *prv;
    real_t *cur;
    real_t *nxt;

} Timesteps;
// I've decided to make a wrapper struct to make it more clear which timestep is
// being accessed. There is one on the host for allocating, freeing, and copying the
// computational results from the GPU into host memory (h_out). The struct is copied to the GPU in
// domain_initialize
Timesteps            h_timesteps;
__device__ Timesteps d_timesteps;
// TODO(ingar): Determine if this is an appropriate way to do this, or if the buffers should just be
// passed in at kernel launch

// Buffer for the host to use when writing the results to file and access macro for it
real_t *h_out;
#define h_U(i, j) h_out[((i) + 1) * (h_N + 2) + (j) + 1]

#define U_prv(i, j) timesteps.prv[((i) + 1) * (d_N + 2) + (j) + 1]
#define U(i, j)     timesteps.cur[((i) + 1) * (d_N + 2) + (j) + 1]
#define U_nxt(i, j) timesteps.nxt[((i) + 1) * (d_N + 2) + (j) + 1]

// #define U_p(i, j) d_timesteps.prv[((i) + 1) * (d_N + 2) + (j) + 1]
// #define U_c(i, j) d_timesteps.cur[((i) + 1) * (d_N + 2) + (j) + 1]
// #define U_n(i, j) d_timesteps.nxt[((i) + 1) * (d_N + 2) + (j) + 1]

#define U_p(i, j) prv[((i) + 1) * (d_N + 2) + (j) + 1]
#define U_c(i, j) cur[((i) + 1) * (d_N + 2) + (j) + 1]
#define U_n(i, j) nxt[((i) + 1) * (d_N + 2) + (j) + 1]

cudaDeviceProp gpu;
#define BYTES_TO_KiB(bytes) ((bytes) / 1024.0)
#define BYTES_TO_MiB(bytes) ((bytes) / (1024.0 * 1024.0))
#define BYTES_TO_GiB(bytes) ((bytes) / (1024.0 * 1024.0 * 1024.0))

// END: T1b

#define cudaErrorCheck(ans)                                                                        \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) {
            exit(code);
        }
    }
}

// Save the present time step in a numbered file under 'data/'
void
domain_save(int_t step)
{
    char filename[256];
    sprintf(filename, "data/%.5ld.dat", step);
    FILE *out = fopen(filename, "wb");
    for(int_t i = 0; i < h_M; i++) {
        for(int_t j = 0; j < h_N; ++j) {
            if(h_U(i, j) > 0.01) {
                // PrintfDbg("i, j, val: %ld, %ld, %f\n", i, j, h_U(i, j));
            }
        }
        fwrite(&h_U(i, 0), sizeof(real_t), h_N, out);
    }
    fclose(out);
}

// TASK: T4
// Get rid of all the memory allocations
void
domain_finalize(void)
{
    // BEGIN: T4
    free(h_out);
    cudaFree(h_timesteps.prv);
    cudaFree(h_timesteps.cur);
    cudaFree(h_timesteps.nxt);
    // END: T4
}

// Rotate the time step buffers.
#if 0
__device__ void
d_move_buffer_window()
{
    real_t *temp    = d_timesteps.prv;
    d_timesteps.prv = d_timesteps.cur;
    d_timesteps.cur = d_timesteps.nxt;
    d_timesteps.nxt = temp;
}
#endif

void
h_move_buffer_window(void)
{
    real_t *temp    = h_timesteps.prv;
    h_timesteps.prv = h_timesteps.cur;
    h_timesteps.cur = h_timesteps.nxt;
    h_timesteps.nxt = temp;
}

// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
__device__ void
d_boundary_condition(int t_i, int t_j, real_t *cur)
{
    if(t_j == 0) {
        U_c(t_i, -1) = U_c(t_i, 1);
        // PrintfDbg("Thread %d, %d performing boundary condition\n", t_i, t_j);
    }

    if(t_j == (d_N - 1)) {
        // PrintfDbg("Thread %d, %d performing boundary condition\n", t_i, t_j);
        U_c(t_i, d_N) = U_c(t_i, d_N - 2);
    }

    if(t_i == 0) {
        // PrintfDbg("Thread %d, %d performing boundary condition\n", t_i, t_j);
        U_c(-1, t_j) = U_c(1, t_j);
    }

    if(t_i == (d_M - 1)) {
        // PrintfDbg("Thread %d, %d performing boundary condition\n", t_i, t_j);
        U_c(d_M, t_j) = U_c(d_M - 2, t_j);
    }
}
// END: T6

// TASK: T5
// Integration formula
// BEGIN; T5

#if 0
__global__ void
d_time_step(void)
{
    cg::thread_block tb = cg::this_thread_block();

    dim3 b_idx = tb.group_index();
    dim3 b_dim = tb.group_dim();
    dim3 t_idx = tb.thread_index();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    if(tb.thread_rank() == 0) {
        // printf("Performing time steps\n");
    }
    for(int_t iteration = 0; iteration < d_snapshot_freq; ++iteration) {
        tb.sync();
        d_boundary_condition(t_i, t_j);

        tb.sync();
        U_n(t_i, t_j) = -U_p(t_i, t_j) + 2.0 * U_c(t_i, t_j)
                      + (d_dt * d_dt * d_c * d_c) / (d_dx * d_dy)
                            * (U_c(t_i - 1, t_j) + U_c(t_i + 1, t_j) + U_c(t_i, t_j - 1)
                               + U_c(t_i, t_j + 1) - 4.0 * U_c(t_i, t_j));

        tb.sync();
        if(tb.thread_rank() == 0) {
            d_move_buffer_window();
        }
    }
}
#endif

#if 0
__global__ void
d_time_step(void)
{
    cg::thread_block tb = cg::this_thread_block();

    dim3 b_idx = tb.group_index();
    dim3 b_dim = tb.group_dim();
    dim3 t_idx = tb.thread_index();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    tb.sync();
    d_boundary_condition(t_i, t_j);

    tb.sync();
    U_n(t_i, t_j) = -U_p(t_i, t_j) + 2.0 * U_c(t_i, t_j)
                  + (d_dt * d_dt * d_c * d_c) / (d_dx * d_dy)
                        * (U_c(t_i - 1, t_j) + U_c(t_i + 1, t_j) + U_c(t_i, t_j - 1)
                           + U_c(t_i, t_j + 1) - 4.0 * U_c(t_i, t_j));

    tb.sync();
    if(tb.thread_rank() == 0) {
        d_move_buffer_window();
    }
}
#endif


#if 1
__global__ void
d_time_step(real_t *prv, real_t *cur, real_t *nxt)
{
    cg::thread_block tb = cg::this_thread_block();

    dim3 b_idx = tb.group_index();
    dim3 b_dim = tb.group_dim();
    dim3 t_idx = tb.thread_index();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    tb.sync();
    d_boundary_condition(t_i, t_j, cur);

    tb.sync();
    U_n(t_i, t_j) = -U_p(t_i, t_j) + 2.0 * U_c(t_i, t_j)
                  + (d_dt * d_dt * d_c * d_c) / (d_dx * d_dy)
                        * (U_c(t_i - 1, t_j) + U_c(t_i + 1, t_j) + U_c(t_i, t_j - 1)
                           + U_c(t_i, t_j + 1) - 4.0 * U_c(t_i, t_j));
    tb.sync();
}
#endif

// END: T5

// TASK: T7
// Main time integration.
// TODO(ingar): Make everything run on gpu then copy data to host asynchronously and call save
// function?
void
simulate(void)
{
    // BEGIN: T7
    // Go through each time step
    int_t grid_x = h_N / BLOCKX;
    int_t grid_y = h_M / BLOCKY;
    dim3  block(BLOCKX, BLOCKY);
    dim3  grid(grid_x, grid_y);

    for(int_t iteration = 0; iteration <= h_max_iteration; iteration += 1) {
        if((iteration % h_snapshot_freq) == 0) {
            // PrintfDbg("Saving domain %ld\n", iteration);
            //  cudaMemcpyToSymbol(&h_timesteps, &d_timesteps, sizeof(Timesteps),
            //  cudaMemcpyDeviceToHost);

            cudaMemcpy(h_out, h_timesteps.cur, SIM_DATA_SIZE, cudaMemcpyDeviceToHost);
            domain_save(iteration / h_snapshot_freq);
        }

        // Derive step t+1 from steps t and t-1
        // boundary_condition();
        // printf("Performing time step\n");
        d_time_step<<<grid, block>>>(h_timesteps.prv, h_timesteps.cur, h_timesteps.nxt);

        // Rotate the time step buffers
        h_move_buffer_window();
    }
}
// END: T7


// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation

__global__ void
d_init_timesteps(real_t *prv, real_t *cur)
{
    cg::thread_block tb = cg::this_thread_block();

    dim3 b_idx = tb.group_index();
    dim3 b_dim = tb.group_dim();
    dim3 t_idx = tb.thread_index();

    int t_i = b_idx.x * b_dim.x + t_idx.x;
    int t_j = b_idx.y * b_dim.y + t_idx.y;

    real_t delta = sqrt(((t_i - d_M / 2.0) * (t_i - d_M / 2.0)) / (real_t)d_M
                        + ((t_j - d_N / 2.0) * (t_j - d_N / 2.0)) / (real_t)d_N);

    real_t val    = exp(-4.0 * delta * delta);
    U_p(t_i, t_j) = U_c(t_i, t_j) = val;

    if(val > 0.1) {
        // PrintfDbg("U(%d, %d): %f\n", t_i, t_j, U(t_i, t_j));
    }
}

// TASK: T8
// GPU occupancy
void
occupancy(void)
{
    // BEGIN: T8
    ;
    // END: T8
}

// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool
init_cuda(void)
{
    // BEGIN: T2
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("Device count: %d\n", dev_count);

    if(dev_count <= 0) {
        return false;
    } else {
        cudaSetDevice(0);
    }

    for(int i = 0; i < dev_count; ++i) {
        cudaError_t ret = cudaGetDeviceProperties(&gpu, i);
        if(ret == cudaErrorInvalidDevice) {
            return false;
        }

        printf("Name: %s\n", gpu.name);
        printf("Compute capability: %d.%d\n", gpu.major, gpu.minor);
        printf("Multiprocessors: %d\n", gpu.multiProcessorCount);
        printf("Warp size: %d\n", gpu.warpSize);
        printf("Global memory: %.3fGiB\n", BYTES_TO_GiB(gpu.totalGlobalMem));
        printf("Per-block shared memory: %.3fKiB\n", BYTES_TO_KiB(gpu.sharedMemPerBlock));
        printf("Per-block registers: %d\n", gpu.regsPerBlock);
        printf("\nMax threads per block: %d\n", gpu.maxThreadsPerBlock);
        printf("Max threads dim: %d x %d x %d\n", gpu.maxThreadsDim[0], gpu.maxThreadsDim[1],
               gpu.maxThreadsDim[2]);
        printf("Max gid size: %d x %d x %d\n", gpu.maxGridSize[0], gpu.maxGridSize[1],
               gpu.maxGridSize[2]);
    }

    return true;
    // END: T2
}
void
domain_initialize(void)
{
    // BEGIN: T3
    bool locate_cuda = init_cuda();
    if(!locate_cuda) {
        printf("Failed to init CUDA\n");
        exit(EXIT_FAILURE);
    }

    // Buffer on the host for writing the results to file
    h_out = (real_t *)malloc(SIM_DATA_SIZE);

    // Allocate the memory so that it is referenced by the host-side Timesteps struct, and then
    // copy the result to the GPU
    cudaMalloc((void **)&h_timesteps.prv, SIM_DATA_SIZE);
    cudaMalloc((void **)&h_timesteps.cur, SIM_DATA_SIZE);
    cudaMalloc((void **)&h_timesteps.nxt, SIM_DATA_SIZE);
    cudaMemcpyToSymbol(d_timesteps, &h_timesteps, sizeof(Timesteps));
    // cudaMemcpyToSymbol defaults to cudaMemcpyHostToDevice

    // Set the time step for 2D case
    h_dt = h_dx * h_dy / (h_c * sqrt(h_dx * h_dx + h_dy * h_dy));

    // Copy all of the variables for the simulation and equation to the GPU
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(int_t));
    cudaMemcpyToSymbol(d_M, &h_M, sizeof(int_t));
    cudaMemcpyToSymbol(d_max_iteration, &h_max_iteration, sizeof(int_t));
    cudaMemcpyToSymbol(d_snapshot_freq, &h_snapshot_freq, sizeof(int_t));

    cudaMemcpyToSymbol(d_c, &h_c, sizeof(real_t));
    cudaMemcpyToSymbol(d_dx, &h_dx, sizeof(real_t));
    cudaMemcpyToSymbol(d_dy, &h_dy, sizeof(real_t));
    cudaMemcpyToSymbol(d_dt, &h_dt, sizeof(real_t));

    int_t grid_x = h_N / BLOCKX;
    int_t grid_y = h_M / BLOCKY;
    // PrintfDbg("grid_x (%ld), grid_y (%ld)\n", grid_x, grid_y);

    dim3 block(BLOCKX, BLOCKY);
    dim3 grid(grid_x, grid_y);
    d_init_timesteps<<<grid, block>>>(h_timesteps.prv, h_timesteps.cur);
    //  d_init_timesteps<<<grid, block>>>();

    // END: T3
}

int
main(void)
{
    // Set up the initial state of the domain
    init_cuda();
    domain_initialize();
    // domain_save(0);
    PrintfDbg("Domain succesfully initialized. Running simulation\n");

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    simulate();
    gettimeofday(&t_end, NULL);

    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));

#if 0
    occupancy();
#endif

    // Clean up and shut down
    domain_finalize();
    PrintfDbg("Domain succesfully finalized. The simulation has been completed\n");
    exit(EXIT_SUCCESS);
}
