#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT_RANK 0

typedef struct
{
    int num_points;
    // Other simulation parameters...
} SimulationContext;

int
main(int argc, char **argv)
{
    int               rank, size;
    SimulationContext context;
    double           *local_data, *global_data = NULL;
    int              *recvcounts = NULL, *displs = NULL;
    int               total_points = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize context for each rank
    // In a real scenario, this would be set based on your simulation setup
    context.num_points = (rank == size - 1) ? 110 : 100; // Last rank has more points

    // Allocate and initialize local data
    local_data = (double *)malloc(context.num_points * sizeof(double));
    for(int i = 0; i < context.num_points; i++) {
        local_data[i] = rank + i * 0.1; // Just for demonstration
    }

    // Root rank allocates arrays for gathering information
    if(rank == ROOT_RANK) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));
    }

    // Gather the number of points from each rank
    MPI_Gather(&context.num_points, 1, MPI_INT, recvcounts, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    // Root rank calculates displacements and total points
    if(rank == ROOT_RANK) {
        displs[0]    = 0;
        total_points = recvcounts[0];
        for(int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
            total_points += recvcounts[i];
        }
        global_data = (double *)malloc(total_points * sizeof(double));
    }

    // Gather the data using Gatherv
    MPI_Gatherv(local_data, context.num_points, MPI_DOUBLE, global_data, recvcounts, displs,
                MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

    // Root rank now has all data in global_data
    if(rank == ROOT_RANK) {
        printf("Total points gathered: %d\n", total_points);
        for(int i = 0; i < total_points; i++) {
            printf("%.1f ", global_data[i]);
        }
        printf("\n");
    }

    // Clean up
    free(local_data);
    if(rank == ROOT_RANK) {
        free(recvcounts);
        free(displs);
        free(global_data);
    }

    MPI_Finalize();
    return 0;
}
