#include "isa.h"
#include "datatypes.h"

#include <mpi.h>

static void
print_mpi_context(const MpiCtx *context)
{
    printf("My rank: %ld\n", context->my_rank);
    printf("Commsize: %ld\n", context->commsize);
    printf("N children: %ld\n", context->n_children);
    printf("I am root rank: %s\n", context->i_am_root_rank ? "true" : "false");
    printf("I am first rank: %s\n", context->i_am_first_rank ? "true" : "false");
    printf("I am last rank: %s\n", context->i_am_last_rank ? "true" : "false");
    printf("I am o: %s\n", context->i_am_only_child ? "true" : "false");
    printf("ThereIsOneChild: %s\n", context->there_is_one_child ? "true" : "false");
    printf("CellsPerRank: %ld\n", context->cells_per_rank);
    printf("RemainingCells: %ld\n", context->remaining_cells);
    printf("NMyCells: %ld\n", context->n_my_cells);
    printf("RecvCounts: %p\n", (void *)context->recv_counts);
    printf("Displacements: %p\n", (void *)context->displacements);
    printf("\n");
}

static void
print_all_mpi_contexts(MpiCtx *mpi_ctx)
{
    MPI_Datatype Mpi_MpiCtx;
    int          n_struct_members = 13; // - RecvCounts and Displacements
    int          member_blocks[]  = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    MPI_Aint     member_displacements[]
        = { offsetof(MpiCtx, my_rank),         offsetof(MpiCtx, commsize),
            offsetof(MpiCtx, n_children),      offsetof(MpiCtx, i_am_root_rank),
            offsetof(MpiCtx, i_am_first_rank), offsetof(MpiCtx, i_am_last_rank),
            offsetof(MpiCtx, i_am_only_child), offsetof(MpiCtx, there_is_one_child),
            offsetof(MpiCtx, cells_per_rank),  offsetof(MpiCtx, remaining_cells),
            offsetof(MpiCtx, n_my_cells),      offsetof(MpiCtx, recv_counts),
            offsetof(MpiCtx, displacements) };

    MPI_Datatype member_types[]
        = { MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_C_BOOL,  MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL,
            MPI_C_BOOL,  MPI_INT64_T, MPI_INT64_T, MPI_INT64_T, MPI_AINT,   MPI_AINT };

    MPI_Type_create_struct(n_struct_members, member_blocks, member_displacements, member_types,
                           &Mpi_MpiCtx);
    MPI_Type_commit(&Mpi_MpiCtx);

    if(mpi_ctx->i_am_root_rank) {
        print_mpi_context(mpi_ctx);

        MpiCtx rank_context = { 0 };
        for(int i = 1; i < mpi_ctx->commsize; ++i) {
            MPI_Recv(&rank_context, 1, Mpi_MpiCtx, i, MPI_ANY_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            print_mpi_context(&rank_context);
        }
    } else {
        MPI_Send(&mpi_ctx, 1, Mpi_MpiCtx, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&Mpi_MpiCtx);
}
