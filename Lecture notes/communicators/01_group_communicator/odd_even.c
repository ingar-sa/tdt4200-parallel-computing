#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// We can divide the world communicator into its even and odd subsets
int
    rank_world,         // Rank in the world comm.
    size_world,         // Number of ranks in total
    rank_subset,        // Rank within the subset each participant belongs to
    size_subset,        // Number of ranks in the subset
    reporting_rank = 0; // Rank 0 reports if nothing else is specified

// Groups are just sets of ranks, we can make communicators from them
MPI_Group
    world,
    odds,
    evens;

// Communicators for even and odd subsets of the world communicator
// Since each rank belongs to only one of these, the other will
//  be set to MPI_COMM_NULL by ranks that don't belong there
MPI_Comm
    even_comm,
    odd_comm;


int
main ( int argc, char **argv )
{
    // If we have a command line argument that we want the status report
    //  to come from a different rank than 0, then read it.
    if ( argc > 1 )
        reporting_rank = strtol ( argv[1], NULL, 10 );

    // Fire up MPI, as usual...
    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank_world );
    MPI_Comm_size ( MPI_COMM_WORLD, &size_world );

    // Make a group out of all ranks, so we can extract subsets from it
    MPI_Comm_group ( MPI_COMM_WORLD, &world );

    // How many (world) ranks will go into each group?
    int
        n_evens = size_world/2+(size_world%2),
        n_odds  = size_world/2;
    // Arrays of WHICH (world) ranks go into each group
    int
        even_ranks[n_evens],
        odd_ranks[n_odds];

    // Work out the indices
    for ( int r=0; r<n_evens; r++ )
        even_ranks[r] = r*2;
    for ( int r=0; r<n_odds; r++ )
        odd_ranks[r] = r*2+1;

    // Make groups out of the index-lists
    MPI_Group_incl ( world, n_evens, even_ranks, &evens );
    MPI_Group_incl ( world, n_odds, odd_ranks, &odds );

    // Make communicators out of both groups.
    // At this point, the communicator that our present rank is excluded
    //  from will come back empty (i.e. equal to MPI_COMM_NULL)
    MPI_Comm_create_group ( MPI_COMM_WORLD, evens, n_evens, &even_comm );
    MPI_Comm_create_group ( MPI_COMM_WORLD, odds , n_odds , &odd_comm );

    // Check that all this stuff worked, but only report from one rank,
    //  so as not to fill the display with too much text at a time:
    if ( rank_world == reporting_rank )
    {
        printf ( "Hello from world rank %d\n", rank_world );

        // If the even_comm is valid, we're in it:
        if ( even_comm != MPI_COMM_NULL )
        {
            MPI_Comm_rank ( even_comm, &rank_subset );
            MPI_Comm_size ( even_comm, &size_subset );
            printf (
                "I am rank %d in the communicator of %d even world ranks\n",
                rank_subset, size_subset
            );
        }
        else
            printf ( "I am not among the even world ranks\n" );

        // Otherwise, the odd_comm should be valid, and we're in it:
        if ( odd_comm != MPI_COMM_NULL )
        {
            MPI_Comm_rank ( odd_comm, &rank_subset );
            MPI_Comm_size ( odd_comm, &size_subset );
            printf (
                "I am rank %d in the communicator of %d odd world ranks\n",
                rank_subset, size_subset
            );
        }
        else
            printf ( "I am not among the odd world ranks\n" );

        // If the setup of these communicators was done correctly,
        //  one and only one of these conditionals should be true for
        //  each rank.
        // Since the code tests both communicators separately, any mistakes
        //  in the splitting logic will show up as ranks that either
        //  report membership in both communicators, or that don't report
        //  at all.
    }

    MPI_Finalize ();
    exit ( EXIT_SUCCESS );
}
