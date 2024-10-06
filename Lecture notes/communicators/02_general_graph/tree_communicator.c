#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Indexing formulas for the position of a node in a binary tree
// These are only used to construct the communicator
int n_neighbors ( int rank, int size );
int parent ( int rank, int size );
int child_left  ( int rank, int size );
int child_right ( int rank, int size );

// Demonstration that the communicator is correctly configured:
//  each rank prints its list of neighbors to a separate file,
//  the collection of files can be concatenated into a file that
//  we can plot a picture of, using the 'dot' tool afterwards.
// ('dot' is part of the GraphicsMagick utility collection)
void neighbors_print ( MPI_Comm comm );


int
main ( int argc, char **argv )
{
    // Initialize MPI, get world rank and size
    int
        rank_world,
        size_world;
    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank_world );
    MPI_Comm_size ( MPI_COMM_WORLD, &size_world );

    // A binary tree-shaped graph communicator needs two main ingredients:
    //  * a list of all neighbor relationships in the whole graph, and
    //  * where in that list to find the neighbors of one particular rank.
    // We will start by creating the list of each rank's index in the list
    //  of edges, which for each rank will be the sum of the neighbor counts
    //  from lower ranks.

    int index[size_world];
    index[0] = n_neighbors(0, size_world);
    for ( int r=1; r<size_world; r++ )
        index[r] = index[r-1] + n_neighbors( r,size_world );

    // We can now create the list of edges.
    // Since we are constructing a binary tree, each node will have at most
    //  three neighbors (one parent and two children).
    int n_edges = index[size_world-1] + 1;
    int edges[n_edges];
    if ( size_world > 1 )
        edges[0] = 1;
    if ( size_world > 2 )
        edges[1] = 2;
    for ( int r=1; r<size_world; r++ )
    {
        int idx    = index[r-1],
            up     = parent      ( r, size_world ),
            left   = child_left  ( r, size_world ),
            right  = child_right ( r, size_world );

        edges[idx] = up;
        if ( left < size_world )
            edges[idx+1] = left;
        if ( right < size_world )
            edges[idx+2] = right;
    }

    // Make a communicator out of this graph description:
    MPI_Comm my_tree;
    MPI_Graph_create ( MPI_COMM_WORLD, size_world, index, edges, 0, &my_tree );

    // Having constructed the graph, we can demonstrate that it
    //  contains all the relevant information, so that the rest of the
    //  program can use it without dealing with all the indexing we put
    //  into it.
    neighbors_print ( my_tree );

    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


// Print a file containing only the neighborhood information from the
//  rank that calls this function
void
neighbors_print ( MPI_Comm the_tree )
{
    // Find my rank, and my number of graph neighbors
    int me, n_neighbors;
    MPI_Comm_rank ( the_tree, &me );
    MPI_Graph_neighbors_count ( the_tree, me, &n_neighbors );

    // Get the ranks of my neighbors
    int neighbors[n_neighbors];
    MPI_Graph_neighbors ( the_tree, me, n_neighbors, neighbors );

    // Print the edges into a text file that is named after my rank
    char filename[256];
    sprintf ( filename, "neighbors_%.2d.txt", me );
    FILE *out = fopen ( filename, "w" );
    for ( int n=0; n<n_neighbors; n++ )
        fprintf ( out, "\t%d -> %d\n", me, neighbors[n] );
    fclose ( out );
}


int parent ( int rank, int size )      { return ((rank+1)/2) - 1; }
int child_left ( int rank, int size )  { return ((rank+1)*2) - 1; }
int child_right ( int rank, int size ) { return ((rank+1)*2);     }


int
n_neighbors ( int rank, int size )
{
    int count = (rank>0) ? 1 : 0;
    if ( child_left(rank,size) < size )
        count += 1;
    if ( child_right(rank,size) < size )
        count += 1;
    return count;
}
