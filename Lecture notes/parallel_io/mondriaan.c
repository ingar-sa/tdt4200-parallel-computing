#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

// Indices into a 5-color palette
typedef enum { RED=0, WHITE=1, OFFWHITE=2, YELLOW=3, BLUE=4 } color_t;

// A palette to paint with
uint8_t colors[5][3] = {
    { 255, 0, 0 },
    { 255, 255, 255 },
    { 224, 224, 224 },
    { 255, 224, 0  },
    { 0, 0, 255 }
};

// A struct for containing the vital data of a rectangle in our surface
typedef struct {
    int x, y, width, height;
    color_t color;
} section_t;

// Layout of our picture
const section_t sections[7] = {
    { .x = 0,   .y = 0,   .width = 400, .height = 200, .color = RED },
    { .x = 400, .y = 0,   .width = 230, .height = 200, .color = WHITE },
    { .x = 0,   .y = 200, .width = 400, .height = 600, .color = WHITE },
    { .x = 400, .y = 200, .width = 230, .height = 550, .color = OFFWHITE },
    { .x = 0,   .y = 800, .width = 100, .height = 100, .color = YELLOW },
    { .x = 100, .y=800,   .width = 300, .height = 100, .color = WHITE },
    { .x = 400, .y=750,   .width = 230, .height = 150, .color = BLUE }
};

// Canvas size and pointer
const int
    width  = 630,
    height = 900;
uint8_t
    *canvas = NULL;

// MPI parameters
int
    size,
    rank;


int
main ( int argc, char **argv )
{
    // Start MPI, as per usual
    MPI_Init ( &argc, &argv );
    MPI_Comm_size ( MPI_COMM_WORLD, &size );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

    // This program is only meant to juggle some coordinate systems,
    //  so its scalability is irrelevant
    if ( size < 7 )
    {
        if ( rank == 0 )
            fprintf ( stderr,
                "This program needs at least 7 ranks (and won't use more)\n"
            );
        MPI_Finalize();
        exit ( EXIT_FAILURE );
    }

    // Data types for pixels, local array of pixels, and global sub-area
    MPI_Datatype triplet, my_data, my_area;
    // Pick coordinates from the image "recipe", by rank
    int global_size[2]  = { height, width };
    int local_size[2]   = { sections[rank].height, sections[rank].width };
    int local_origin[2] = { sections[rank].y, sections[rank].x };

    // Allocate our slice of the surface in memory
    canvas = malloc ( local_size[0]*local_size[1]*3*sizeof(uint8_t) );
    #define CANVAS(y,x,c) canvas[(y)*local_size[1]*3+(x)*3+(c)]
    memset ( canvas, 0, local_size[0]*local_size[1]*3*sizeof(uint8_t) ); 

    // Color the area, leaving a black border
    for ( int y=1; y<local_size[0]-1; y++ )
        for ( int x=1; x<local_size[1]-1; x++ )
            for ( int c=0; c<3; c++ )
                CANVAS(y,x,c) = colors[sections[rank].color][c];

    // Make types for the local representation of the area, and the global

    // First, this is one pixel (three color values):
    MPI_Type_contiguous ( 3, MPI_UINT8_T, &triplet );
    MPI_Type_commit ( &triplet );

    // The local representation is just a contiguous array of color values
    MPI_Type_contiguous ( local_size[0]*local_size[1], triplet, &my_data );
    MPI_Type_commit ( &my_data );

    // For the global coordinate space, we fill in only our sub-area
    MPI_Type_create_subarray (
        2, global_size, local_size, local_origin,
        MPI_ORDER_C, triplet, &my_area
    );
    MPI_Type_commit ( &my_area );

    // Open a shared file where everyone can contribute
    const char *filename = "composition.data";
    MPI_File out;
    MPI_File_open (
        MPI_COMM_WORLD, filename, MPI_MODE_CREATE|MPI_MODE_WRONLY,
        MPI_INFO_NULL, &out
    );

    // Restrict writing from this rank to the sub-area we defined above
    MPI_File_set_view (
        out, 0, triplet, my_area, "native", MPI_INFO_NULL
    );

    // Everyone writes in their own window
    MPI_File_write_all (
        out, canvas, 1, my_data, MPI_STATUS_IGNORE
    );

    // Close it up, and finish
    MPI_File_close ( &out );
    free ( canvas );
    MPI_Finalize ();
    exit ( EXIT_SUCCESS );
}
