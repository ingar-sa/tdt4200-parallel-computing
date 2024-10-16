#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <omp.h>

int
main ()
{
    const int height = 6000, width = 8000;                  // Image size

    #define SET(y,x,c) set[(y)*width*3+(x)*3+(c)]           // Indexing macro
    char *set = malloc ( width*height*3*sizeof(char) );     // Set array

    // Time the computation
    double
        t_start = omp_get_wtime(),
        t_end;
    // For all x/y pairs (parallelize by y/lines)
    #pragma omp parallel for
    for ( int y=0; y<height; y++ )
    {
        for ( int x=0; x<width; x++ )
        {
            // Find complex coordinates of the seed
            float complex c = (2.0*x-width)/width-0.5 + I*(2.0*y-height)/height;
            float complex z = c;
            int k;
            for ( k=0; k<128 && cabsf(z) < 2.0; k++ )   // Test the point
                z = z * z + c;
            SET(y,x,0) = SET(y,x,1) = SET(y,x,2) = k;   // Store result
        }
    }
    t_end = omp_get_wtime();
    printf ( "Calculated for %lf seconds\n", t_end-t_start );

    // Save the result in a PPM file
    FILE *out = fopen ( "output.ppm", "w" );
    fprintf ( out, "P6 %d %d 127\n", width, height );
    fwrite ( set, width*height*3, sizeof(char), out );
    fclose ( out );

    // Free the set array and close shop
    free ( set );
    exit ( EXIT_SUCCESS );
}
