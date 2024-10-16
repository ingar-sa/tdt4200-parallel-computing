#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* Problem size */
#define XSIZE 2560
#define YSIZE 2048

/* Divide the problem into blocks of BLOCKX x BLOCKY threads */
#define BLOCKY 32
#define BLOCKX 32
#define MAXITER                                                                                    \
    255 // you may want to increase this
        //
double xleft  = -2.01;
double xright = 1;
double yupper, ylower;
double ycenter = 1e-6;
double step;

int host_pixel[XSIZE * YSIZE];
int device_pixel[XSIZE * YSIZE];

typedef struct
{
    double real, imag;
} my_complex_t;

#define PIXEL(i, j) ((i) + (j) * XSIZE)

// ********** SUBTASK1: Create kernel device_calculate ******************/
// Insert code here
// Hint: Use _global_ for the kernal function to be executed on the GPU.
// Also set up a single grid with a 2D thread block
// ********** SUBTASK1 END ***********************************************/
void
host_calculate()
{
    for(int j = 0; j < YSIZE; j++) {
        for(int i = 0; i < XSIZE; i++) {
            /* Calculate the number of iterations until divergence for
            each pixel.
            If divergence never happens, return MAXITER */
            my_complex_t c, z, temp;
            int          iter = 0;
            c.real            = (xleft + step * i);
            c.imag            = (yupper - step * j);
            z                 = c;
            while(z.real * z.real + z.imag * z.imag < 4.0) {
                temp.real = z.real * z.real - z.imag * z.imag + c.real;
                temp.imag = 2.0 * z.real * z.imag + c.imag;
                z         = temp;
                if(++iter == MAXITER) {
                    break;
                }
            }
            host_pixel[PIXEL(i, j)] = iter;
        }
    }
}

typedef unsigned char uchar;
// save 24-bits bmp file, buffer must be in bmp format: upside-down

void
savebmp(const char *name, uchar *buffer, int x, int y)
{
    FILE *f = fopen(name, "wb");
    if(!f) {
        printf("Error writing image to disk.\n");
        return;
    }
    unsigned int size = x * y * 3 + 54;

    uchar size_mask     = size & 255;
    uchar size_shift_8  = (size >> 8) & 255;
    uchar size_shift_16 = (size >> 16) & 255;
    uchar size_shift_24 = (size >> 24);
    uchar x_mask        = x & 255;
    uchar x_shift_8     = x >> 8;
    uchar y_mask        = y & 255;
    uchar y_shift_8     = y >> 8;

    uchar header[54] = { 'B',
                         'M',
                         size_mask,
                         size_shift_8,
                         size_shift_16,
                         size_shift_24,
                         0,
                         0,
                         0,
                         0,
                         54,
                         0,
                         0,
                         0,
                         40,
                         0,
                         0,
                         0,
                         x_mask,
                         x_shift_8,
                         0,
                         0,
                         y_mask,
                         y_shift_8,
                         0,
                         0,
                         1,
                         0,
                         24,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0 };
    fwrite(header, 1, 54, f);
    fwrite(buffer, 1, x * y * 3, f);
    fclose(f);
}
// given iteration number, set a color
void
fancycolour(uchar *p, int iter)
{
    if(iter == MAXITER)
        ;
    else if(iter < 8) {
        p[0] = 128 + iter * 16;
        p[1] = p[2] = 0;
    } else if(iter < 24) {
        p[0] = 255;
        p[1] = p[2] = (iter - 8) * 16;
    } else if(iter < 160) {
        p[0] = p[1] = 255 - (iter - 24) * 2;
        p[2]        = 255;
    } else {
        p[0] = p[1] = (iter - 160) * 2;
        p[2]        = 255 - (iter - 160) * 2;
    }
}
// Get system time to microsecond precision
// ostensibly, similar to MPI_Wtime),
// returns time in seconds
double
walltime(void)
{
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}
int
main(int argc, char **argv)
{
    if(argc == 1) {
        puts("Usage: MANDEL n");
        puts("n decides whether image should be written to disk (1=yes, 0 = no) ");
        return 0;
    }

    double start;
    double hosttime   = 0;
    double devicetime = 0;
    double memtime    = 0;

    cudaDeviceProp p;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&p, 0);
    printf("Device compute capability: %d.%d\n", p.major, p.minor);

    /* Calculate the range in the y-axis such that we preserve the
    aspect ratio */
    step   = (xright - xleft) / XSIZE;
    yupper = ycenter + (step * YSIZE) / 2;
    ylower = ycenter - (step * YSIZE) / 2;

    /* Host calculates image */
    start = walltime();
    host_calculate();
    hosttime += walltime() - start;

    //********** SUBTASK2: Set up device memory ***************************/
    // Insert code here
    /********** SUBTASK2 END **********************************************/
    start = walltime();

    //********* SUBTASK3: Execute the kernel on the device ************/
    // Insert code here
    //********** SUBTASK3 END *****************************************/
    devicetime += walltime() - start;
    start = walltime();

    //***** SUBTASK4: Transfer the result from device to device_pixel[][]*/
    // Insert code here
    //********** SUBTASK4 END ******************************************/
    memtime += walltime() - start;

    /****** SUBTASK5: Free the device memory also ************************/
    // Insert code here
    /********** SUBTASK5 END ******************************************/
    int errors = 0;

    /* check if result is correct */
    for(int i = 0; i < XSIZE; i++) {
        for(int j = 0; j < YSIZE; j++) {
            int diff = host_pixel[PIXEL(i, j)] - device_pixel[PIXEL(i, j)];
            if(diff < 0) {
                diff = -diff;
            }
            /* allow +-1 difference */
            if(diff > 1) {
                if(errors < 10) {
                    printf("Error on pixel %d %d: expected% d,found% d\n", i, j,
                           host_pixel[PIXEL(i, j)], device_pixel[PIXEL(i, j)]);
                } else if(errors == 10) {
                    puts("...");
                }
                errors++;
            }
        }
    }
    if(errors > 0) {
        printf("Found %d errors.\n", errors);
    } else {
        puts("Device calculations are correct.");
    }
    printf("\n");
    printf("Host time: %7.3f ms\n", hosttime * 1e3);
    printf("Device calculation: %7.3f ms\n", devicetime * 1e3);
    printf("Copy result: %7.3f ms\n", memtime * 1e3);
    if(strtol(argv[1], NULL, 10) != 0) {
        /* create nice image from iteration counts. take care to create it
        upside
        down (bmp format) */
        unsigned char *buffer = (unsigned char *)calloc(XSIZE * YSIZE * 3, 1);
        for(int i = 0; i < XSIZE; i++) {
            for(int j = 0; j < YSIZE; j++) {
                int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
                fancycolour(buffer + p, device_pixel[PIXEL(i, j)]);
            }
        }
        /* write image to disk */
        savebmp("mandel1.bmp", buffer, XSIZE, YSIZE);
    }
    return 0;
}
