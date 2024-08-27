#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

typedef struct
{
    uchar *Bitmap;
    int    X, Y;
} image;

/*
 * Copies the color from one pixel to another
 */
static inline void
CopyColor(uchar *From, uchar *To)
{
    *To++ = *From++;
    *To++ = *From++;
    *To++ = *From++;
}

static void
DoubleImageSize(image *Image)
{
    int    NewX      = Image->X * 2;
    int    NewY      = Image->Y * 2;
    uchar *NewBitmap = calloc(NewX * NewY * 3, sizeof(uchar));

    // The number of uchars for each row in the bitmap.
    // Used to move the row pointers one row down
    int XStride = NewX * 3;

    // Interpolating one pixel in the original bitmap is done by copying its color
    // to the now four pixels in the new bitmap. Therefore, we advance two pixels in
    // in two rows in the new bitmap per one pixel in a row in the original
    uchar *Pixel  = Image->Bitmap;
    uchar *NBRow1 = NewBitmap;
    uchar *NBRow2 = NewBitmap + XStride;

    // This could be collapsed into a single for-loop, but I believe the logic for
    // determining when to advance the rows in the new bitmap (using modulo, for example)
    // might add more overhead than the simple less-than comparison of the for-loops.
    for(int Y = 0; Y < Image->Y; ++Y)
    {
        for(int X = 0; X < Image->X; ++X)
        {
            // Copy the color of the pixel to the two pixels in the first row
            // of the new bitmap
            CopyColor(Pixel, NBRow1);
            CopyColor(Pixel, NBRow1 + 3);

            // Same as above, but for the second row
            CopyColor(Pixel, NBRow2);
            CopyColor(Pixel, NBRow2 + 3);

            // Advance the pointers to the next un-copied pixels
            Pixel += 3;
            NBRow1 += 6;
            NBRow2 += 6;
        }

        // Advance the pointers to the next un-copied rows
        NBRow1 += XStride;
        NBRow2 += XStride;
    }

    Image->X      = NewX;
    Image->Y      = NewY;
    Image->Bitmap = NewBitmap;
}

void
RecolorImage(image *Image)
{
    uchar *Pixel = Image->Bitmap;
    uchar  R, G, B;

    // Rotates the pixel values: R->B, G->R, B->G.
    for(int i = 0; i < Image->X * Image->Y; ++i)
    {
        R = *Pixel;
        G = *(Pixel + 1);
        B = *(Pixel + 2);

        *Pixel       = B;
        *(Pixel + 1) = R;
        *(Pixel + 2) = G;

        Pixel += 3;
    }
}

int
main(void)
{
    uchar *Bitmap = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
    readbmp("before.bmp", Bitmap);

    image Image = { Bitmap, XSIZE, YSIZE };

    RecolorImage(&Image);
    DoubleImageSize(&Image);

    savebmp("after.bmp", Image.Bitmap, Image.X, Image.Y);

    // This is strictly not necessary here since the program ends right after anyways,
    // but it's good practice to always free pointers.
    free(Bitmap);
    free(Image.Bitmap);

    return 0;
}
