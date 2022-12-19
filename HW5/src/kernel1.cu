#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WITDH 16

// add by myself
// copy from serial
__device__ int mandel(float x, float y, int maxIterations)
{
    float z_re = x, z_im = y;
    int k;

    for (k = 0; k < maxIterations; ++k)
    {
        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }

    return k;
}
//

__global__ void mandelKernel(int* dimg, float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // add by myself
    // indexing, x is horizontal, y is vertical
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + i * stepX;
    float y = lowerY + j * stepY;
    dimg[j * resX + i] = mandel(x, y, maxIterations);
    //
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // add by myself
    int *temp_himg = (int*) malloc(sizeof(int) * resX * resY); // because of hw require

    // declare gpu var.
    int *dimg;

    // allocate gpu memory
    cudaMalloc(&dimg, sizeof(int) * resX * resY);

    // call gpu kernel func.
    dim3 dimGrid(resX / BLOCK_WITDH, resY / BLOCK_WITDH), imBlock(BLOCK_WITDH, BLOCK_WITDH);
    mandelKernel<<<dimGrid, imBlock>>>(dimg, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    // copy ans from gpu to host
    cudaMemcpy(temp_himg, dimg, sizeof(int) * resX * resY, cudaMemcpyDeviceToHost);

    // copy data to result (because of hw require)
    memcpy(img, temp_himg, sizeof(int) * resX * resY);

    // free gpu memory
    cudaFree(dimg);
    //
}
