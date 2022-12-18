#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float* dlowerX, float* dlowerY, float* dstepX, float* dstepY, int *dimg) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // add by myself
    float x = *dlowerX + (blockIdx.x * blockDim.x + threadIdx.x) * *dstepX;
    float y = *dlowerY + (blockIdx.y * blockDim.y + threadIdx.y) * *dstepY;
    //
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // add by myself
    int *temp_himg = (int*) malloc(sizeof(int) * resX * resY), *dimg;
    float *dlowerX, *dlowerY, *dstepX, *dstepY;
    cudaMalloc(&dimg, sizeof(int) * resX * resY);
    cudaMalloc(&dlowerX, sizeof(float));
    cudaMalloc(&dlowerY, sizeof(float));
    cudaMalloc(&dstepX, sizeof(float));
    cudaMalloc(&dstepY, sizeof(float));
    cudaMemcpy(dlowerX, &lowerX, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dlowerY, &lowerY, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dstepX, &stepX, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dstepY, &stepY, sizeof(float), cudaMemcpyHostToDevice);



    cudaMemcpy(dimg, temp_himg, sizeof(int) * resX * resY, cudaMemcpyDeviceToHost);
    cudaFree(dimg); cudaFree(dlowerX); cudaFree(dlowerY); cudaFree(dstepX); cudaFree(dstepY);
    //
}
