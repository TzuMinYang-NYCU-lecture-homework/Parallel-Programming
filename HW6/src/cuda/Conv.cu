#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "helper.h"
#include "bmpfuncs.h"

#define BLOCK_WITDH 1

__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    int halffilterSize = filterWidth / 2;
    float sum = 0; // Reset sum for new source pixel

    // get index
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;  
    printf("%d %d\n", row, col);
    if (row == 100 && col == 100)
    {
        printf("%d %d %d\n", filterWidth, imageHeight, imageWidth);
        printf("filter\n");
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
                printf("%f ", filter[i * 3 + j]);
            printf("\n");
        }
        printf("input\n");
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 10; ++j)
                printf("%f ", inputImage[i * imageWidth + j]);
            printf("\n");
        }
    }

    // compute boundary
    int k_start = row - halffilterSize >= 0 ? -halffilterSize : 0, k_end = halffilterSize + row < imageHeight ? halffilterSize : halffilterSize + row - imageHeight - 1;
    int l_start = col - halffilterSize >= 0 ? -halffilterSize : 0, l_end = halffilterSize + col < imageWidth ? halffilterSize : halffilterSize + col - imageWidth - 1;

    // apply filter
    for (int k = k_start; k <= k_end; ++k)
    {
        int idxImage = (row + k) * imageWidth + col, idxFilter = (k + halffilterSize) * filterWidth + halffilterSize;
        for (int l = l_start; l <= l_end; ++l) 
            sum += inputImage[idxImage + l] * filter[idxFilter + l];        
    }

    // assign output
    outputImage[row * imageWidth + col] = sum;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    // add by myself
    // declare gpu var.
    size_t image_size = sizeof(float) * imageHeight * imageWidth;
    float *dinput, *doutput;

    // allocate gpu memory
    cudaMalloc(&dinput, image_size);
    cudaMalloc(&doutput, image_size);

    // copy input from host to gpu
    cudaMemcpy(dinput, inputImage, image_size, cudaMemcpyHostToDevice);

    // cut filter
    int filterCutSzie = 0;
    int canCut = 1;
    do
    {
        int checkEnd = (filterWidth - filterCutSzie - 1);
        // up
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * filterCutSzie + i] != 0)
                canCut = 0;

        // down
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * checkEnd + i] != 0)
                canCut = 0;

        // left
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * i + filterCutSzie] != 0)
                canCut = 0;

        // right
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * i + checkEnd] != 0)
                canCut = 0;
    } while (canCut && ++filterCutSzie && filterCutSzie <= filterWidth / 2);

    int cuttedFilterWidth = filterWidth - filterCutSzie * 2;
    float *cuttedFilter = (float*) malloc(sizeof(float) * cuttedFilterWidth * cuttedFilterWidth);
    size_t cuttedFilterSize = sizeof(float) * cuttedFilterWidth * cuttedFilterWidth;
    for (int i = 0 ; i < cuttedFilterWidth; ++i)
        for (int j = 0; j < cuttedFilterWidth; ++j)
            cuttedFilter[cuttedFilterWidth * i + j] = filter[filterWidth * (i + filterCutSzie) + (j + filterCutSzie)];

    // copy filter from host to gpu
    float *dfilter;
    cudaMalloc(&dfilter, cuttedFilterSize);
    cudaMemcpy(dfilter, cuttedFilter, cuttedFilterSize, cudaMemcpyHostToDevice);

    // call gpu kernel func.
    dim3 dimGrid(imageWidth / BLOCK_WITDH, imageHeight / BLOCK_WITDH), imBlock(BLOCK_WITDH, BLOCK_WITDH);
    convolution<<<dimGrid, imBlock>>>(filterWidth, dfilter, imageHeight, imageWidth, inputImage, doutput);

    // copy ans from gpu to host
    cudaMemcpy(outputImage, doutput, image_size, cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dfilter);
    //
}
