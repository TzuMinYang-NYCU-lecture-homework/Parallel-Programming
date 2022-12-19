#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WITDH 8
#define TILE_WIDTH 2

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

__global__ void mandelKernel(size_t pitch, int* dimg, float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // add by myself
    // indexing, x is horizontal, y is vertical
    int i_start = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_WIDTH, j_start = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_WIDTH;

    for (int j = j_start; j < j_start + TILE_WIDTH; ++j)
    {
        for (int i = i_start; i < i_start + TILE_WIDTH; ++i)
        {
            float x = lowerX + i * stepX;
            float y = lowerY + j * stepY;
            // use this index because pitch memory
            // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
            *((int*)((char*)dimg + j * pitch) + i) = mandel(x, y, maxIterations);
        }
    }
    //
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // add by myself
    int *temp_himg; // because of hw require

    // allocate pinned-page host memory (because of hw require)
    // __host__​cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )
    cudaHostAlloc((void**) &temp_himg, sizeof(int) * resX * resY, cudaHostAllocDefault);

    // declare gpu var.
    int *dimg;

    // allocate pitch gpu memory (because of hw require), it will align 256 or 512, fast for hardware, useful for 2D or 3D picture
    size_t pitch;
    // __host__​cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )
    cudaMallocPitch((void**) &dimg, &pitch, sizeof(int) * resX, resY);

    // call gpu kernel func.
    dim3 dimGrid(resX / BLOCK_WITDH / TILE_WIDTH, resY / BLOCK_WITDH / TILE_WIDTH), dimBlock(BLOCK_WITDH, BLOCK_WITDH); // block num(dimGrid) should reduce because of tile, one thread will deal with a tile of pixels
    mandelKernel<<<dimGrid, dimBlock>>>(pitch, dimg, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    // copy ans from gpu to host, pitch memory can't use "cudaMemcpy"
    // __host__ cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    cudaMemcpy2D(temp_himg, sizeof(int) * resX, dimg, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);

    // copy data to result (because of hw require)
    memcpy(img, temp_himg, sizeof(int) * resX * resY);

    // free host&gpu memory
    cudaFreeHost(temp_himg);
    cudaFree(dimg);
    //
}
