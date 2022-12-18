#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WITDH 16

// add by myself
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

__global__ void mandelKernel(size_t* dpitch, float* dlowerX, float* dlowerY, float* dstepX, float* dstepY, int* dresX, int* dimg, int* dmaxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // add by myself
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;

    float x = *dlowerX + i * *dstepX;
    float y = *dlowerY + j * *dstepY;
    // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
    *((int*)((char*)dimg + j * *dpitch) + i) = mandel(x, y, *dmaxIterations);
    //
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // add by myself
    int *temp_himg;
    // __host__​cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )
    cudaHostAlloc((void**) &temp_himg, sizeof(int) * resX * resY, cudaHostAllocDefault);

    int *dimg, *dmaxIterations, *dresX;
    float *dlowerX, *dlowerY, *dstepX, *dstepY;

    size_t pitch, *dpitch;
    // __host__​cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )
    cudaMallocPitch((void**) &dimg, &pitch, sizeof(int) * resX, resY);
    cudaMalloc(&dpitch, sizeof(size_t));
    cudaMalloc(&dlowerX, sizeof(float));
    cudaMalloc(&dlowerY, sizeof(float));
    cudaMalloc(&dstepX, sizeof(float));
    cudaMalloc(&dstepY, sizeof(int));
    cudaMalloc(&dstepY, sizeof(int));
    cudaMalloc(&dresX, sizeof(int));
    cudaMalloc(&dmaxIterations, sizeof(int));

    cudaMemcpy(dpitch, &pitch, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dlowerX, &lowerX, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dlowerY, &lowerY, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dstepX, &stepX, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dstepY, &stepY, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dresX, &resX, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dmaxIterations, &maxIterations, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(resX / BLOCK_WITDH, resY / BLOCK_WITDH), dimBlock(BLOCK_WITDH, BLOCK_WITDH);
    mandelKernel<<<dimGrid, dimBlock>>>(dpitch, dlowerX, dlowerY, dstepX, dstepY, dresX, dimg, dmaxIterations);

    // __host__ cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    cudaMemcpy2D(temp_himg, sizeof(int) * resX, dimg, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);

    memcpy(img, temp_himg, sizeof(int) * resX * resY);

    cudaFreeHost(temp_himg);
    cudaFree(dimg); cudaFree(dlowerX); cudaFree(dlowerY); 
    cudaFree(dstepX); cudaFree(dstepY); cudaFree(dresX); cudaFree(dmaxIterations);
    //
}
